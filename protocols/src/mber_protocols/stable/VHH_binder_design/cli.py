import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

from mber_protocols.stable.VHH_binder_design.config import (
    TemplateConfig,
    ModelConfig,
    LossConfig,
    TrajectoryConfig,
    EnvironmentConfig,
    EvaluationConfig,
)
from mber_protocols.stable.VHH_binder_design.template import TemplateModule
from mber_protocols.stable.VHH_binder_design.trajectory import TrajectoryModule
from mber_protocols.stable.VHH_binder_design.evaluation import EvaluationModule
from mber_protocols.stable.VHH_binder_design.state import DesignState, TemplateData


DEFAULT_MASKED_VHH = (
    "EVQLVESGGGLVQPGGSLRLSCAASG*********WFRQAPGKEREF***********NADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYC************WGQGTLVTVSS"
)

DEFAULT_NUM_ACCEPTED = 100
DEFAULT_MAX_TRAJ = 10000
DEFAULT_MIN_IPTM = 0.75
DEFAULT_MIN_PLDDT = 0.70


def _load_settings(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        text = f.read()
    if (path.endswith(".yml") or path.endswith(".yaml")) and _HAS_YAML:
        return yaml.safe_load(text)
    return json.loads(text)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_hotspots(hotspots: Optional[Any]) -> Optional[str]:
    if hotspots is None:
        return None
    if isinstance(hotspots, str):
        if not hotspots.strip():
            return None
        return ",".join([h.strip() for h in hotspots.split(",") if h.strip()])
    if isinstance(hotspots, (list, tuple)):
        return ",".join([str(h).strip() for h in hotspots if str(h).strip()])
    return None


def _write_csv_header(csv_path: str) -> None:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trajectory_name",
            "binder_index",
            "binder_seq",
            "i_ptm",
            "plddt",
            "ptm",
            "complex_pdb_path",
            "relaxed_pdb_path",
        ])


def _count_existing_accepted(csv_path: str) -> int:
    """Count the number of existing accepted designs in the CSV."""
    if not os.path.exists(csv_path):
        return 0
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            # Skip header
            next(reader, None)
            # Count rows
            return sum(1 for _ in reader)
    except Exception:
        return 0


def _save_accepted_pdbs(
    output_dir: str,
    trajectory_name: str,
    binder_index: int,
    i_ptm: float,
    complex_pdb: Optional[str],
    relaxed_pdb: Optional[str],
) -> Dict[str, Optional[str]]:
    accepted_dir = os.path.join(output_dir, "Accepted")
    _ensure_dir(accepted_dir)

    saved: Dict[str, Optional[str]] = {"complex": None, "relaxed": None}

    if complex_pdb:
        fname = f"{trajectory_name}_binder-{binder_index}_iptm-{i_ptm:.4f}_complex.pdb"
        path = os.path.join(accepted_dir, fname)
        with open(path, "w") as f:
            f.write(complex_pdb)
        saved["complex"] = path

    if relaxed_pdb:
        fname = f"{trajectory_name}_binder-{binder_index}_iptm-{i_ptm:.4f}_relaxed.pdb"
        path = os.path.join(accepted_dir, fname)
        with open(path, "w") as f:
            f.write(relaxed_pdb)
        saved["relaxed"] = path

    return saved


def _append_csv(
    csv_path: str,
    trajectory_name: str,
    binder_index: int,
    binder_seq: Optional[str],
    i_ptm: Optional[float],
    plddt: Optional[float],
    ptm: Optional[float],
    complex_path: Optional[str],
    relaxed_path: Optional[str],
) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            trajectory_name,
            binder_index,
            binder_seq or "",
            f"{i_ptm:.4f}" if isinstance(i_ptm, float) else "",
            f"{plddt:.4f}" if isinstance(plddt, float) else "",
            f"{ptm:.4f}" if isinstance(ptm, float) else "",
            complex_path or "",
            relaxed_path or "",
        ])


DEFAULT_WARM_START_ITERS_MULTIPLIER = 2.0


def _resolve_path(path: str) -> str:
    """Download S3 paths to a local temp file; return local paths unchanged."""
    if not path.startswith("s3://"):
        return path
    import subprocess
    import tempfile
    suffix = Path(path).suffix or ".csv"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    print(f"[warm-start] Downloading {path} → {tmp.name}")
    subprocess.check_call(["aws", "s3", "cp", path, tmp.name], stdout=subprocess.DEVNULL)
    return tmp.name


def _load_seed_sequences_from_csv(
    csv_path: str,
    min_iptm: float,
    min_plddt: float,
) -> List[Tuple[str, float, float]]:
    """Parse a CSV (accepted.csv or all_trajectories.csv) for seed sequences.

    Expects columns ``binder_seq``, ``i_ptm``, and optionally ``plddt``.
    """
    candidates: List[Tuple[str, float, float]] = []
    seen_seqs: set = set()

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = (row.get("binder_seq") or "").strip()
            raw_iptm = row.get("i_ptm", "")
            raw_plddt = row.get("plddt", "")
            if not seq or not raw_iptm:
                continue
            try:
                iptm = float(raw_iptm)
            except (ValueError, TypeError):
                continue
            try:
                plddt = float(raw_plddt) if raw_plddt else 0.0
            except (ValueError, TypeError):
                plddt = 0.0
            if seq in seen_seqs:
                continue
            seen_seqs.add(seq)
            candidates.append((seq, iptm, plddt))

    return candidates


def _load_seed_sequences_from_dir(
    run_dir: str,
) -> List[Tuple[str, float, float]]:
    """Scan ``runs/*/evaluation_data/evaluation_data.json`` for binder entries."""
    runs_dir = os.path.join(run_dir, "runs")
    if not os.path.isdir(runs_dir):
        print(f"[warm-start] No runs/ directory in {run_dir}", file=sys.stderr)
        return []

    candidates: List[Tuple[str, float, float]] = []
    seen_seqs: set = set()

    for traj_dir in sorted(Path(runs_dir).iterdir()):
        eval_json = traj_dir / "evaluation_data" / "evaluation_data.json"
        if not eval_json.exists():
            continue
        try:
            data = json.loads(eval_json.read_text())
            binders = data.get("binders", []) if isinstance(data, dict) else data
            if not isinstance(binders, list):
                binders = [binders]
            for b in binders:
                seq = b.get("binder_seq")
                iptm = b.get("i_ptm")
                plddt = b.get("plddt")
                if not seq or iptm is None:
                    continue
                if seq in seen_seqs:
                    continue
                seen_seqs.add(seq)
                candidates.append((seq, float(iptm), float(plddt or 0.0)))
        except Exception as exc:
            print(f"[warm-start] Skipping {eval_json}: {exc}", file=sys.stderr)

    return candidates


def _load_seed_sequences(
    warm_start_source: str,
    min_iptm: float = 0.0,
    min_plddt: float = 0.0,
) -> List[Tuple[str, float, float]]:
    """Load evaluated binder sequences from a previous run.

    ``warm_start_source`` can be:
      - A CSV file path (local or ``s3://``) with ``binder_seq`` / ``i_ptm`` columns
      - A local output directory containing ``runs/*/evaluation_data/``

    Returns ``(sequence, iPTM, pLDDT)`` tuples sorted by iPTM descending,
    filtered to semi-successful candidates (below acceptance thresholds).
    """
    local_path = _resolve_path(warm_start_source)

    if local_path.endswith(".csv"):
        candidates = _load_seed_sequences_from_csv(local_path, min_iptm, min_plddt)
    elif os.path.isdir(local_path):
        candidates = _load_seed_sequences_from_dir(local_path)
    else:
        print(f"[warm-start] {warm_start_source} is not a CSV or directory", file=sys.stderr)
        return []

    # Sort by iPTM descending (best candidates first)
    candidates.sort(key=lambda t: t[1], reverse=True)

    # Filter: keep sequences that didn't pass BOTH acceptance thresholds
    semi = [c for c in candidates if c[1] < min_iptm or c[2] < min_plddt]
    if not semi:
        semi = candidates

    if semi:
        print(
            f"[warm-start] Loaded {len(semi)} seed sequences from {warm_start_source} "
            f"(iPTM range {semi[-1][1]:.4f} – {semi[0][1]:.4f})"
        )
    else:
        print(f"[warm-start] No seed sequences found in {warm_start_source}")
    return semi


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "mber-vhh",
        description=(
            "Run VHH binder design using a simple settings file or flags.\n"
            "Required flags (if no --settings): --input-pdb, --output-dir, --chains.\n"
            f"Defaults: num-accepted={DEFAULT_NUM_ACCEPTED}, max-trajectories={DEFAULT_MAX_TRAJ}, min-iptm={DEFAULT_MIN_IPTM}, min-plddt={DEFAULT_MIN_PLDDT}.\n"
            "Examples:\n"
            "  mber-vhh --settings /path/settings.yml\n"
            "  mber-vhh --input-pdb /abs/PDL1.pdb --output-dir /abs/out --chains A --hotspots A56 --num-accepted 50\n"
            "  mber-vhh --interactive\n"
        ),
    )
    p.add_argument("--settings", "-s", help="Path to YAML or JSON settings file")
    p.add_argument("--input-pdb", help="Path/ID of target (local PDB, PDB code, UniProt ID, or s3://...")
    p.add_argument("--output-dir", help="Output directory")
    p.add_argument("--target-name", help="Optional target name (defaults to PDB filename stem)", default=None)
    p.add_argument("--chains", help="Target chains, e.g. 'A' or 'A,B'")
    p.add_argument("--hotspots", help="Optional hotspots, e.g. 'A56' or 'A56,B20'", default=None)
    p.add_argument("--num-accepted", type=int, default=DEFAULT_NUM_ACCEPTED, help=f"Desired number of accepted designs (default {DEFAULT_NUM_ACCEPTED})")
    p.add_argument("--max-trajectories", type=int, default=DEFAULT_MAX_TRAJ, help=f"Max trajectories to attempt (default {DEFAULT_MAX_TRAJ})")
    p.add_argument("--min-iptm", type=float, default=DEFAULT_MIN_IPTM, help=f"Minimum iPTM to accept (default {DEFAULT_MIN_IPTM})")
    p.add_argument("--min-plddt", type=float, default=DEFAULT_MIN_PLDDT, help=f"Minimum pLDDT to accept (default {DEFAULT_MIN_PLDDT})")
    p.add_argument("--warm-start", metavar="PATH", default=None,
                   help="CSV file (local or s3://) or output directory from a previous run; loads semi-successful sequences as starting points")
    p.add_argument("--warm-start-iters-multiplier", type=float, default=DEFAULT_WARM_START_ITERS_MULTIPLIER,
                   help=f"Multiply trajectory iterations when warm-starting (default {DEFAULT_WARM_START_ITERS_MULTIPLIER})")
    p.add_argument("--no-templates", action="store_true", help="Disable AlphaFold template features during design and evaluation")
    p.add_argument("--no-animations", action="store_true", help="Skip saving animated trajectory HTML files to save space")
    p.add_argument("--no-pickle", action="store_true", help="Skip saving design_state.pickle files to save space")
    p.add_argument("--no-png", action="store_true", help="Skip saving PNG plots (e.g., pssm_logits.png) to save space")
    p.add_argument("--interactive", action="store_true", help="Prompt for required values with defaults shown")
    return p.parse_args()


def _prompt(text: str, default: Optional[str] = None) -> str:
    hint = f" [{default}]" if default is not None else ""
    resp = input(f"{text}{hint}: ").strip()
    return resp if resp else (default or "")


def _isatty_stdin() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _collect_interactive() -> Dict[str, Any]:
    print("\nMBER VHH Interactive Setup\n")
    input_pdb = _prompt("Enter path/ID to target PDB", None)
    while not input_pdb:
        input_pdb = _prompt("Enter path/ID to target PDB", None)

    output_dir = _prompt("Enter output directory", os.path.join(os.getcwd(), "mber_vhh_out"))
    target_name = _prompt("Enter target name [leave empty to use PDB filename]", "")
    chains = _prompt("Enter target chains (e.g., A or A,B)", "A")
    hotspots = _prompt("Enter hotspot(s) (e.g., A56 or A56,B20) [leave empty for none]", "")
    num_accepted = _prompt(f"Desired number of accepted designs (default {DEFAULT_NUM_ACCEPTED})", str(DEFAULT_NUM_ACCEPTED))
    max_traj = _prompt(f"Maximum number of trajectories (default {DEFAULT_MAX_TRAJ})", str(DEFAULT_MAX_TRAJ))
    min_iptm = _prompt(f"Minimum iPTM threshold (default {DEFAULT_MIN_IPTM})", str(DEFAULT_MIN_IPTM))
    min_plddt = _prompt(f"Minimum pLDDT threshold (default {DEFAULT_MIN_PLDDT})", str(DEFAULT_MIN_PLDDT))
    
    no_templates = _prompt("Disable AlphaFold templates? (y/N)", "N").lower() in ("y", "yes")
    skip_animations = _prompt("Skip saving animations? (y/N)", "N").lower() in ("y", "yes")
    skip_pickle = _prompt("Skip saving pickle files? (y/N)", "N").lower() in ("y", "yes")
    skip_png = _prompt("Skip saving PNG plots? (y/N)", "N").lower() in ("y", "yes")

    cfg: Dict[str, Any] = {
        "output": {"dir": output_dir, "skip_animations": skip_animations, "skip_pickle": skip_pickle, "skip_png": skip_png},
        "target": {"pdb": input_pdb, "name": target_name or None, "chains": chains, "hotspots": hotspots or None},
        "model": {"use_templates": not no_templates},
        "stopping": {"num_accepted": int(num_accepted), "max_trajectories": int(max_traj)},
        "filters": {"min_iptm": float(min_iptm), "min_plddt": float(min_plddt)},
    }
    return cfg


def _merge_flags_into_cfg(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if args.settings:
        return None
    if not (args.input_pdb and args.output_dir and args.chains):
        return None
    cfg: Dict[str, Any] = {
        "output": {
            "dir": args.output_dir,
            "skip_animations": args.no_animations,
            "skip_pickle": args.no_pickle,
            "skip_png": args.no_png,
        },
        "target": {
            "pdb": args.input_pdb,
            "name": args.target_name,
            "chains": args.chains,
            "hotspots": args.hotspots,
        },
        "model": {
            "use_templates": not args.no_templates,
        },
        "warm_start": {
            "dir": args.warm_start,
            "iters_multiplier": args.warm_start_iters_multiplier,
        },
        "stopping": {
            "num_accepted": int(args.num_accepted),
            "max_trajectories": int(args.max_trajectories),
        },
        "filters": {
            "min_iptm": float(args.min_iptm),
            "min_plddt": float(args.min_plddt),
        },
    }
    return cfg


def _load_cfg_from_settings(path: str) -> Dict[str, Any]:
    raw = _load_settings(path)
    return raw


def _build_state_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out_cfg = cfg["output"]
    out_dir = out_cfg["dir"]
    tgt = cfg["target"]
    binder = cfg.get("binder", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    warm_start_cfg = cfg.get("warm_start", {}) or {}
    stopping = cfg.get("stopping", {})
    filters = cfg.get("filters", {})

    chains_str = str(tgt.get("chains", "A"))
    region = chains_str  # region string supports "A" or "A,B"
    hotspots = _normalize_hotspots(tgt.get("hotspots"))

    num_accepted = int(stopping.get("num_accepted", DEFAULT_NUM_ACCEPTED))
    max_trajectories = int(stopping.get("max_trajectories", DEFAULT_MAX_TRAJ))
    min_iptm = float(filters.get("min_iptm", DEFAULT_MIN_IPTM))
    min_plddt = float(filters.get("min_plddt", DEFAULT_MIN_PLDDT))

    # Use provided target name or default to PDB filename stem
    target_name = tgt.get("name") or str(Path(str(tgt["pdb"]).split("/")[-1]).stem)
    
    # Use provided masked sequence or default
    masked_sequence = binder.get("masked_sequence") or DEFAULT_MASKED_VHH
    
    state = DesignState(
        template_data=TemplateData(
            target_id=str(tgt["pdb"]),
            target_name=target_name,
            masked_binder_seq=masked_sequence,
            region=region,
            target_hotspot_residues=hotspots,
        )
    )

    return {
        "output_dir": out_dir,
        "num_accepted": num_accepted,
        "max_trajectories": max_trajectories,
        "min_iptm": min_iptm,
        "min_plddt": min_plddt,
        "use_templates": model_cfg.get("use_templates", True),
        "warm_start_dir": warm_start_cfg.get("dir"),
        "warm_start_iters_multiplier": float(
            warm_start_cfg.get("iters_multiplier", DEFAULT_WARM_START_ITERS_MULTIPLIER)
        ),
        "skip_animations": out_cfg.get("skip_animations", False),
        "skip_pickle": out_cfg.get("skip_pickle", False),
        "skip_png": out_cfg.get("skip_png", False),
        "state": state,
    }


def _run_one_trajectory(
    state: DesignState,
    use_templates: bool = True,
    warm_start_iters_multiplier: float = 1.0,
) -> DesignState:
    template_cfg = TemplateConfig()
    model_cfg = ModelConfig(use_templates=use_templates)
    loss_cfg = LossConfig()
    traj_cfg = TrajectoryConfig(warm_start_iters_multiplier=warm_start_iters_multiplier)
    env_cfg = EnvironmentConfig()
    eval_cfg = EvaluationConfig()

    tmpl = TemplateModule(template_cfg, env_cfg)
    tmpl.setup(state)
    state = tmpl.run(state)
    tmpl.teardown(state)

    traj = TrajectoryModule(model_cfg, loss_cfg, traj_cfg, env_cfg)
    traj.setup(state)
    state = traj.run(state)
    traj.teardown(state)

    ev = EvaluationModule(model_cfg, loss_cfg, eval_cfg, env_cfg)
    ev.setup(state)
    state = ev.run(state)
    ev.teardown(state)

    return state


def _accept_binders(state: DesignState, min_iptm: float, min_plddt: float) -> Dict[int, Dict[str, Any]]:
    accepted: Dict[int, Dict[str, Any]] = {}
    binders = state.evaluation_data.binders or []
    for idx, b in enumerate(binders):
        try:
            b_i_ptm = getattr(b, "i_ptm", None)
            b_plddt = getattr(b, "plddt", None)
            if b_i_ptm is None or b_plddt is None:
                continue
            if float(b_i_ptm) >= min_iptm and float(b_plddt) >= min_plddt:
                accepted[idx] = {
                    "seq": getattr(b, "binder_seq", None),
                    "i_ptm": float(b_i_ptm),
                    "plddt": float(b_plddt),
                    "ptm": getattr(b, "ptm", None),
                    "complex_pdb": getattr(b, "complex_pdb", None),
                    "relaxed_pdb": getattr(b, "relaxed_pdb", None),
                }
        except Exception:
            continue
    return accepted


def _save_run_state(output_dir: str, state: DesignState, skip_animations: bool = False, skip_pickle: bool = False, skip_png: bool = False) -> str:
    traj_name = state.trajectory_data.trajectory_name or "trajectory"
    run_dir = os.path.join(output_dir, "runs", traj_name)
    _ensure_dir(run_dir)
    # Save the state with gating to avoid writing unwanted files
    state.to_dir(
        run_dir,
        save_pickle=not skip_pickle,
        save_png=not skip_png,
        save_animations=not skip_animations,
    )
    
    return run_dir


def main() -> None:
    args = _parse_args()

    if args.settings:
        cfg = _load_cfg_from_settings(args.settings)
    elif args.interactive:
        cfg = _collect_interactive()
    else:
        flags_cfg = _merge_flags_into_cfg(args)
        if flags_cfg is None and _isatty_stdin():
            print("\nMissing required flags and no --settings provided. Entering interactive mode...\n")
            cfg = _collect_interactive()
        elif flags_cfg is None:
            print(
                "Error: provide --settings PATH or use required flags --input-pdb, --output-dir, --chains (run in a TTY to be prompted).",
                file=sys.stderr,
            )
            sys.exit(2)
        else:
            cfg = flags_cfg

    built = _build_state_from_cfg(cfg)
    output_dir = built["output_dir"]
    num_accepted = built["num_accepted"]
    max_trajectories = built["max_trajectories"]
    min_iptm = built["min_iptm"]
    min_plddt = built["min_plddt"]
    use_templates = built["use_templates"]
    warm_start_dir = built["warm_start_dir"]
    warm_start_iters_multiplier = built["warm_start_iters_multiplier"]
    state: DesignState = built["state"]
    skip_animations = built["skip_animations"]
    skip_pickle = built["skip_pickle"]
    skip_png = built["skip_png"]

    # Load seed sequences from a previous run if warm-starting
    seed_sequences: List[str] = []
    if warm_start_dir:
        seed_entries = _load_seed_sequences(warm_start_dir, min_iptm=min_iptm, min_plddt=min_plddt)
        seed_sequences = [entry[0] for entry in seed_entries]
        if not seed_sequences:
            print("[warm-start] WARNING: no seed sequences found, falling back to normal design")
            warm_start_iters_multiplier = 1.0

    _ensure_dir(output_dir)
    accepted_csv = os.path.join(output_dir, "accepted.csv")
    _write_csv_header(accepted_csv)

    # Count existing accepted designs for resume support
    existing_accepted = _count_existing_accepted(accepted_csv)
    remaining_needed = max(0, num_accepted - existing_accepted)
    
    if existing_accepted > 0:
        print(f"Found {existing_accepted} existing accepted designs. Need {remaining_needed} more to reach target of {num_accepted}.")
    
    if remaining_needed == 0:
        print(f"Target of {num_accepted} accepted designs already reached. Exiting.")
        return

    ws_info = ""
    if seed_sequences:
        ws_info = f", warm_start={len(seed_sequences)} seeds, iters_multiplier={warm_start_iters_multiplier}"

    print(
        f"Running VHH design: output={output_dir}, chains={state.template_data.region}, hotspots={state.template_data.target_hotspot_residues or 'None'}, "
        f"target_accepted={num_accepted}, existing={existing_accepted}, remaining={remaining_needed}, "
        f"max_trajectories={max_trajectories} (default {DEFAULT_MAX_TRAJ}), "
        f"min_iptm={min_iptm} (default {DEFAULT_MIN_IPTM}), min_plddt={min_plddt} (default {DEFAULT_MIN_PLDDT}), "
        f"use_templates={use_templates}{ws_info}"
    )

    accepted_count = 0
    traj_count = 0

    while accepted_count < remaining_needed and traj_count < max_trajectories:
        traj_count += 1

        # When warm-starting, cycle through seed sequences then fall back to normal
        if seed_sequences:
            idx = (traj_count - 1) % len(seed_sequences)
            state.trajectory_data.seed_seq = seed_sequences[idx]
            iters_mult = warm_start_iters_multiplier
        else:
            state.trajectory_data.seed_seq = None
            iters_mult = 1.0

        run_state = _run_one_trajectory(
            state,
            use_templates=use_templates,
            warm_start_iters_multiplier=iters_mult,
        )

        run_dir = _save_run_state(
            output_dir,
            run_state,
            skip_animations=skip_animations,
            skip_pickle=skip_pickle,
            skip_png=skip_png,
        )
        traj_name = run_state.trajectory_data.trajectory_name or f"traj_{traj_count}"

        accepted = _accept_binders(run_state, min_iptm=min_iptm, min_plddt=min_plddt)

        for idx, info in accepted.items():
            saved_paths = _save_accepted_pdbs(
                output_dir=output_dir,
                trajectory_name=traj_name,
                binder_index=idx,
                i_ptm=float(info.get("i_ptm", 0.0)),
                complex_pdb=info.get("complex_pdb"),
                relaxed_pdb=info.get("relaxed_pdb"),
            )
            _append_csv(
                accepted_csv,
                trajectory_name=traj_name,
                binder_index=idx,
                binder_seq=info.get("seq"),
                i_ptm=info.get("i_ptm"),
                plddt=info.get("plddt"),
                ptm=info.get("ptm"),
                complex_path=saved_paths.get("complex"),
                relaxed_path=saved_paths.get("relaxed"),
            )
            accepted_count += 1
            if accepted_count >= remaining_needed:
                break

        # Prepare a fresh state for next trajectory (new seed handled internally)
        state = DesignState(template_data=run_state.template_data)

    total_accepted = existing_accepted + accepted_count
    print(f"Accepted designs: {accepted_count} new + {existing_accepted} existing = {total_accepted} total after {traj_count} trajectories. Results in {output_dir}")


if __name__ == "__main__":
    main()


