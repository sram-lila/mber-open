# mber-vhh CLI Features

## Disable AlphaFold Templates (`--no-templates`)

Disables AlphaFold template features during both design (trajectory) and evaluation forward passes. When templates are disabled, the model runs purely from sequence — the template embedding has no structural prior influence on the Evoformer.

### Usage

```bash
mber-vhh --input-pdb /path/to/target.pdb --output-dir /path/to/out --chains A --no-templates
```

Or via a settings YAML/JSON file:

```yaml
model:
  use_templates: false
```

### How it works

ColabDesign's forward-pass closure checks `_args["use_templates"]` on every call. When `True`, it calls `_update_template()` which populates template features (coordinates, aatype, masks) from the input PDB and sets `template_mask[0] = 1`. When `False`, `_update_template()` is skipped — template features stay zeroed and `template_mask` stays at `0`, so the template stack in the Evoformer receives no signal.

For multimer models (used by VHH), the flag has no effect on model initialization — it only controls runtime behavior.

---

## Warm-Start from Previous Run (`--warm-start`)

Re-uses sequences from a previous run as starting points for new trajectories, with optionally longer optimization. This is useful when a previous run produced "semi-successful" designs (reasonable iPTM/pLDDT but below acceptance thresholds) that could improve with continued optimization from a better starting point.

### Usage

Point at a CSV from a previous run (local path or S3):

```bash
mber-vhh --input-pdb target.pdb --output-dir /path/to/out --chains A \
          --warm-start s3://bucket/previous_run/all_trajectories.csv \
          --warm-start-iters-multiplier 2.0
```

Or point at a previous output directory:

```bash
mber-vhh --input-pdb target.pdb --output-dir /path/to/out --chains A \
          --warm-start /path/to/previous/output/
```

Or via settings YAML/JSON:

```yaml
warm_start:
  dir: s3://bucket/previous_run/all_trajectories.csv
  iters_multiplier: 2.0
```

### Accepted sources for `--warm-start`

| Source | Format |
|--------|--------|
| Local CSV | `/path/to/all_trajectories.csv` or `/path/to/accepted.csv` |
| S3 CSV | `s3://bucket/path/all_trajectories.csv` |
| Local output directory | `/path/to/output/` (scans `runs/*/evaluation_data/evaluation_data.json`) |

Any CSV with `binder_seq` and `i_ptm` columns works. The `plddt` column is optional.

### `--warm-start-iters-multiplier`

Multiplies trajectory iteration counts (soft, temp, hard) when warm-starting. Default: **2.0** (doubles the trajectory length). Set to `1.0` to keep the same trajectory length.

| Phase | Default iters | At 2.0x |
|-------|--------------|---------|
| Soft  | 65           | 130     |
| Temp  | 25           | 50      |
| Hard  | 0            | 0       |

### How it works

1. **Seed loading**: Scans the source for binder sequences with iPTM/pLDDT scores. Deduplicates by sequence.
2. **Semi-successful filtering**: Keeps sequences that failed at least one acceptance threshold (iPTM or pLDDT). Falls back to all candidates if none are semi-successful.
3. **Sorting**: Seeds are sorted by iPTM descending, so the most promising candidates are tried first.
4. **Cycling**: Trajectories cycle through seed sequences round-robin. With 5 seeds and 20 trajectories, each seed is tried 4 times (each with a different random seed for the optimizer).
5. **Initialization**: Instead of starting from the PLM-derived probability distribution, `set_seq()` is called with the actual amino acid string. The bias matrix still applies.
6. **Extended optimization**: Iteration counts are multiplied by `--warm-start-iters-multiplier`, giving the optimizer more time to refine the starting sequence.

### Flyte / Hydra usage

In a Hydra settings YAML for Flyte:

```yaml
warm_start_source: s3://lila-fl97-projects/lila-bindaid/executions_mber/previous_run/all_trajectories.csv
warm_start_iters_multiplier: 2.0
```

The Flyte task downloads the CSV from S3 using `FlyteFile` before passing it to the `mber-vhh` subprocess. No AWS CLI is needed inside the container.

---

## Full CLI Reference

```
mber-vhh [options]

Required (if no --settings):
  --input-pdb PATH          Target PDB (local path, PDB code, UniProt ID, or s3://)
  --output-dir DIR          Output directory
  --chains CHAINS           Target chains (e.g. 'A' or 'A,B')

Optional:
  --settings, -s PATH       YAML or JSON settings file (overrides all flags)
  --target-name NAME        Target name (default: PDB filename stem)
  --hotspots HOTSPOTS       Hotspot residues (e.g. 'A56' or 'A56,B20')
  --num-accepted N          Desired accepted designs (default: 100)
  --max-trajectories N      Max trajectories to attempt (default: 10000)
  --min-iptm FLOAT          Minimum iPTM to accept (default: 0.75)
  --min-plddt FLOAT         Minimum pLDDT to accept (default: 0.70)
  --warm-start PATH         CSV or directory from a previous run for warm-starting
  --warm-start-iters-multiplier FLOAT
                            Trajectory iteration multiplier (default: 2.0)
  --no-templates            Disable AlphaFold template features
  --no-animations           Skip saving trajectory animation HTML files
  --no-pickle               Skip saving design_state.pickle files
  --no-png                  Skip saving PNG plots
  --interactive             Prompt for values interactively
```
