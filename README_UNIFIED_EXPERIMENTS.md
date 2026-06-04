# Unified experiment runner

This cleaned version keeps **one experiment entry point only**:

```bash
python run_all_experiments.py --models all --encodings all --datasets repo_balanced_42
```

The old files under `experiments/`, `examples/`, and `configs/` were removed to avoid running different benchmark scripts with different defaults.

## Recommended quick run

```bash
python run_all_experiments.py \
  --models all \
  --encodings all \
  --datasets repo_balanced_42 \
  --max-train 64 \
  --max-test 32 \
  --epochs 5 \
  --qubits 6 \
  --layers 1 \
  --out results/all_models_all_encodings.csv
```

## Full feature-sweep run

This reproduces the style of the review CSV with all canonical datasets, all model variants, and all encoding families:

```bash
python run_all_experiments.py \
  --datasets all \
  --models all \
  --encodings all \
  --max-train 64 \
  --max-test 32 \
  --epochs 5 \
  --qubits 6 \
  --layers 1 \
  --out results/full_feature_sweep.csv
```

## Lists

```bash
python run_all_experiments.py --list
```

## Included model variants

- `qnn`
- `qcnn`
- `qsvm`
- `qknn`
- `qdt_axis`
- `qdt_entangled`
- `qdt_stump`
- `qdt_cmtsd`

`qdt_cmtsd` requires an encoding with a `_project` margin adapter, so it works with `ha_sage` and `ha_sage_cmtsd` and is skipped for unrelated encodings.
`--qdt-feature-mode auto` keeps ordinary QDT variants on probability features and enables the hybrid `both` mode for `qdt_entangled` with HA-SAGE-CMTSD-compatible encodings.

## Included encoding families

- `ha_sage`
- `ha_sage_cmtsd`
- `hardware_aware`
- `amplitude`
- `histogram`
- `sparse_amplitude`
- `angle`
- `denseangle`
- `reuploading`
- `havlicek`
- `hamiltonian`
- `trainable_kernel`
- `zzprod`
- `basis`
- `integer`
- `onehot`

Aliases such as `iqp`/`zz_feature_map` are not repeated because they point to the same implementation as `havlicek`.

`integer` and `onehot` are reported as theoretical resource-only encodings. The runner records their encoder widths from the resource table (`integer = 2 * n_features`, `onehot = 4 * n_features`), then skips circuit simulation instead of hashing them down to a smaller circuit.

## Output

The CSV includes accuracy, balanced accuracy, F1, recalls, eval counts, timing columns, circuit resource columns, parameters JSON, tags, extras JSON, and errors. Failures are recorded as rows instead of stopping the whole sweep unless you pass `--fail-fast`.


## HA-SAGE v8 advanced run

Use `README_HA_SAGE_V8_ADVANCED.md` for the upgraded SR-HA-SAGE and RC-CMTSD controls. The canonical run is:

```bash
python run_all_experiments.py --datasets all --models all --encodings hardware_aware,ha_sage,ha_sage_cmtsd --max-train 64 --max-test 32 --qubits 6 --layers 1 --out results/ha_sage_v8_advanced_full_feature_sweep.csv
```
