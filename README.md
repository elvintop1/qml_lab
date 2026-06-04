# QML Lab

Experimental quantum machine learning benchmark code for comparing classical
feature encodings, quantum encodings, kernel models, neural quantum models, and
quantum decision tree variants on small tabular and vision-derived datasets.

The main entry point is:

```bash
python run_all_experiments.py
```

## Repository Layout

```text
baselines/          Classical and deep-learning baselines
datasets/           Built-in tabular datasets and feature-sweep loaders
encodings/          Quantum feature maps and encoding registry
features/           Feature embedding utilities
kernels/            Quantum kernel estimators and post-processing
models/             QNN, QCNN, QSVM, QKNN, QDT, and related model code
run_all_experiments.py
                    Unified experiment runner
requirements_phase2.txt
                    Python dependencies
```

Generated outputs are intentionally ignored by Git:

```text
results/
outputs/
```

Private/local encoding work is also ignored and should not be pushed:

```text
encodings/ha_sage*.py
encodings/hardware_aware.py
README_HA_SAGE*.md
R_HASAGE*.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements_phase2.txt
```

If you run commands from outside this folder, make sure the parent directory is
on `PYTHONPATH` so imports such as `qml_lab.models` resolve correctly.

## List Available Options

```bash
python run_all_experiments.py --list
```

The runner supports these model families:

```text
qnn, qcnn, qsvm, qknn,
qdt_axis, qdt_entangled, qdt_stump, qdt_cmtsd
```

Public-safe encodings in the repository include:

```text
amplitude, histogram, sparse_amplitude,
angle, denseangle, reuploading,
havlicek, hamiltonian, trainable_kernel,
zzprod, basis, integer, onehot
```

Local/private encodings may appear in the runner if the ignored local files are
present:

```text
ha_sage, ha_sage_cmtsd, hardware_aware
```

## Quick Public-Safe Run

This avoids local/private encoding files and writes results into the ignored
`results/` folder:

```bash
python run_all_experiments.py ^
  --datasets repo_balanced_42 ^
  --models qnn,qcnn,qsvm,qknn,qdt_axis,qdt_entangled,qdt_stump ^
  --encodings amplitude,histogram,sparse_amplitude,angle,denseangle,reuploading,havlicek,hamiltonian,trainable_kernel,zzprod,basis,integer,onehot ^
  --max-train 64 ^
  --max-test 32 ^
  --epochs 5 ^
  --out results/public_safe_smoke.csv
```

## Full Public-Safe Feature Sweep

```bash
python run_all_experiments.py ^
  --datasets all ^
  --models qnn,qcnn,qsvm,qknn,qdt_axis,qdt_entangled,qdt_stump ^
  --encodings amplitude,histogram,sparse_amplitude,angle,denseangle,reuploading,havlicek,hamiltonian,trainable_kernel,zzprod,basis,integer,onehot ^
  --max-train 64 ^
  --max-test 32 ^
  --epochs 5 ^
  --out results/public_safe_feature_sweep.csv
```

## Local Private Encoding Runs

If your local ignored encoding files are present, you can run the private
encoding family too:

```bash
python run_all_experiments.py ^
  --datasets mnist_36_16pca,cifar10_01_16pca ^
  --models all ^
  --encodings all ^
  --qubits 16 ^
  --layers 10 ^
  --max-train 64 ^
  --max-test 32 ^
  --out results/mnist_cifar_private_full.csv
```

Keep the output under `results/`; it is ignored and should remain local.

## Datasets

The default feature-sweep datasets include:

```text
repo_balanced_42, iris_4, synthetic_8, wine_13,
synthetic_16, breast_cancer_30, digits_32var, synthetic_32
```

Additional loaders may be available locally:

```text
mnist_36_16pca, mnist_36_32pca, cifar10_01_16pca,
synthetic_1m_8, synthetic_1m_32, poker_hand_1m
```

You can also pass a CSV dataset:

```bash
python run_all_experiments.py --datasets csv:path\to\data.csv --label-column label
```

## Resource Accounting Notes

The runner records circuit resource columns such as qubit counts, depth, size,
and two-qubit gate counts where available.

For discrete basis encodings:

```text
integer = 2 * n_features qubits
onehot  = 4 * n_features qubits
```

These two encodings are reported as resource-only rows and skipped for faithful
circuit simulation.

For PCA16-style datasets:

```text
angle        = 16 qubits
denseangle   = 8 qubits
amplitude    = 4 qubits
reuploading  = 16 qubits when --qubits 16 is used
integer      = 32 qubits, skipped
onehot       = 64 qubits, skipped
```

## Git Hygiene

Do not commit generated results or local/private encoding files. The repository
history has been cleaned so the public `main` branch contains only the current
clean tree.

Before pushing, check:

```bash
git status --short
git ls-files results
```

`git ls-files results` should print nothing.

