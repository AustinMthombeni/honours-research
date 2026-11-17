
# Running th notebook

This repository contains a notebook comparing:

- **MAP-Elites + ROSARL**
- **Plain MAP-Elites**

The experiments use QDax on the host Python environment and run the native Safety-Gymnasium / MuJoCo environments in a separate Python 3.10 worker process.

## 1. Recommended: run in Google Colab

The notebook has automatic setup for Colab, including installation of QDax, creation of the Python 3.10 worker environment, and installation of the pinned Safety-Gymnasium / MuJoCo dependencies.

1. Upload `MapElites vs MapELites+ROSARL.ipynb` to Google Colab, or place it in Google Drive and open it with Colab.
2. In **Runtime → Change runtime type**, select a runtime with sufficient memory. A GPU is not required by the notebook's setup.
3. Run the notebook from the first cell, in order. Do not skip the setup or worker-generation cells.
4. Wait for each cell to finish before running the next one. The environment-validation cell runs tests against all six environments before the training cell.
5. The final cells generate plots, summary tables, CSV files, and validation results.

The first setup cell detects Colab and installs the required packages. It also creates a Python 3.10 virtual environment under `/content/safe_qd_py310` for the native environment worker.

## 2. Run locally

The notebook does **not** automatically create or install the Python 3.10 worker environment outside Colab. You must prepare both Python environments before running it.

### 2.1 Host environment

Use a Python environment compatible with QDax 0.5.0. Install the host dependencies:

```bash
python -m pip install "qdax==0.5.0" "uv>=0.8" matplotlib pandas scipy
```

QDax will install its required JAX dependencies. If your platform requires a specific JAX build, install the appropriate JAX version/build for your machine before running the notebook.

### 2.2 Python 3.10 worker environment

Install `uv`, then create the worker environment and install the notebook's pinned native-environment dependencies.

**Linux / macOS:**

```bash
uv python install 3.10
uv venv --python 3.10 safe_qd_py310
uv pip install --python safe_qd_py310/bin/python --only-binary :all: \
  "numpy==1.23.5" \
  "gymnasium==0.28.1" \
  "gymnasium-robotics==1.2.2" \
  "mujoco==2.3.3" \
  "pygame==2.1.0" \
  "safety-gymnasium==1.0.0" \
  "xmltodict>=0.13" \
  "PyYAML>=6.0"
uv pip check --python safe_qd_py310/bin/python
```

**Windows PowerShell:**

```powershell
uv python install 3.10
uv venv --python 3.10 safe_qd_py310
uv pip install --python safe_qd_py310\Scripts\python.exe --only-binary :all: `
  "numpy==1.23.5" `
  "gymnasium==0.28.1" `
  "gymnasium-robotics==1.2.2" `
  "mujoco==2.3.3" `
  "pygame==2.1.0" `
  "safety-gymnasium==1.0.0" `
  "xmltodict>=0.13" `
  "PyYAML>=6.0"
uv pip check --python safe_qd_py310\Scripts\python.exe
```

### 2.3 Point the notebook to the worker interpreter

The notebook expects the worker environment at `safeqd_work/safe_qd_py310` relative to the current working directory by default. If you created `safe_qd_py310` elsewhere, set `SAFEQD_WORK_DIR` or `SAFEQD_WORKER_PYTHON` **before running the first notebook cell**.

For example, on Linux / macOS, if the notebook's working directory contains `safe_qd_py310`:

```python
import os
os.environ["SAFEQD_WORK_DIR"] = "."
os.environ["SAFEQD_WORKER_PYTHON"] = "./safe_qd_py310/bin/python"
```

On Windows:

```python
import os
os.environ["SAFEQD_WORK_DIR"] = "."
os.environ["SAFEQD_WORKER_PYTHON"] = r".\safe_qd_py310\Scripts\python.exe"
```

Run this configuration cell before the notebook's setup cell. The notebook writes the worker script into `SAFEQD_WORK_DIR` and launches it using `SAFEQD_WORKER_PYTHON`.

For headless Linux, MuJoCo may require a working OpenGL/EGL setup. The notebook sets `MUJOCO_GL` to `glfw` locally and to `egl` in Colab; change this only if your machine's graphics configuration requires it.

## 3. Run configuration

The notebook reads its run configuration from environment variables in the tasks / methods cell.

### Presets

| Preset | Generations | Population | Training scenes | Validation scenes | Held-out scenes | Horizon | Seeds |
|---|---:|---:|---:|---:|---:|---:|---|
| `smoke` | 3 | 6 | 2 | 2 | 4 | 100 | 1 |
| `pilot` | 30 | 24 | 5 | 6 | 12 | 300 | 1, 2, 3 |
| `research` | 100 | 32 | 8 | 8 | 32 | 500 | 1, 2, 3, 4, 5 |



Set a preset before running the notebook's setup and configuration cells:

```python
import os
os.environ["SAFEQD_PRESET"] = "pilot"  # smoke, pilot, or research
```

For a full research run:

```python
import os
os.environ["SAFEQD_PRESET"] = "research"
```

### Select tasks

By default, all six tasks are run:

- `Point Navigation`
- `Car Navigation`
- `Ant Navigation`
- `Ant SafeVelocity`
- `Humanoid SafeVelocity`
- `HalfCheetah SafeVelocity`

To run a subset, set `SAFEQD_TASKS` to a comma-separated list. The names must match the task names above exactly:

```python
import os
os.environ["SAFEQD_TASKS"] = "Point Navigation,Ant Navigation"
```

### Select seeds and workers

```python
import os
os.environ["SAFEQD_SEEDS"] = "1,2,3"
os.environ["SAFEQD_WORKERS"] = "2"
```

`SAFEQD_WORKERS` controls the number of native worker processes used for a task. Keep this modest if memory is limited.

### Override preset values

`SAFEQD_CFG_JSON` accepts a JSON object whose keys override values from the selected preset. For example:

```python
import os
os.environ["SAFEQD_PRESET"] = "smoke"
os.environ["SAFEQD_CFG_JSON"] = '{"generations": 5, "population": 8}'
```

Set configuration variables in a cell **before** the notebook's setup / configuration cells. The notebook reads them when those cells execute.

## 4. Notebook execution order

Run all cells from top to bottom:

1. **Colab setup / worker environment check** — detects the runtime and checks the Python 3.10 worker interpreter.
2. **Tasks, methods, presets** — imports QDax and configures tasks, seeds, preset values, and output directories.
3. **Native Safety-Gymnasium worker** — writes the worker program to disk.
4. **Worker pool and episode invariants** — defines the host-side worker interface and checks episode data.
5. **Candidate-specific ROSARL and scoring** — defines the candidate-specific penalty calculation and scoring logic.
6. **Validation A** — runs mathematical, isolation, and reward-replacement unit tests.
7. **QDax MAP-Elites and paired runs** — defines the MAP-Elites setup and training routines.
8. **Validation C** — tests the six real MuJoCo environments.
9. **Training** — runs both methods for each selected task and seed.
10. **Training dynamics** — creates training and penalty plots.
11. **Held-out evaluation and validation summary** — evaluates the selected policies on held-out scenes and writes the result files.
12. **Interpretation guide** — review the limitations before drawing conclusions.


## 5. Output files

The notebook creates a results directory named according to the selected preset:

```text
safeqd_work/
└── safe_qd_results_<preset>/
    ├── figures/
    │   ├── <task>_training.png
    │   ├── <task>_minmax.png
    │   └── <task>_heldout.png
    ├── generation_metrics.csv
    ├── heldout_episodes.csv
    ├── candidate_penalties.csv
    ├── validation_tests.csv
    └── heldout_summary.csv
```

In Colab, the default working directory is `/content`, so results are written under `/content/safe_qd_results_<preset>/`. Download or copy the results from the Colab runtime before it is deleted.

The notebook also writes the worker script and worker log files under `SAFEQD_WORK_DIR`.

## 6. Interpreting the run

- Check the validation output for `PASS`, `FAIL`, `INFO`, and `NOT TESTED` statuses.
- Review `validation_tests.csv` and the environment-validation table. A completed training run does not mean every validation test passed.
- Use `heldout_summary.csv` and the held-out plots for the final comparison between ROSARL and plain MAP-Elites.
- The smoke preset uses one seed, so its results are a pipeline check and descriptive only. It does not support statistical or general performance claims.
- Use multiple seeds and the larger presets for more meaningful experimental evaluation.
- A more negative ROSARL penalty is not, by itself, evidence that the learned policy is safer. Interpret it alongside unsafe-episode rates, native costs, success, and held-out performance.

