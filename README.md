# Cross-Check Cost/Quality Framework – Reproducibility Package

This repository accompanies the paper  
**“Framework for Optimizing Cross-Check Procedures”** (ProMAC 2025).

It provides  
* fully reproducible Python scripts (`src/`)  
* a minimal data sample (`data/sample_changes.json`)  
* a one-liner wrapper (`run.sh`) that sets up a virtual environment via **python3 venv**,  
  runs the CLI script, and launches the updated Streamlit app with full parameterized sensitivity analysis.

## Quick Start (one line)

```bash
./run.sh
```


The script will

- create .venv/ (if not present)
- compute and display cross-check efficiency metrics per project
- launch an interactive Streamlit dashboard with:
  - Elasticity and Standardized Sensitivity analysis
  - Tornado diagrams
  - Monte Carlo distributions
  - Configurable display language (EN/JP)
  - Unified layout and updated parameter names

Tested on macOS 14 (Python 3.9) and Ubuntu 22.04 (Python 3.10).

---

### Command-Line Arguments (CLI)

The CLI version accepts the following optional arguments:

- `--streamlit` — Launch only the Streamlit app  
- `--gerrit` — Run Gerrit-related CLI extraction only  
- `--all` (default) — Run both CLI and Streamlit in one go  
- `--lang en|ja` — Set language for CLI/Streamlit display  

Examples:
```bash
python src/main.py --streamlit
python src/main.py --gerrit --lang ja
```

---

### Streamlit App Overview

The Streamlit dashboard displays:

- **Efficiency Metrics (E_total, C, S)** — per project or scenario
- **Sensitivity Analysis**  
  - Relative Sensitivity (`∂E/∂x × x/E`)
  - Standardized Sensitivity (`∂E/∂x × σₓ / σ_E`)
- **Tornado Diagrams** (±20% variation)
- **Monte Carlo Simulation** with histogram and spider charts
- **Language Toggle (EN/JA)** — upper-right language switch
- **Scenario Filters** — quality/schedule grid (2×2), with visual comparison

All graphs are consistently styled and labeled, with markdown hints in both languages.

---

### Gerrit Integration

The CLI version automatically reads recent Gerrit logs (e.g. `data/sample_changes.json`) and:

- extracts submission and review data
- computes work vs review durations
- exports summary metrics for visualization
- serves as reproducible backend for the GUI

To update the dataset, replace `data/sample_changes.json` with your own JSON export from Gerrit.

---

#### Notes

* If you later **re-enable pyenv**, simply replace the first block with the previous pyenv logic; everything else stays the same.  
* The Streamlit app reads Gerrit live each time, so the GUI reflects the same data that the CLI just printed.

* The latest version reflects improvements in visualization clarity, parameter control,  
  and reproducibility for academic use.

