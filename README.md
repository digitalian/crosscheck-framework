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

#### Notes

* If you later **re-enable pyenv**, simply replace the first block with the previous pyenv logic; everything else stays the same.  
* The Streamlit app reads Gerrit live each time, so the GUI reflects the same data that the CLI just printed.

* The latest version reflects improvements in visualization clarity, parameter control,  
  and reproducibility for academic use.

