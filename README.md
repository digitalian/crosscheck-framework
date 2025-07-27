# Cross-Check Cost/Quality Framework – Reproducibility Package

This repository accompanies the paper  
**“Framework for Optimizing Cross-Check Procedures”** (ProMAC 2025).

It provides  
* fully reproducible Python scripts (`src/`)  
* a minimal data sample (`data/sample_changes.json`)  
* a one-liner wrapper (`run.sh`) that creates a local virtual env via **pyenv**  
  and executes the CLI script.

## Quick Start (one line)

```bash
./run.sh

The script will

create .venv/ (if not present)

print the per-project a / b ratios to the console

open a Streamlit dashboard at http://localhost:8501
(Ctrl-C to quit)

Tested on macOS 14 (Python 3.9) and Ubuntu 22.04 (Python 3.10).

---

#### Notes

* If you later **re-enable pyenv**, simply replace the first block with the previous pyenv logic; everything else stays the same.  
* The Streamlit app reads Gerrit live each time, so the GUI reflects the same data that the CLI just printed.
::contentReference[oaicite:0]{index=0}
