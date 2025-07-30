#!/usr/bin/env bash
# ───────────────────────────────────────────────
# One-stop launcher:  CLI + Streamlit GUI
# ------------------------------------------------

set -e

# Parse first argument as MODE, default to "all"
MODE="${1:-all}"

VENV=".venv"
PYEXEC="python3"    # system python (3.9+ works fine)

echo ">> Cleaning existing virtual environment …"
#rm -rf "${VENV}"

echo ">> Setting up virtual environment …"
if [ ! -d "${VENV}" ]; then
  ${PYEXEC} -m venv "${VENV}"
  "${VENV}/bin/pip" install --upgrade pip
  "${VENV}/bin/pip" install -r requirements.txt
fi


case "$MODE" in
  gerrit)
    echo ">> Running Gerrit CLI script (prints a_hat / b_dir table) …"
    "${VENV}/bin/python" src/gerrit_ab_estimator_ratio_bazel.py
    ;;
  streamlit)
    echo ">> Launching Streamlit dashboard …"
    echo "   (Ctrl-C to stop, then deactivate the venv: 'deactivate')"
    "${VENV}/bin/streamlit" run src/streamlit_app.py
    ;;
  *)
    echo ">> Running Gerrit CLI script (prints a_hat / b_dir table) …"
    "${VENV}/bin/python" src/gerrit_ab_estimator_ratio_bazel.py
    echo ">> Launching Streamlit dashboard …"
    echo "   (Ctrl-C to stop, then deactivate the venv: 'deactivate')"
    "${VENV}/bin/streamlit" run src/streamlit_app.py
    ;;
esac

