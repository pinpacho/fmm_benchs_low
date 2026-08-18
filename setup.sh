#!/usr/bin/env bash
# =============================================================================
# setup.sh - build the FULL benchmark environment on Linux / WSL Ubuntu.
#
# Usage:
#   bash setup.sh                       # standard setup (venv at $HOME/.venvs/fmmbench)
#   FMMBENCH_DEV=1 bash setup.sh        # stablefmmpy editable from ../../stablefmmpy
#   FMMBENCH_VENV=/path bash setup.sh   # custom venv location
#   bash setup.sh --local               # ALSO copy this folder to ~/fmm2d_helmholtz_bench
#                                       # (for when running off /mnt/d is slow/flaky)
#
# Design notes (all verified on WSL Ubuntu 24.04, no sudo required):
#   * The venv lives on the Linux filesystem (ext4), NOT under /mnt/* - venvs on
#     the Windows-mounted drive are slow and have symlink quirks.
#   * If the system python3 lacks pip/venv or the C headers (python3-dev not
#     installed and no sudo), a self-contained CPython 3.12 is bootstrapped with
#     `uv` - it ships include/Python.h, which f2py/meson builds need.
#   * If `make` is missing (needed by fmm2d), GNU make is built from source into
#     ~/.local/bin using its own no-make bootstrap (build.sh).
#   * pyfmmlib: PyPI wheels/sdist are compiled against numpy 1.x; with numpy 2.x
#     in the venv the import fails (numpy.core.multiarray). Fix: build the sdist
#     with --no-build-isolation so it compiles against the venv's numpy.
#   * fmm2dpy is NOT on PyPI and its setup.py needs the removed numpy.distutils.
#     Fix: build libfmm2d.a with make, then build the 4 f2py extensions with
#     modern numpy.f2py via the shipped build_fmm2dpy.py.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${FMMBENCH_VENV:-$HOME/.venvs/fmmbench}"
SRC_DIR="${FMMBENCH_SRC:-$HOME/src}"

echo "== fmm2d_helmholtz_bench setup =="
echo "   folder : $SCRIPT_DIR"
echo "   venv   : $VENV"

# --- 0. pick a Python with dev headers ---------------------------------------
have_headers() {
    "$1" -c "import sysconfig, os; raise SystemExit(0 if os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')) else 1)" 2>/dev/null
}
PYBASE="$(command -v python3 || true)"
if [ -z "$PYBASE" ] || ! have_headers "$PYBASE"; then
    echo "== System python3 lacks dev headers - bootstrapping CPython 3.12 via uv =="
    if ! command -v "$HOME/.local/bin/uv" >/dev/null 2>&1 && ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
    "$UV" python install 3.12
    PYBASE="$("$UV" python find 3.12)"
fi
echo "   python : $PYBASE"

# --- 1. venv (with pip bootstrap for --without-pip situations) ---------------
if [ ! -x "$VENV/bin/python" ]; then
    mkdir -p "$(dirname "$VENV")"
    if ! "$PYBASE" -m venv "$VENV" 2>/dev/null; then
        echo "[INFO] ensurepip unavailable - venv --without-pip + get-pip.py"
        rm -rf "$VENV"
        "$PYBASE" -m venv --without-pip "$VENV" || { echo "[ERROR] venv creation failed"; exit 1; }
    fi
fi
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$VENV/bin/python" /tmp/get-pip.py --quiet || { echo "[ERROR] pip bootstrap failed"; exit 1; }
fi
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip wheel setuptools
export PATH="$VENV/bin:$HOME/.local/bin:$PATH"

# --- 2. core requirements (includes stablefmmpy from PyPI) --------------------
echo "== Installing core requirements =="
"$PY" -m pip install -r "$SCRIPT_DIR/requirements.txt" || echo "[WARN] some core requirements failed"

if [ "${FMMBENCH_DEV:-0}" = "1" ] && [ -d "$SCRIPT_DIR/../../stablefmmpy" ]; then
    echo "== FMMBENCH_DEV=1: stablefmmpy editable from sibling checkout =="
    "$PY" -m pip install -e "$SCRIPT_DIR/../../stablefmmpy" || echo "[WARN] editable install failed"
fi

# --- 3. TensorFlow for the MNN-H2 section (best-effort) -----------------------
echo "== Installing tensorflow-cpu + tf-keras (Section 6; ~250 MB) =="
"$PY" -m pip install tensorflow-cpu tf-keras || echo "[WARN] tensorflow install failed - Section 6 will [SKIP]"

# --- 4. build tools: gfortran (system) and make (bootstrap if missing) --------
if ! command -v gfortran >/dev/null 2>&1; then
    echo "[WARN] gfortran not found - pyfmmlib/fmm2dpy builds will fail."
    echo "       Install it with: sudo apt install gfortran"
fi
if ! command -v make >/dev/null 2>&1; then
    echo "== Bootstrapping GNU make into ~/.local/bin (no sudo needed) =="
    mkdir -p "$HOME/tmp" "$HOME/.local/bin"
    ( cd "$HOME/tmp" \
      && curl -sSL -o mk.tgz https://ftp.gnu.org/gnu/make/make-4.4.1.tar.gz \
      && tar xzf mk.tgz && cd make-4.4.1 \
      && ./configure --prefix="$HOME/.local" > conf.log 2>&1 || true
      cd "$HOME/tmp/make-4.4.1" && ./build.sh > build.log 2>&1 \
      && cp make "$HOME/.local/bin/make" ) \
    && echo "   $($HOME/.local/bin/make --version | head -1)" \
    || echo "[WARN] make bootstrap failed"
fi

# --- 5. pyfmmlib: source build against the venv's numpy ----------------------
if ! "$PY" -c "import pyfmmlib" >/dev/null 2>&1; then
    echo "== Building pyfmmlib from source (f2py/meson; a few minutes) =="
    "$PY" -m pip install --quiet meson-python meson ninja mako
    "$PY" -m pip uninstall -y -q pyfmmlib 2>/dev/null || true
    "$PY" -m pip install --no-cache-dir --no-build-isolation --no-binary pyfmmlib pyfmmlib \
        && "$PY" -c "import pyfmmlib; print('   pyfmmlib import OK')" \
        || echo "[WARN] pyfmmlib build failed - notebook will [SKIP] it"
fi

# --- 6. fmm2dpy: PyPI first, then clone + make + modern-f2py fallback --------
# PyPI has fmm2dpy 0.0.5 (Nov 2022) with wheels only up to CPython 3.11, built
# against numpy 1.x, and no sdist - so on Python >=3.12 / numpy 2.x the pip
# route fails and we build from the official source per
# https://fmm2d.readthedocs.io/en/latest/install.html#obtaining-fmm2d
if ! "$PY" -c "import fmm2dpy" >/dev/null 2>&1; then
    echo "== Trying pip install fmm2dpy (works only if a compatible wheel exists) =="
    "$PY" -m pip install fmm2dpy 2>/dev/null && "$PY" -c "import fmm2dpy" >/dev/null 2>&1 \
        || "$PY" -m pip uninstall -y -q fmm2dpy 2>/dev/null || true
fi
if ! "$PY" -c "import fmm2dpy" >/dev/null 2>&1; then
    echo "== Building fmm2d (Flatiron) + fmm2dpy from GitHub source =="
    mkdir -p "$SRC_DIR"
    if [ ! -d "$SRC_DIR/fmm2d" ]; then
        git clone --depth 1 https://github.com/flatironinstitute/fmm2d.git "$SRC_DIR/fmm2d" \
            || echo "[WARN] fmm2d clone failed"
    fi
    if [ -d "$SRC_DIR/fmm2d" ]; then
        ( cd "$SRC_DIR/fmm2d" && make lib > "$HOME/fmm2d_lib_build.log" 2>&1 ) \
            || echo "[WARN] fmm2d 'make lib' failed (see ~/fmm2d_lib_build.log)"
        FMM2D_ROOT="$SRC_DIR/fmm2d" "$PY" "$SCRIPT_DIR/build_fmm2dpy.py" \
            && "$PY" -c "import fmm2dpy; print('   fmm2dpy import OK')" \
            || echo "[WARN] fmm2dpy build failed - notebook will [SKIP] it"
    fi
fi

# --- 7. Jupyter kernel ---------------------------------------------------------
"$PY" -m ipykernel install --user --name fmmbench --display-name "Python (fmmbench)" >/dev/null \
    && echo "== Registered Jupyter kernel 'fmmbench' =="

# --- 8. optional --local copy ---------------------------------------------------
if [ "${1:-}" = "--local" ] || [ "${FMMBENCH_LOCAL:-0}" = "1" ]; then
    DEST="$HOME/fmm2d_helmholtz_bench"
    echo "== --local: copying benchmark folder to $DEST =="
    mkdir -p "$DEST"
    cp -r "$SCRIPT_DIR/." "$DEST/"
    echo "   Run there, then copy back the executed notebook + figures/ + results/:"
    echo "   cp $DEST/FMM2D_Helmholtz_Bench.ipynb '$SCRIPT_DIR/'"
    echo "   cp -r $DEST/figures $DEST/results '$SCRIPT_DIR/'"
fi

# --- 9. import self-test ----------------------------------------------------------
echo "== Import self-test =="
"$PY" - <<'EOF'
import importlib
for name in ["numpy", "scipy", "matplotlib", "pandas", "h5py", "numba",
             "stablefmmpy", "pyfmmlib", "fmm2dpy", "tf_keras"]:
    try:
        m = importlib.import_module(name)
        print(f"  OK   {name:<12} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  SKIP {name:<12} ({type(e).__name__})")
EOF

echo ""
echo "== Done. To run the notebook: =="
echo "   source $VENV/bin/activate"
echo "   jupyter lab --notebook-dir \"$SCRIPT_DIR\""
echo "   # or headless:"
echo "   jupyter nbconvert --to notebook --execute --inplace \"$SCRIPT_DIR/FMM2D_Helmholtz_Bench.ipynb\" --ExecutePreprocessor.timeout=3600"
