# =============================================================================
# setup.ps1 - Windows-native DEGRADED setup (PowerShell 5.1 compatible).
#
# On native Windows the Fortran-based libraries (pyfmmlib, fmm2dpy) and the
# TensorFlow/MNN-H2 experiment are NOT installed (no Fortran compiler; Linux
# wheels only). The notebook runs with those cells marked [SKIP].
# For the FULL benchmark use WSL Ubuntu:  bash setup.sh
#
# NOTE: this installs a HEADLESS set (ipykernel + nbconvert, no JupyterLab UI).
# JupyterLab's widget assets exceed the classic 260-char Windows path limit and
# break pip unless Long Path support is enabled. If you want the UI on Windows,
# enable long paths first (https://pip.pypa.io/warnings/enable-long-paths) and
# then run:  .venv-win\Scripts\pip install jupyterlab
#
# Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1
# =============================================================================

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $ScriptDir ".venv-win"

Write-Host "== fmm2d_helmholtz_bench setup (Windows degraded mode) =="
Write-Host "   folder : $ScriptDir"
Write-Host "   venv   : $Venv"

# --- 1. venv ---
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    python -m venv $Venv
    if (-not $?) { Write-Host "[ERROR] venv creation failed"; exit 1 }
}
$Py = Join-Path $Venv "Scripts\python.exe"
& $Py -m pip install --upgrade pip wheel setuptools | Out-Null

# --- 2. headless scientific stack + stablefmmpy (long-path-safe set) ---
Write-Host "== Installing core packages (headless set) =="
& $Py -m pip install numpy scipy matplotlib pandas h5py "numba>=0.61" stablefmmpy ipykernel nbconvert nbclient
if (-not $?) { Write-Host "[WARN] some core packages failed" }

Write-Host "== NOTE: skipping pyfmmlib, fmm2dpy, tensorflow (Linux/WSL only)"
Write-Host "         Those notebook cells will show [SKIP]. Use WSL for the full run."

# --- 3. Jupyter kernel ---
& $Py -m ipykernel install --user --name fmmbench-win --display-name "Python (fmmbench-win)"

# --- 4. import self-test (via temp file: python -c quoting is unreliable in PS 5.1) ---
Write-Host "== Import self-test =="
$SelfTestFile = Join-Path $env:TEMP "fmmbench_selftest.py"
@'
import importlib
for name in ["numpy", "scipy", "matplotlib", "pandas", "h5py", "numba", "stablefmmpy"]:
    try:
        m = importlib.import_module(name)
        print("  OK   " + name.ljust(12) + str(getattr(m, "__version__", "?")))
    except Exception as e:
        print("  SKIP " + name.ljust(12) + "(" + type(e).__name__ + ")")
'@ | Out-File -Encoding utf8 $SelfTestFile
& $Py $SelfTestFile
Remove-Item $SelfTestFile -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "== Done. To execute the notebook headlessly: =="
Write-Host "   & '$Py' -m nbconvert --to notebook --execute --inplace '$ScriptDir\FMM2D_Helmholtz_Bench.ipynb' --ExecutePreprocessor.timeout=3600"
