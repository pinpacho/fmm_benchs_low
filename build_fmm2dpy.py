"""Build fmm2dpy against numpy 2.x by replicating python/setup.py with modern f2py.

Why this exists: fmm2dpy is not on PyPI, and fmm2d's python/setup.py depends on
numpy.distutils, which was removed in numpy 2.0 / Python 3.12. This script
rebuilds the same four f2py extension modules (identical `only:` function lists)
with the modern numpy.f2py meson backend, linking the static libfmm2d.a built by
`make lib`, and installs the package into the running interpreter's site-packages.

Usage:  FMM2D_ROOT=/path/to/fmm2d python build_fmm2dpy.py
        (FMM2D_ROOT defaults to ~/src/fmm2d; run `make lib` there first)
"""
import glob
import os
import shutil
import subprocess
import sys
import sysconfig

HOME = os.path.expanduser("~")
ROOT = os.environ.get("FMM2D_ROOT", os.path.join(HOME, "src", "fmm2d"))
S = os.path.join(ROOT, "src")
LIBDIR = os.path.join(ROOT, "lib-static")
BUILD = os.path.join(HOME, ".cache", "fmm2dpy_build")
os.makedirs(BUILD, exist_ok=True)
if not os.path.exists(os.path.join(LIBDIR, "libfmm2d.a")):
    sys.exit(f"libfmm2d.a not found in {LIBDIR} - run 'make lib' in {ROOT} first")

c_opts  = ["_c", "_d", "_cd"];  c_opts2 = ["c", "d", "cd"]
st_opts = ["_s", "_t", "_st"]
p_optsh = ["_p", "_g"];         p_optsh2 = ["p", "g"]
p_optsl = ["_p", "_g", "_h"];   p_optsl2 = ["p", "g", "h"]

helm, helm_v, helm_d = [], [], []
lap, lap_v, lap_d = [], [], []
for st in st_opts:
    for cd in c_opts:
        for pg in p_optsh:
            helm.append("hfmm2d"+st+cd+pg); helm_v.append("hfmm2d"+st+cd+pg+"_vec")
        for pg in p_optsl:
            for k in ("rfmm2d","lfmm2d","cfmm2d"):
                lap.append(k+st+cd+pg); lap_v.append(k+st+cd+pg+"_vec")
bh, bh_d = ["bhfmm2dwrap_guru"], []
for cd in c_opts2:
    for pg in p_optsh2:
        helm_d.append("h2d_direct"+cd+pg); bh_d.append("bh2d_direct"+cd+pg)
    for pg in p_optsl2:
        for k in ("r2d_direct","l2d_direct","c2d_direct"):
            lap_d.append(k+cd+pg)

EXTS = [
    ("hfmm2d_fortran",
     [S+"/helmholtz/"+f for f in ("hfmm2dwrap.f","hfmm2dwrap_vec.f","helmkernels2d.f")],
     helm+helm_v+helm_d),
    ("lfmm2d_fortran",
     [S+"/laplace/"+f for f in ("rfmm2dwrap.f","rfmm2dwrap_vec.f","rlapkernels2d.f",
        "lfmm2dwrap.f","lfmm2dwrap_vec.f","lapkernels2d.f",
        "cfmm2dwrap.f","cfmm2dwrap_vec.f","cauchykernels2d.f")],
     lap+lap_v+lap_d),
    ("bhfmm2d_fortran",
     [S+"/biharmonic/"+f for f in ("bhfmm2dwrap.f","bhkernels2d.f")],
     bh+bh_d),
    ("stfmm2d_fortran",
     [S+"/stokes/stfmm2d.f", S+"/stokes/stokkernels2d.f"],
     ["stfmm2d","st2ddirectstokg","st2ddirectstokstrsg"]),
]

for name, sources, only in EXTS:
    print(f"=== f2py {name} ({len(only)} funcs) ===", flush=True)
    cmd = ([sys.executable, "-m", "numpy.f2py", "-c"] + sources +
           ["-m", name, "only:"] + only + [":",
            "-L"+LIBDIR, "-lfmm2d", "-lgomp",
            "--f77flags=-fPIC -O3 -std=legacy", "--f90flags=-fPIC -O3"])
    r = subprocess.run(cmd, cwd=BUILD, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:]); sys.exit(1)

sp = sysconfig.get_paths()["purelib"]
dst = os.path.join(sp, "fmm2dpy")
os.makedirs(dst, exist_ok=True)
for py in glob.glob(os.path.join(ROOT, "python", "fmm2dpy", "*.py")):
    shutil.copy(py, dst)
n = 0
for so in glob.glob(os.path.join(BUILD, "*.so")):
    shutil.copy(so, dst); n += 1
print(f"Installed fmm2dpy -> {dst}  ({n} extension modules)")
