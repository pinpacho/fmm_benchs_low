# fmm2d_helmholtz_bench — A Unified Benchmark for 2D Helmholtz FMM Implementations and Low-Rank Stability

A single, reproducible Jupyter-notebook platform that compares the **stable
(balanced) matrix version of the wideband Fast Multipole Method** — the method of
Michelle, Ou & Xia implemented in [`stablefmmpy`](https://pypi.org/project/stablefmmpy/) —
against other established FMM libraries on the 2D Helmholtz kernel

```
K[i,j] = H0^(1)( k * |x_i - y_j| ),    points in the complex plane C
```

and studies how **low-rank approximations of kernel matrices** behave across
techniques (algebraic vs analytic) and regimes (static Laplace vs oscillatory
Helmholtz).

Everything lives in one executed notebook: **`FMM2D_Helmholtz_Bench.ipynb`**
(regenerated from `gen_fmm2d_helmholtz_bench.py`).

## What the notebook does

| Section | Content |
|---|---|
| 1. Installation | Verifies the environment built by the setup scripts; every later cell degrades gracefully (`[SKIP]`) if a library is missing. |
| 2. Research module | Low-rank survey + experiments: singular-value decay and ε-rank of Laplace vs Helmholtz blocks (rank ∝ kδ in the high-frequency regime); SVD / rSVD / CPQR / partially-pivoted ACA / ID head-to-head on an oscillatory block; RPCholesky demo and the road to RPLU; analytic Taylor [M2D] and 2D Chebyshev (bbFMM-style) vs the SVD optimum. |
| 3. Definitions | The balancing factors λ_{x,p} = max{1, p!·(2/(kδ))^p} [HK Eq. 2.17] plotted in log10; the two-regime (LF/HF) map with the switching boundary kδ = r/e. |
| 4. Benchmark harness | Common dataset, per-library adapters, kernel-convention gate (`stablefmmpy` computes bare H0; `pyfmmlib`/`fmm2dpy` compute (i/4)·H0 — outputs are normalized by 4/i and cross-checked to ~1e-15 before any benchmark runs), N sweeps and accuracy-knob sweeps. |
| 5. Visualization | Time vs N (with the published single-core `hfmm2d` timings as reference), accuracy vs wavenumber k, accuracy-vs-cost Pareto fronts, the stability reproduction ([HK] Table 6.1 and [M2D] Tables 6.1–6.2: balanced errors ~1e-14/1e-15 while the naive factors overflow to Inf), M2L memory (dense (2r+1)² LF vs diagonal HF), and the switching-level experiment. |
| 6. MNN-H2 | Trains the H²-matrix neural network of Fan et al. **without MATLAB**: the original repo's MATLAB scripts only produce two HDF5 arrays, which we generate in Python from the dense Helmholtz kernel on a circle (translation-invariant, matching their Conv1D architecture). Two networks (Re/Im). Compared head-to-head with the classical FMMs on the same geometry. |
| 7. Library landscape | Qualitative matrix of all six libraries, with verified reasons why ScalFMM (no Helmholtz kernel in the release) and FastMMLib (3D-only, no Python bindings) are not benchmarked numerically; reproduction check against the published paper tables. |
| 8. Reproducibility | Environment dump (`results/environment.json`) + full references. |

Machine-greppable check lines are printed throughout (`[CHECK] ... PASS/FAIL`,
`[SKIP] ...`) so a headless run can be validated with `grep`.

## Quickstart — WSL / Linux (full benchmark)

```bash
cd fmm2d_helmholtz_bench
bash setup.sh                # creates ~/.venvs/fmmbench and installs everything
source ~/.venvs/fmmbench/bin/activate
jupyter lab --notebook-dir .
# or headless:
jupyter nbconvert --to notebook --execute --inplace FMM2D_Helmholtz_Bench.ipynb \
    --ExecutePreprocessor.timeout=3600
```

`setup.sh` is deliberately self-sufficient (verified on WSL Ubuntu 24.04 **without
sudo**): if the system Python lacks `pip`/`venv`/dev headers it bootstraps a
self-contained CPython 3.12 via [`uv`](https://docs.astral.sh/uv/); if `make` is
missing it builds GNU make into `~/.local/bin`. It needs `gfortran` and `git`
(both usually present; otherwise `sudo apt install gfortran git`).

Options:

- `FMMBENCH_DEV=1 bash setup.sh` — install `stablefmmpy` editable from a sibling
  `../../stablefmmpy` checkout instead of PyPI (for benchmarking uncommitted work).
- `FMMBENCH_VENV=/path bash setup.sh` — custom venv location.
- `bash setup.sh --local` — additionally copy this folder to
  `~/fmm2d_helmholtz_bench` on the Linux filesystem. Use this if executing the
  notebook from a Windows-mounted path (`/mnt/d/...`) is slow or flaky; run it
  there and copy the executed notebook + `figures/` + `results/` back.

### Fortran libraries: why the setup script does something unusual

- **pyfmmlib** ([inducer/pyfmmlib](https://github.com/inducer/pyfmmlib)): the PyPI
  wheels and sdist are compiled against **numpy 1.x**; importing them under
  numpy 2.x fails (`numpy.core.multiarray failed to import`). `setup.sh` builds the
  sdist with `--no-build-isolation` so it compiles against the venv's numpy.
- **fmm2dpy** ([flatironinstitute/fmm2d](https://github.com/flatironinstitute/fmm2d)):
  PyPI has [fmm2dpy 0.0.5](https://pypi.org/project/fmm2dpy/) (Nov 2022) with wheels
  only for CPython 3.6–3.11 and **no sdist**, so on Python ≥3.12 pip finds nothing —
  and the repo's own `setup.py` needs `numpy.distutils`, removed in numpy 2.x.
  `setup.sh` tries pip first, then follows the
  [official source route](https://fmm2d.readthedocs.io/en/latest/install.html#obtaining-fmm2d)
  (`git clone` + `make lib`) and rebuilds the four f2py extension modules with the
  modern `numpy.f2py` meson backend via the shipped **`build_fmm2dpy.py`**
  (identical `only:` function lists as upstream's `python/setup.py`).

Both were validated against the dense bare-H0 sum: relative error ≈ 1.1e-15 after
the 4/i normalization.

## Quickstart — native Windows (degraded)

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
# headless execution:
.venv-win\Scripts\python.exe -m nbconvert --to notebook --execute --inplace `
    FMM2D_Helmholtz_Bench.ipynb --ExecutePreprocessor.timeout=3600
```

No Fortran compiler on native Windows → `pyfmmlib`, `fmm2dpy` and the TensorFlow
section print `[SKIP]`; `stablefmmpy`, the research module, the definitions and the
stability sections run fully. Use WSL for the complete comparison.

`setup.ps1` installs a **headless** set (ipykernel + nbconvert, no JupyterLab UI):
JupyterLab's widget assets exceed the classic 260-character Windows path limit and
break `pip` unless [Long Path support](https://pip.pypa.io/warnings/enable-long-paths)
is enabled. If you want the UI on Windows, enable long paths and then
`.venv-win\Scripts\pip install jupyterlab`.

## Regenerating the notebook

The `.ipynb` is generated (repo convention — notebooks are authored as Python
generator scripts, not edited as JSON):

```bash
python gen_fmm2d_helmholtz_bench.py     # rewrites FMM2D_Helmholtz_Bench.ipynb
```

## Outputs

- `figures/*.png` — all plots (rank vs kδ, algebraic methods, RPCholesky,
  analytic-vs-algebraic, λ factors, regime map, time vs N, accuracy vs k, Pareto,
  stability, M2L memory, switching level, MNN-H2 comparison).
- `results/*.csv` + `results/environment.json` — every table and the exact
  environment of the run.
- `data/` — the Python-generated MNN-H2 training files (gitignored).

Expected full-run wall time: ~15–25 min in WSL (MNN-H2 training dominates);
~4 min on Windows degraded.

## Libraries compared

| Library | 2D Helmholtz | Role here |
|---|---|---|
| [stablefmmpy](https://pypi.org/project/stablefmmpy/) ([repo](https://github.com/pinpacho/stablefmmpy)) | yes (wideband, balanced) | numeric — the method under study |
| [pyfmmlib](https://github.com/inducer/pyfmmlib) / [fmmlib2d](https://github.com/zgimbutas/fmmlib2d) | yes (low-frequency FMM) | numeric |
| [fmm2d](https://github.com/flatironinstitute/fmm2d) (`fmm2dpy`) | yes | numeric |
| [mnn-H2](https://github.com/ywfan/mnn-H2) | learned here (Section 6) | numeric via Python-generated data |
| [ScalFMM](https://gitlab.inria.fr/solverstack/ScalFMM) | **no** — no Helmholtz kernel in the release; experimental Python binding is 2D-Laplace-only | qualitative |
| [FastMMLib](https://plmlab.math.cnrs.fr/fastmmlib/fastmmlib) | **no** — 3D kernels only, no Python bindings | qualitative |

## References

**Core papers (the method under study)**

- **[HK]** M. Michelle, X. Ou, J. Xia, *A Stable Matrix Version of the Wideband Fast
  Multipole Method for the 2D Helmholtz Kernel* (preprint).
- **[M2D]** X. Ou, M. Michelle, J. Xia, *A Stable Matrix Version of the 2D Fast
  Multipole Method*, SIAM J. Matrix Anal. Appl. 46(1), 2025.
  [doi:10.1137/24M1636953](https://doi.org/10.1137/24M1636953)

**Low-rank approximation**

- **[HMT11]** N. Halko, P.-G. Martinsson, J. A. Tropp, *Finding Structure with
  Randomness: Probabilistic Algorithms for Constructing Approximate Matrix
  Decompositions*, SIAM Review 53(2), 2011.
  [doi:10.1137/090771806](https://doi.org/10.1137/090771806)
- **[CGMR05]** H. Cheng, Z. Gimbutas, P.-G. Martinsson, V. Rokhlin, *On the
  Compression of Low Rank Matrices*, SIAM J. Sci. Comput. 26(4), 2005.
  [doi:10.1137/030602678](https://doi.org/10.1137/030602678) — the interpolative
  decomposition behind
  [`scipy.linalg.interpolative`](https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html).
- **[Beb00]** M. Bebendorf, *Approximation of boundary element matrices*,
  Numerische Mathematik 86, 2000.
  [doi:10.1007/PL00005410](https://doi.org/10.1007/PL00005410)
- **[CETW22]** Y. Chen, E. N. Epperly, J. A. Tropp, R. J. Webber, *Randomly pivoted
  Cholesky: Practical approximation of a kernel matrix with few entry evaluations*,
  2022. [arXiv:2207.06503](https://arxiv.org/abs/2207.06503)
- **[GW26]** M. A. Gilles, H. Wilber, *Low-Rank Approximation by Randomly Pivoted
  LU*, 2026. [arXiv:2601.22344](https://arxiv.org/abs/2601.22344)
- **[FD09]** W. Fong, E. Darve, *The black-box fast multipole method*,
  J. Comput. Phys. 228(23), 2009.
  [doi:10.1016/j.jcp.2009.08.031](https://doi.org/10.1016/j.jcp.2009.08.031)
- **[CD13]** C. Cecka, E. Darve, *Fourier-Based Fast Multipole Method for the
  Helmholtz Equation*, SIAM J. Sci. Comput. 35(1), 2013.
  [doi:10.1137/11085774X](https://doi.org/10.1137/11085774X)

**Surveys mined for the research module**

- **[Kai25]** S. Kailasa, *Modern Research Software for Fast Multipole Methods*,
  PhD thesis, University College London, May 2025.
  [UCL Discovery](https://discovery.ucl.ac.uk/)
- **[Bla17]** P. Blanchard, *Fast hierarchical algorithms for the low-rank
  approximation of matrices, with applications to materials physics, geostatistics
  and data analysis*, PhD thesis, Université de Bordeaux, 2017.
  [HAL tel-01534930](https://theses.hal.science/tel-01534930)

**Libraries and their papers**

- **[GG15]** Z. Gimbutas, L. Greengard, *Computational software: Simple FMM
  libraries for electrostatics, slow viscous flow, and frequency-domain wave
  propagation*, Comm. Comput. Phys. 18(2), 2015.
  [doi:10.4208/cicp.150215.260615sw](https://doi.org/10.4208/cicp.150215.260615sw)
- fmm2d documentation: [fmm2d.readthedocs.io](https://fmm2d.readthedocs.io/)
- ScalFMM: [gitlab.inria.fr/solverstack/ScalFMM](https://gitlab.inria.fr/solverstack/ScalFMM)
  (see `include/scalfmm/utils/low_rank.hpp` for its production pACA + SVD pipeline)
- FastMMLib: É. Darrigrand, Y. Lafranche, R. Rais, *FastMMLib: a generic Fast
  Multipole Method library*, [HAL hal-05679634](https://hal.science/hal-05679634) ·
  [PLM GitLab](https://plmlab.math.cnrs.fr/fastmmlib/fastmmlib)
- **[FFLYZ19]** Y. Fan, J. Feliu-Fabà, L. Lin, L. Ying, L. Zepeda-Núñez, *A
  multiscale neural network based on hierarchical nested bases*, Research in the
  Mathematical Sciences 6, 2019. [arXiv:1808.02376](https://arxiv.org/abs/1808.02376) ·
  [github.com/ywfan/mnn-H2](https://github.com/ywfan/mnn-H2)
