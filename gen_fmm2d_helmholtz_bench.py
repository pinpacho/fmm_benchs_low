"""Generator for FMM2D_Helmholtz_Bench.ipynb.

Unified benchmark platform comparing 2D Helmholtz FMM implementations
(stablefmmpy / pyfmmlib / fmm2dpy / dense baseline / MNN-H2 neural network)
plus a research module on low-rank approximation techniques.

Run:  python gen_fmm2d_helmholtz_bench.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _nbtools import md, code, write_notebook  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "FMM2D_Helmholtz_Bench.ipynb")


# =============================================================================
# S0 - Header
# =============================================================================

CELL_01_TITLE = md(r"""# A Unified Benchmark for 2D Helmholtz FMM Implementations and Low-Rank Stability

**Kernel under study:** $K_{ij} = H_0^{(1)}(k\,|x_i - y_j|)$, points in the complex plane $\mathbb{C}$.

This notebook is a single test platform that compares the **stable (balanced) matrix
version of the wideband FMM** implemented in
[`stablefmmpy`](https://pypi.org/project/stablefmmpy/) — the method of Michelle,
Ou & Xia [HK], [M2D] — against other established FMM libraries, and studies how
**low-rank approximations of kernel matrices** behave across techniques and regimes.

## Contents

| Section | Module |
|---|---|
| 1 | Installation & environment verification |
| 2 | **Research module** — low-rank behavior: algebraic vs analytic schemes, Laplace vs Helmholtz rank growth, recent randomized-pivoting methods (ID, RPCholesky, RPLU) |
| 3 | **Definitions** — kernel, balancing factors $\lambda_{x,p}$, two-regime (LF/HF) architecture |
| 4 | **Benchmark harness** — common dataset, kernel-convention normalization, per-library adapters |
| 5 | **Visualization** — time vs $N$, accuracy vs $k$, accuracy-vs-cost Pareto, stability (error vs $r$), M2L memory, switching level |
| 6 | **MNN-H2 experiment** — training the H2-matrix neural network of Fan et al. on Helmholtz data generated *without MATLAB* |
| 7 | Qualitative library comparison + reproduction of the published tables |
| 8 | Reproducibility footer + references |

**Libraries benchmarked numerically:** `stablefmmpy` (balanced + naive variants),
`pyfmmlib` (Greengard/Gimbutas `fmmlib2d`), `fmm2dpy` (Flatiron Institute `fmm2d`),
dense direct evaluation, and the MNN-H2 network.
**Compared qualitatively:** ScalFMM, FastMMLib (no 2D Helmholtz kernel / no Python
bindings — see Section 7).

Citation keys used throughout: **[HK]** Michelle, Ou & Xia, *A Stable Matrix Version of
the Wideband FMM for the 2D Helmholtz Kernel*; **[M2D]** Ou, Michelle & Xia, *A Stable
Matrix Version of the 2D FMM*, SIAM J. Matrix Anal. Appl. 46(1), 2025. Full linked
bibliography in Section 8.
""")

# =============================================================================
# S1 - Installation & environment
# =============================================================================

CELL_02_SETUP_MD = md(r"""## 1. Installation module — environment & availability

The environment is created **outside** this notebook by the setup scripts shipped
alongside it (a notebook cannot create the virtual environment its own kernel runs in):

- **`setup.sh`** (Linux / WSL Ubuntu — *full* benchmark): creates a pure-Python venv,
  installs the core stack + `stablefmmpy` (PyPI), builds the Fortran-based
  `pyfmmlib` from source, builds Flatiron `fmm2d` + its Python wrappers from GitHub,
  and installs `tensorflow-cpu` + `tf-keras` for Section 6. If the system Python
  lacks `pip`/headers and there is no `sudo`, it bootstraps a self-contained
  CPython via [`uv`](https://docs.astral.sh/uv/) and a user-local GNU `make` — no root required.
- **`setup.ps1`** (native Windows — *degraded* benchmark): core stack +
  `stablefmmpy` only. The Fortran libraries have no Windows wheels and need a
  Fortran compiler, so their cells below print `[SKIP]` and the published
  reference timings are used instead.

This notebook only **verifies** what is importable and adapts. Every later cell is
guarded by the `AVAILABLE` dictionary built here.
""")

CELL_03_AVAILABILITY = code(r"""import importlib
import os
import platform
import sys
from time import perf_counter

T0 = perf_counter()

LIBS = ["numpy", "scipy", "matplotlib", "pandas", "h5py", "numba",
        "stablefmmpy", "pyfmmlib", "fmm2dpy", "tf_keras"]
AVAILABLE = {}
for _name in LIBS:
    try:
        _mod = importlib.import_module(_name)
        AVAILABLE[_name] = getattr(_mod, "__version__", "installed")
    except Exception as _exc:
        AVAILABLE[_name] = None
        print(f"[SKIP] {_name}: {type(_exc).__name__}: {_exc}")

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

print()
print(f"Platform : {platform.platform()}")
print(f"Python   : {sys.version.split()[0]}")
print(f"{'library':<14}{'version':<14}")
print("-" * 28)
for _name in LIBS:
    print(f"{_name:<14}{AVAILABLE[_name] or '-- NOT AVAILABLE --'}")
""")

CELL_04_PARAMS = code(r"""import numpy as np

# ----- global experiment configuration (single source of truth) -----
SEED       = 42          # every cell builds rng = np.random.default_rng(SEED)
DENSE_MAX  = 4000        # max N for a full dense ground-truth matvec (4000^2 c128 = 256 MB)
SUBSET_M   = 200         # targets used for subset-error estimation when N > DENSE_MAX
REPEATS    = 3           # best-of-REPEATS timing
WARMUP     = 1           # warmup calls before timing
K_BENCH    = 10.0        # wavenumber for the N sweeps
R_STABLE   = 25          # stablefmmpy expansion order for the N sweeps
N_FIX      = 1000        # fixed N for the k sweep and the Pareto sweep

N_SWEEP_STABLE  = [250, 500, 1000, 2000, 4000]              # adaptively capped below
N_SWEEP_FORTRAN = [500, 1000, 2000, 4000, 8000, 16000, 32000]
K_SWEEP         = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]
R_PARETO        = [5, 10, 15, 20, 25, 30, 40]
IPREC_SWEEP     = [-1, 0, 1, 2, 3, 4, 5]
EPS_SWEEP       = [1e-3, 1e-6, 1e-9, 1e-12]

print("Experiment configuration")
print("-" * 40)
for _k, _v in [("SEED", SEED), ("DENSE_MAX", DENSE_MAX), ("SUBSET_M", SUBSET_M),
               ("REPEATS", REPEATS), ("K_BENCH", K_BENCH), ("R_STABLE", R_STABLE),
               ("N_FIX", N_FIX)]:
    print(f"{_k:<12}= {_v}")
""")

# =============================================================================
# S2 - Research module
# =============================================================================

CELL_05_RESEARCH_MD = md(r"""## 2. Research module — how low-rank approximations of kernel matrices behave

For well-separated point sets ($\delta_x + \delta_y \le \tau\,|o_x - o_y|$), the kernel
block $K$ admits a low-rank factorization $K \approx U B V^{\mathsf T}$. **How** that
factorization is computed splits into two families with very different trade-offs.

### 2.1 Algebraic (data-driven) schemes

| Method | Access needed | Cost (rank $r$, $m\times n$) | Optimality | Reference |
|---|---|---|---|---|
| Truncated **SVD** | full matrix | $O(mnr)$ | optimal in every unitarily invariant norm (Eckart–Young) | [GVL13] |
| **CPQR** (column-pivoted QR) | full matrix | $O(mnr)$, smaller constants | quasi-optimal | [GVL13] |
| **ID** (interpolative decomposition) | full matrix (or sketch) | $O(mnr)$ | quasi-optimal; factors reuse actual columns/rows | [CGMR05] |
| **ACA** (adaptive cross approximation, partial pivoting) | rows+columns on demand | $O(r^2(m+n))$ | heuristic, excellent on asymptotically smooth kernels | [Beb00] |
| **rSVD** (randomized SVD) | matvecs / sketch | $O(mn\log r + (m{+}n)r^2)$ | near-optimal w.h.p. | [HMT11] |
| **RPCholesky** (randomly pivoted Cholesky, PSD) | diagonal + $r{+}1$ columns | $O(r^2 n)$ | expected-trace-error guarantees | [CETW22] |
| **RPLU** (randomly pivoted LU, nonsymmetric) | entries on demand | $O(r^2(m+n))$-type | randomized complete pivoting; beats greedy cross methods on adversarial geometry | [GW26] |

The 2022–2026 randomized-pivoting line (RPCholesky → accelerated/rejection variants →
**RPLU** for general nonsymmetric matrices, Gilles & Wilber, Jan 2026) is directly aimed
at **hierarchical-matrix assembly**: like ACA it touches only $O(r)$ rows/columns, but the
pivot is *sampled* from the residual distribution instead of chosen greedily, which
repairs ACA's known failure modes. ScalFMM already ships a production C++ pipeline of
the older generation — partial ACA followed by QR re-compression and a small SVD
(`include/scalfmm/utils/low_rank.hpp`, functions `paca()` / `tsvd()`), toggled in
`examples/test_dimension_low_rank.cpp`.

### 2.2 Analytic (expansion) schemes

- **Taylor expansions** with balancing — the [M2D] approach for non-oscillatory kernels
  (generalized Cauchy, log). Explicit error bounds; rank $r{+}1$ in 2D but $O(p^{d-1})$
  terms in $d$ dimensions (the "curse" that motivates 3D alternatives).
- **Graf-addition / Bessel–Hankel expansions** with balancing — the [HK] low-frequency
  Helmholtz scheme benchmarked in this notebook (Section 5).
- **Chebyshev interpolation** (bbFMM, Fong & Darve [FD09]) — kernel-independent, uniform
  exponential convergence, no Runge phenomenon; the M2L block is often re-compressed
  with an SVD.
- **Equispaced / DFT plane-wave basis** (Cecka & Darve [CD13]) — the [HK] high-frequency
  scheme: inherently stable, diagonal $B$ of size $(2r{+}1)$.

Analytic schemes need **no matrix entries at all** (assembly is $O((m{+}n)r)$), but their
rank is a constant factor above the $\varepsilon$-optimal one; algebraic schemes are
(near-)optimal in rank but must touch the matrix. Production FMM codes therefore often
**combine** them: analytic basis + algebraic re-compression (bbFMM, ScalFMM, kiFMM
[Kai25] §2.3).

### 2.3 Rank behavior: Laplace (static) vs Helmholtz (oscillatory)

For the Laplace/log kernel the $\varepsilon$-rank of a well-separated block is bounded
by a constant depending only on $\varepsilon$ and the separation ratio $\tau$ — it does
**not** grow with the box size. For the Helmholtz kernel the block behaves like the
Laplace one only while the box is sub-wavelength ($k\delta \lesssim 1$); beyond that the
$\varepsilon$-rank grows **linearly with $k\delta$** (the kernel oscillates $\sim k\delta/\pi$
times across the box, and each oscillation costs basis vectors) [Kai25] §2.5.1. This is
exactly why [HK] needs a **two-regime** design (Section 3): a balanced Bessel expansion
where the rank would otherwise be inflated by instability (LF), and a plane-wave basis
whose size $2r{+}1 \sim k\delta$ tracks the physical rank (HF). We measure both effects
below.
""")

CELL_06_READ_NOTES = code(r"""# Optional: peek at the two PhD-thesis survey notes stored one directory up.
# NOTE: those extractions are UTF-16LE encoded; a plain open() would fail.
from pathlib import Path

for _fname, _desc in [("compilated_FMM.md", "Kailasa, UCL PhD thesis 2025 [Kai25]"),
                      ("scallfmm.md", "Blanchard, U. Bordeaux PhD thesis 2017 [Bla17]")]:
    _p = Path("..") / _fname
    if _p.exists():
        _txt = _p.read_text(encoding="utf-16", errors="replace")
        print(f"--- {_fname}  ({_desc}; {len(_txt):,} chars) ---")
        print(_txt[:400].replace("\n\n", "\n") + " ...")
        print()
    else:
        print(f"[INFO] {_fname} not present (standalone clone) - see the linked citation instead: {_desc}")
""")

CELL_07_RANK_GEOM = code(r"""import numpy as np
from scipy.special import hankel1

# Canonical research geometry: two disks of radius delta=1, centers 0 and 3
# (separation ratio tau = (1+1)/3 = 2/3 of the |o_x - o_y| distance -> well separated).
rng = np.random.default_rng(SEED)

def disk_points(n, center, radius, rng):
    rr = radius * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    return center + rr * np.exp(1j * th)

N_RES = 400
DELTA_RES = 1.0
x_res = disk_points(N_RES, 0.0 + 0.0j, DELTA_RES, rng)      # targets X
y_res = disk_points(N_RES, 3.0 + 0.0j, DELTA_RES, rng)      # sources Y

DIFF = x_res[:, None] - y_res[None, :]
K_LOG = np.log(DIFF)                                        # complex log kernel [M2D]

K_RES_KS = [0.1, 1.0, 10.0, 25.0, 50.0, 100.0]              # k*delta = k here
K_HELM = {k: hankel1(0, k * np.abs(DIFF)) for k in K_RES_KS}

print(f"Blocks built: log kernel + Helmholtz at k*delta in {K_RES_KS}")
print(f"Block size  : {N_RES} x {N_RES}")
""")

CELL_08_RANK_SVD = code(r"""import matplotlib.pyplot as plt
import pandas as pd

def eps_rank(s, eps):
    return int(np.sum(s > eps * s[0]))

sv_log = np.linalg.svd(K_LOG, compute_uv=False)
sv_helm = {k: np.linalg.svd(K_HELM[k], compute_uv=False) for k in K_RES_KS}

rows = []
for k in K_RES_KS:
    rows.append({"kdelta": k,
                 "rank_1e-6": eps_rank(sv_helm[k], 1e-6),
                 "rank_1e-10": eps_rank(sv_helm[k], 1e-10)})
df_rank = pd.DataFrame(rows)
df_rank.to_csv("results/eps_rank.csv", index=False)
print("Laplace (log) block:  rank(1e-6) =", eps_rank(sv_log, 1e-6),
      "  rank(1e-10) =", eps_rank(sv_log, 1e-10))
print(df_rank.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.semilogy(sv_log / sv_log[0], "k-", lw=2, label="log kernel (Laplace)")
for k in K_RES_KS:
    ax1.semilogy(sv_helm[k] / sv_helm[k][0], label=rf"Helmholtz, $k\delta$={k:g}")
ax1.set_xlabel("singular value index"); ax1.set_ylabel(r"$\sigma_i/\sigma_0$")
ax1.set_ylim(1e-17, 2); ax1.set_xlim(0, 160)
ax1.set_title("Singular value decay of a well-separated block")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

for eps, mark in [(1e-6, "o-"), (1e-10, "s-")]:
    ax2.loglog(K_RES_KS, [eps_rank(sv_helm[k], eps) for k in K_RES_KS], mark,
               label=rf"Helmholtz $\epsilon$={eps:g}")
ax2.axhline(eps_rank(sv_log, 1e-10), color="k", ls="--",
            label=r"log kernel $\epsilon$=1e-10 (flat)")
kd = np.array(K_RES_KS)
ax2.loglog(kd[kd >= 10], kd[kd >= 10], ":", color="gray", label=r"slope-1 guide (rank $\propto k\delta$)")
ax2.set_xlabel(r"$k\delta$"); ax2.set_ylabel(r"$\epsilon$-rank")
ax2.set_title("Numerical rank vs box wavenumber")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig("figures/rank_vs_kdelta.png", dpi=140, bbox_inches="tight")
plt.show()
print("Saved: figures/rank_vs_kdelta.png, results/eps_rank.csv")
""")

CELL_09_ALGEBRAIC_MD = md(r"""### 2.4 Algebraic methods head-to-head

We now build rank-$r$ approximations of **one fixed oscillatory block**
($k\delta = 20$, $400\times400$) with five algebraic methods and measure
$\|K - K_r\|_2 / \|K\|_2$ against the SVD optimum:

- `rsvd` — randomized SVD with Gaussian sketch + one power iteration [HMT11];
- `cpqr_lowrank` — column-pivoted QR via `scipy.linalg.qr(pivoting=True)`;
- `aca_partial` — partially pivoted ACA [Beb00], touching only $O(r)$ rows/columns;
- `id_lowrank` — interpolative decomposition via `scipy.linalg.interpolative` [CGMR05];
- and, on a PSD proxy in §2.5, **RPCholesky** [CETW22] — the sampled-pivot idea that
  **RPLU** [GW26] extends to nonsymmetric kernel blocks like ours.

What to look for: SVD/rSVD/ID track each other closely; ACA is quasi-optimal at a
fraction of the entry evaluations — that access pattern (entries on demand) is what
makes cross/pivoting methods the workhorses of $\mathcal{H}$/HSS assembly, and what the
randomized-pivot variants make robust.
""")

CELL_10_ALGEBRAIC_IMPL = code(r"""import scipy.linalg as sla
import scipy.linalg.interpolative as sli

def rsvd(A, r, oversample=10, n_iter=1, rng=None):
    # Randomized SVD [HMT11]: Gaussian sketch + power iteration + small SVD.
    rng = rng or np.random.default_rng(SEED)
    m, n = A.shape
    Om = rng.standard_normal((n, r + oversample)) + 1j * rng.standard_normal((n, r + oversample))
    Y = A @ Om
    for _ in range(n_iter):
        Y = A @ (A.conj().T @ Y)
    Q, _ = np.linalg.qr(Y)
    Uh, s, Vh = np.linalg.svd(Q.conj().T @ A, full_matrices=False)
    return (Q @ Uh)[:, :r], s[:r], Vh[:r]

def cpqr_lowrank(A, r):
    # Column-pivoted (rank-revealing) QR, truncated at rank r.
    Q, R, piv = sla.qr(A, pivoting=True, mode="economic")
    inv = np.empty_like(piv); inv[piv] = np.arange(len(piv))
    return Q[:, :r] @ R[:r][:, inv]

def aca_partial(A, rmax, tol=0.0):
    # Partially pivoted Adaptive Cross Approximation [Beb00].
    # Touches only the rows/columns it selects; O(rmax^2 (m+n)) arithmetic.
    m, n = A.shape
    U = np.zeros((m, rmax), dtype=complex)
    V = np.zeros((n, rmax), dtype=complex)
    used_rows = set()
    i = 0
    for t in range(rmax):
        row = A[i] - U[i, :t] @ V[:, :t].T
        j = int(np.argmax(np.abs(row)))
        piv = row[j]
        if abs(piv) < 1e-300:
            return U[:, :t], V[:, :t]
        V[:, t] = row / piv
        col = A[:, j] - U[:, :t] @ V[j, :t]
        U[:, t] = col
        used_rows.add(i)
        resid = np.abs(col)
        resid[list(used_rows)] = -1.0
        i = int(np.argmax(resid))
        if tol > 0 and np.linalg.norm(col) * np.linalg.norm(V[:, t]) < tol:
            return U[:, :t + 1], V[:, :t + 1]
    return U, V

def id_lowrank(A, r):
    # Interpolative decomposition [CGMR05]: A ~ A[:, idx[:r]] @ P.
    idx, proj = sli.interp_decomp(np.asfortranarray(A), r, rand=False)
    P = sli.reconstruct_interp_matrix(idx, proj)
    return A[:, idx[:r]] @ P

print("Algebraic low-rank toolkit defined: rsvd, cpqr_lowrank, aca_partial, id_lowrank")
""")

CELL_11_ALGEBRAIC_COMPARE = code(r"""from time import perf_counter

A_TEST = K_HELM[25.0]          # oscillatory block, k*delta = 25
normA = np.linalg.norm(A_TEST, 2)
sv = np.linalg.svd(A_TEST, compute_uv=False)
RANKS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]

def spec_err(Ar):
    return np.linalg.norm(A_TEST - Ar, 2) / normA

res = {"SVD (optimal)": [], "rSVD": [], "CPQR": [], "ACA": [], "ID": []}
for r in RANKS:
    res["SVD (optimal)"].append(sv[r] / sv[0])
    U, s, Vh = rsvd(A_TEST, r);            res["rSVD"].append(spec_err((U * s) @ Vh))
    res["CPQR"].append(spec_err(cpqr_lowrank(A_TEST, r)))
    Ua, Va = aca_partial(A_TEST, r);        res["ACA"].append(spec_err(Ua @ Va.T))
    res["ID"].append(spec_err(id_lowrank(A_TEST, r)))

timing = {}
R_T = 40
for name, fn in [("rSVD", lambda: rsvd(A_TEST, R_T)),
                 ("CPQR", lambda: cpqr_lowrank(A_TEST, R_T)),
                 ("ACA", lambda: aca_partial(A_TEST, R_T)),
                 ("ID", lambda: id_lowrank(A_TEST, R_T)),
                 ("full SVD", lambda: np.linalg.svd(A_TEST, compute_uv=False))]:
    t0 = perf_counter(); fn(); timing[name] = perf_counter() - t0

import pandas as pd
df_alg = pd.DataFrame(res, index=RANKS); df_alg.index.name = "rank"
df_alg.to_csv("results/algebraic_methods.csv")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 4.5))
styles = {"SVD (optimal)": "k-", "rSVD": "o--", "CPQR": "s--", "ACA": "^--", "ID": "v--"}
for name, errs in res.items():
    ax.semilogy(RANKS, errs, styles[name], label=name, ms=4)
ax.set_xlabel("rank $r$"); ax.set_ylabel("$||K - K_r||_2 / ||K||_2$")
ax.set_title(r"Algebraic low-rank methods on the Helmholtz block ($k\delta$=25)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig("figures/algebraic_methods.png", dpi=140, bbox_inches="tight")
plt.show()

print(f"{'method':<12}{'time at r=40 (s)':>18}")
print("-" * 30)
for name, t in timing.items():
    print(f"{name:<12}{t:>18.4f}")
print("Saved: figures/algebraic_methods.png, results/algebraic_methods.csv")
""")

CELL_12_RPCHOL = code(r"""# RPCholesky [CETW22] on a PSD proxy (Gaussian-kernel Gram matrix of the same
# source points). The same sampled-pivot idea, generalized to nonsymmetric
# matrices, is RPLU [GW26] - directly applicable to kernel blocks like K_HELM.
rng = np.random.default_rng(SEED)
SIG = 0.4
G = np.exp(-np.abs(y_res[:, None] - y_res[None, :])**2 / (2 * SIG**2))   # PSD, 400x400

def rpcholesky(A, r, rng):
    n = A.shape[0]
    F = np.zeros((n, r))
    d = np.diag(A).astype(float).copy()
    for t in range(r):
        p = np.clip(d, 0, None); p = p / p.sum()
        i = rng.choice(n, p=p)
        g = A[:, i] - F[:, :t] @ F[i, :t]
        F[:, t] = g / np.sqrt(max(g[i], 1e-300))
        d = d - F[:, t]**2
    return F

eigs = np.linalg.eigvalsh(G)[::-1]
trG = eigs.sum()
ranks_rp = [5, 10, 20, 40, 80]
tr_err, tr_opt = [], []
for r in ranks_rp:
    F = rpcholesky(G, r, rng)
    tr_err.append(max(np.trace(G) - np.sum(F * F), 1e-300) / trG)
    tr_opt.append(max(eigs[r:].sum(), 1e-300) / trG)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(ranks_rp, tr_err, "o-", label="RPCholesky (sampled pivots)")
ax.semilogy(ranks_rp, tr_opt, "k--", label="optimal (eigenvalue tail)")
ax.set_xlabel("rank $r$"); ax.set_ylabel("trace error  tr$(G - FF^T)/$tr$(G)$")
ax.set_title("Randomly pivoted Cholesky on a PSD Gaussian Gram matrix")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig("figures/rpcholesky.png", dpi=140, bbox_inches="tight")
plt.show()
print("RPCholesky needs only (r+1)*n entry evaluations - no full matrix.")
print("Saved: figures/rpcholesky.png")
""")

CELL_13_ANALYTIC = code(r"""# Analytic expansions vs the algebraic optimum, on the complex log kernel [M2D].
# - Taylor + balancing: stablefmmpy's BenchmarkSuite._log_UBV (rank r+1)
# - 2D tensor Chebyshev interpolation (bbFMM-style [FD09], rank p^2)
normL = np.linalg.norm(K_LOG, 2)
svL = np.linalg.svd(K_LOG, compute_uv=False)

def cheb_lagrange(pts, a, b, p):
    # Lagrange basis evaluated at pts for p Chebyshev nodes on [a, b].
    nodes = (a + b) / 2 + (b - a) / 2 * np.cos((2 * np.arange(p) + 1) * np.pi / (2 * p))
    L = np.ones((len(pts), p))
    for m_ in range(p):
        for n_ in range(p):
            if n_ != m_:
                L[:, m_] *= (pts - nodes[n_]) / (nodes[m_] - nodes[n_])
    return L, nodes

def cheb2d_lowrank(xpts, ypts, cx, cy, delta, p):
    # K ~ Sx @ K(nodes_x, nodes_y) @ Sy^T, storage rank p^2.
    Lxr, nxr = cheb_lagrange(xpts.real, cx.real - delta, cx.real + delta, p)
    Lxi, nxi = cheb_lagrange(xpts.imag, cx.imag - delta, cx.imag + delta, p)
    Lyr, nyr = cheb_lagrange(ypts.real, cy.real - delta, cy.real + delta, p)
    Lyi, nyi = cheb_lagrange(ypts.imag, cy.imag - delta, cy.imag + delta, p)
    Sx = (Lxr[:, :, None] * Lxi[:, None, :]).reshape(len(xpts), p * p)
    Sy = (Lyr[:, :, None] * Lyi[:, None, :]).reshape(len(ypts), p * p)
    gx = (nxr[:, None] + 1j * nxi[None, :]).ravel()
    gy = (nyr[:, None] + 1j * nyi[None, :]).ravel()
    Kn = np.log(gx[:, None] - gy[None, :])
    return Sx @ Kn @ Sy.T

cheb_p = [2, 3, 4, 5, 6, 7, 8]
cheb_rank = [p * p for p in cheb_p]
cheb_err = [np.linalg.norm(K_LOG - cheb2d_lowrank(x_res, y_res, 0j, 3 + 0j, DELTA_RES, p), 2) / normL
            for p in cheb_p]

taylor_rank, taylor_err = [], []
if AVAILABLE.get("stablefmmpy"):
    from stablefmmpy import PointSet, BenchmarkSuite
    PX, PY = PointSet(x_res), PointSet(y_res)
    for r in [3, 7, 11, 15, 23, 31, 47, 63]:
        U, B, V = BenchmarkSuite._log_UBV(PX, PY, r, balanced=True)
        taylor_rank.append(r + 1)
        taylor_err.append(np.linalg.norm(K_LOG - U @ B @ V.T, 2) / normL)
else:
    print("[SKIP] stablefmmpy not available - Taylor curve omitted")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4.5))
ranks_axis = np.arange(1, 70)
ax.semilogy(ranks_axis, svL[1:70] / svL[0], "k-", label="SVD optimum")
ax.semilogy(cheb_rank, cheb_err, "s--", label="Chebyshev 2D (rank $p^2$) [FD09]")
if taylor_rank:
    ax.semilogy(taylor_rank, taylor_err, "o--", label="Taylor + balancing (rank $r{+}1$) [M2D]")
ax.set_xlabel("storage rank"); ax.set_ylabel("$||K - K_r||_2/||K||_2$")
ax.set_title("Analytic expansions vs the algebraic optimum (log kernel)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig("figures/analytic_vs_algebraic.png", dpi=140, bbox_inches="tight")
plt.show()
print("Analytic ranks are a constant factor above optimal - the price of never touching K.")
print("Saved: figures/analytic_vs_algebraic.png")
""")

# =============================================================================
# S3 - Definitions module
# =============================================================================

CELL_14_DEFS_MD = md(r"""## 3. Definitions module — balancing and the two-regime architecture

**Kernel.** $\phi = Kq$ with $K_{ij} = H_0^{(1)}(k\,|x_i - y_j|)$, all points stored as
complex numbers $x + \mathrm{i}y \in \mathbb{C}$.

**The instability.** The low-frequency (Graf addition) expansion of $K$ contains Bessel
factors $J_p(k\delta) \sim \frac{1}{p!}\left(\frac{k\delta}{2}\right)^p$ and Hankel factors
$Y_p \sim \frac{(p-1)!}{\pi}\left(\frac{2}{k\delta}\right)^p$. For $k\delta \ll 1$ the naive
factors overflow **factorially** in the expansion order $r$: the translation matrix
$B$ reaches $10^{300}$ while the product $UBV^{\mathsf T}$ stays $O(1)$ — catastrophic
for backward stability.

**The fix ([HK] Eq. 2.17, Lemma 2.3).** Balance each basis column $p$ with

$$\lambda_{x,p} = \max\Big\{1,\; p!\,\Big(\tfrac{2}{k\delta}\Big)^{p}\Big\},$$

applied as $U \to U\Lambda^{-1}$, $B \to \Lambda_x B \Lambda_y$. Then
$\|U\|_{max} \le 1$ and $\|B\|_{max} = O(1)$ ([HK] Theorem 2.7), and the backward error
of the whole FMM grows only **logarithmically** in the matrix size.

**Two regimes.** The Bessel expansion is used only while it is the right tool:

| Regime | Condition | Basis | $B$ matrix |
|---|---|---|---|
| **LF** (sub-wavelength) | $k\delta \le r/e$ | balanced Bessel/Hankel, Miller backward recurrence | dense $(2r{+}1)\times(2r{+}1)$ |
| **HF** (oscillatory) | $k\delta > r/e$ | equispaced plane waves (DFT) [CD13] | **diagonal**, $2r{+}1$ entries, always finite |

The boundary $k\delta = r/e$ is the **switching level** measured experimentally in
Section 5.6.
""")

CELL_15_LAMBDA_PLOT = code(r"""import matplotlib.pyplot as plt

if AVAILABLE.get("stablefmmpy"):
    from stablefmmpy import ScalingFactors
    R_LAM = 60
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for kdelta in [0.25, 1.0, 5.0, 20.0]:
        sf = ScalingFactors(R_LAM, 1.0, kdelta)        # (r, k, delta): k*delta = kdelta
        log10_lam = sf.log_array() / np.log(10.0)      # always finite - never exp() this
        ax.plot(np.arange(R_LAM + 1), log10_lam, label=rf"$k\delta$ = {kdelta:g}")
    ax.set_xlabel("order $p$")
    ax.set_ylabel(r"$\log_{10} \lambda_p$")
    ax.set_title(r"Balancing factors $\lambda_p = \max(1,\; p!\,(2/k\delta)^p)$  [HK Eq. 2.17]")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("figures/lambda_factors.png", dpi=140, bbox_inches="tight")
    plt.show()
    print("At k*delta = 0.25 and p = 60 the naive factor would be 10^%.0f" %
          (ScalingFactors(60, 1.0, 0.25).log_array()[-1] / np.log(10.0)))
    print("Saved: figures/lambda_factors.png")
else:
    print("[SKIP] stablefmmpy not available")
""")

CELL_16_REGIME_MAP = code(r"""import matplotlib.pyplot as plt

r_ax = np.arange(1, 101)
kd_ax = np.logspace(-1.3, 2.3, 400)
LF = kd_ax[None, :] <= (r_ax[:, None] / np.e)          # boundary: k*delta = r/e

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.pcolormesh(kd_ax, r_ax, LF, cmap="RdYlBu", alpha=0.55, shading="auto")
ax.plot(r_ax / np.e, r_ax, "k-", lw=2, label=r"switching level  $k\delta = r/e$")
ax.set_xscale("log")
ax.set_xlabel(r"$k\delta$ (box size in wavenumbers)")
ax.set_ylabel("expansion order $r$")
ax.text(0.3, 75, "LF regime\n(balanced Bessel,\ndense B)", fontsize=10)
ax.text(40, 18, "HF regime\n(plane waves,\ndiagonal B)", fontsize=10)
ax.set_title("Two-regime map of the wideband FMM [HK]")
ax.legend(loc="lower right")

if AVAILABLE.get("stablefmmpy"):
    from stablefmmpy import LeafMatrices
    probes = [(20, 1.0), (20, 12.0), (60, 10.0), (10, 50.0)]
    agree = all(LeafMatrices(r_, 1.0).regime(kd_) == ("lf" if kd_ <= r_ / np.e else "hf")
                for r_, kd_ in probes)
    print(f"LeafMatrices.regime agrees with the analytic boundary on {len(probes)} probes: {agree}")

fig.tight_layout(); fig.savefig("figures/regime_map.png", dpi=140, bbox_inches="tight")
plt.show()
print("Saved: figures/regime_map.png")
""")

# =============================================================================
# S4 - Benchmark harness
# =============================================================================

CELL_17_HARNESS_MD = md(r"""## 4. Benchmark harness

**Common dataset.** Sources and targets are two independent uniform clouds of $N$
points each in the unit square (as complex numbers), with complex Gaussian charges —
identical seed for every library.

**Kernel-convention normalization.** `stablefmmpy` evaluates the *bare* kernel
$H_0^{(1)}(kr)$; `pyfmmlib` and `fmm2dpy` evaluate the Green's function
$\tfrac{\mathrm i}{4}H_0^{(1)}(kr)$. Every adapter therefore multiplies the Fortran
output by $4/\mathrm{i}$, and the **convention gate** (Section 4.3) verifies the match
to near machine precision *before* any benchmark is trusted.

**Timing protocol.** `time.perf_counter`, `WARMUP` warmup call(s), best of `REPEATS`.
All libraries are measured as **one-shot setup + apply**: `stablefmmpy` rebuilds its
quadtree inside every `solve()`, and the Fortran codes plan internally per call, so
this is the only apples-to-apples protocol. Calls whose first run exceeds 10 s are
timed once (no repeats).

**Ground truth.** Full dense $H_0^{(1)}$ matvec up to $N \le$ `DENSE_MAX`; beyond that,
the error is estimated on `SUBSET_M` random targets (standard practice in the FMM
literature) and timing baselines are extrapolated.

**Accuracy knobs are not the same.** `stablefmmpy` fixes the expansion order $r$;
`pyfmmlib` uses the integer `iprec` ($2 \approx 10^{-6}$, $4 \approx 10^{-12}$);
`fmm2dpy` takes a tolerance `eps`. The Pareto plot (Section 5.3) puts all three on the
common axis that actually matters: achieved error vs time.
""")

CELL_18_ADAPTERS = code(r"""from time import perf_counter
from scipy.special import hankel1

# ---- coordinate conversions: complex (N,) -> real layouts -----------------
def c2xy(z):            # (N, 2) row-per-point  [pyfmmlib]
    return np.column_stack([z.real, z.imag])

def c2rows(z):          # (2, N) coordinate-per-row  [fmm2dpy]
    return np.vstack([z.real, z.imag])

GREEN_NORM = 4.0 / 1j   # bare H0 = (4/i) * Green's function (i/4) H0

def rel_err(a, e):
    a = np.asarray(a); e = np.asarray(e)
    return float(np.linalg.norm(a - e) / np.linalg.norm(e))

def make_sets(n, seed=SEED):
    rng = np.random.default_rng(seed)
    src = rng.random(n) + 1j * rng.random(n)
    tgt = rng.random(n) + 1j * rng.random(n)
    q = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return src, tgt, q

# ---- ground truth ----------------------------------------------------------
def eval_dense(src, tgt, q, k, block=1024):
    # Blocked O(MN) direct sum of H0^(1)(k|t - s|) q; memory O(block*N).
    out = np.empty(len(tgt), dtype=complex)
    for i0 in range(0, len(tgt), block):
        D = np.abs(tgt[i0:i0 + block, None] - src[None, :])
        out[i0:i0 + block] = hankel1(0, k * D) @ q
    return out

def eval_dense_subset(src, tgt, q, k, m=SUBSET_M, seed=SEED):
    idx = np.random.default_rng(seed).choice(len(tgt), size=m, replace=False)
    return idx, eval_dense(src, tgt[idx], q, k)

# ---- library adapters (each raises if its library is unavailable) ----------
def eval_stablefmmpy(src, tgt, q, k, r=R_STABLE, multilevel=False, balanced=True):
    from stablefmmpy import FMMSolver, PointSet
    solver = FMMSolver(k, r, tau=0.6, N0=32, balanced=balanced, multilevel=multilevel)
    return solver.solve(PointSet(tgt), PointSet(src), np.asarray(q, dtype=complex))

def eval_pyfmmlib(src, tgt, q, k, iprec=2):
    from pyfmmlib import HelmholtzKernel as _PK
    from pyfmmlib import fmm_part
    pot = fmm_part("P", iprec=iprec, kernel=_PK(k), sources=c2xy(src),
                   mop_charge=np.asarray(q, dtype=complex), target=c2xy(tgt))
    return np.asarray(pot) * GREEN_NORM

def eval_fmm2dpy(src, tgt, q, k, eps=1e-9):
    import fmm2dpy as _f2
    out = _f2.hfmm2d(eps=eps, zk=complex(k), sources=c2rows(src),
                     charges=np.asarray(q, dtype=complex),
                     targets=c2rows(tgt), pgt=1)
    return np.asarray(out.pottarg) * GREEN_NORM

# ---- timing ------------------------------------------------------------------
def time_call(fn, *args, repeats=REPEATS, warmup=WARMUP, slow_cutoff=10.0, **kw):
    t0 = perf_counter(); out = fn(*args, **kw); t_first = perf_counter() - t0
    if t_first > slow_cutoff:
        return t_first, out                       # too slow to repeat
    for _ in range(warmup):
        fn(*args, **kw)
    best = t_first
    for _ in range(repeats - 1):
        t0 = perf_counter(); fn(*args, **kw); best = min(best, perf_counter() - t0)
    return best, out

bench_rows = []      # every benchmark cell appends dicts here
print("Harness ready: eval_dense, eval_stablefmmpy, eval_pyfmmlib, eval_fmm2dpy, time_call")
""")

CELL_19_NUMBA_ASIDE = code(r"""# Aside: what numba can and cannot accelerate here.
# scipy.special.hankel1 is NOT supported in numba's nopython mode, so the dense
# HELMHOLTZ baseline stays vectorized scipy. The LAPLACE (log) direct sum jits fine:
if AVAILABLE.get("numba"):
    from numba import njit

    @njit(cache=False)
    def direct_log_numba(sr, si, tr, ti, q):
        out = np.zeros(len(tr))
        for i in range(len(tr)):
            acc = 0.0
            for j in range(len(sr)):
                dx = tr[i] - sr[j]; dy = ti[i] - si[j]
                acc += 0.5 * np.log(dx * dx + dy * dy) * q[j]
            out[i] = acc
        return out

    src, tgt, q = make_sets(3000)
    qr = q.real.copy()
    direct_log_numba(src.real, src.imag, tgt.real, tgt.imag, qr)      # JIT compile
    t_nb, _ = time_call(direct_log_numba, src.real, src.imag, tgt.real, tgt.imag, qr)
    t_np, _ = time_call(lambda: np.log(np.abs(tgt[:, None] - src[None, :])) @ qr)
    print(f"log-kernel direct sum, N=3000:  numba {t_nb*1e3:7.1f} ms   numpy {t_np*1e3:7.1f} ms")
else:
    print("[SKIP] numba not available")
""")

CELL_20_CONVENTION_CHECK = code(r"""# Convention gate: all libraries must agree with the bare-H0 dense sum at N=200
# to ~1e-12 AFTER the (i/4) Green's-function normalization. Gates the benchmarks.
N_GATE, K_GATE = 200, 10.0
src, tgt, q = make_sets(N_GATE)
phi_ref = eval_dense(src, tgt, q, K_GATE)
CONVENTION_OK = {}

def _gate(name, fn, tol):
    try:
        phi = fn()
        s_fit = np.vdot(phi_ref, phi) / np.vdot(phi_ref, phi_ref)   # ~1 if convention right
        e = rel_err(phi, phi_ref)
        ok = e < tol
        CONVENTION_OK[name] = ok
        print(f"[CHECK] convention {name}: {'PASS' if ok else 'FAIL'} "
              f"(rel_err={e:.2e}, fitted scale={s_fit:.6f})")
    except Exception as exc:
        CONVENTION_OK[name] = False
        print(f"[SKIP] convention {name}: {type(exc).__name__}: {exc}")

if AVAILABLE.get("pyfmmlib"):
    _gate("pyfmmlib", lambda: eval_pyfmmlib(src, tgt, q, K_GATE, iprec=5), 1e-10)
else:
    print("[SKIP] convention pyfmmlib: library not available")
if AVAILABLE.get("fmm2dpy"):
    _gate("fmm2dpy", lambda: eval_fmm2dpy(src, tgt, q, K_GATE, eps=1e-12), 1e-10)
else:
    print("[SKIP] convention fmm2dpy: library not available")
if AVAILABLE.get("stablefmmpy"):
    # sanity gate only: on this mixed unit-square geometry the leaf-level FMM sits
    # around 1e-6; the accuracy-vs-r story is measured properly in Section 5.
    _gate("stablefmmpy", lambda: eval_stablefmmpy(src, tgt, q, K_GATE, r=40), 1e-4)
else:
    print("[SKIP] convention stablefmmpy: library not available")
""")

CELL_21_BENCH_STABLE = code(r"""# stablefmmpy N sweep (balanced, r=R_STABLE, k=K_BENCH). The solver is pure
# Python, so the sweep self-caps: it stops extending N once a solve is projected
# to exceed 180 s.
if AVAILABLE.get("stablefmmpy") and CONVENTION_OK.get("stablefmmpy"):
    t_prev, n_prev = None, None
    for n in N_SWEEP_STABLE:
        if t_prev is not None and t_prev * (n / n_prev) ** 1.3 > 180.0:
            print(f"[INFO] skipping N >= {n} (projected > 180 s per solve)")
            break
        src, tgt, q = make_sets(n)
        t, phi = time_call(eval_stablefmmpy, src, tgt, q, K_BENCH,
                           repeats=1 if n >= 2000 else REPEATS)
        e = rel_err(phi, eval_dense(src, tgt, q, K_BENCH))
        bench_rows.append({"library": "stablefmmpy", "N": n, "k": K_BENCH,
                           "param": f"r={R_STABLE}", "time_s": t, "rel_err": e})
        print(f"stablefmmpy  N={n:>6}  t={t:8.3f} s  rel_err={e:.2e}")
        t_prev, n_prev = t, n

    # one multilevel data point for reference
    try:
        n_ml = 1000
        src, tgt, q = make_sets(n_ml)
        t, phi = time_call(eval_stablefmmpy, src, tgt, q, K_BENCH,
                           multilevel=True, repeats=1)
        e = rel_err(phi, eval_dense(src, tgt, q, K_BENCH))
        bench_rows.append({"library": "stablefmmpy-ML", "N": n_ml, "k": K_BENCH,
                           "param": f"r={R_STABLE},multilevel", "time_s": t, "rel_err": e})
        print(f"stablefmmpy  N={n_ml:>6}  t={t:8.3f} s  rel_err={e:.2e}  (multilevel=True)")
    except TypeError:
        print("[INFO] this stablefmmpy version has no multilevel mode")
else:
    print("[SKIP] stablefmmpy benchmarks")
""")

CELL_22_BENCH_PYFMMLIB = code(r"""if AVAILABLE.get("pyfmmlib") and CONVENTION_OK.get("pyfmmlib"):
    for n in N_SWEEP_FORTRAN:
        src, tgt, q = make_sets(n)
        t, phi = time_call(eval_pyfmmlib, src, tgt, q, K_BENCH, iprec=2)
        if n <= DENSE_MAX:
            e = rel_err(phi, eval_dense(src, tgt, q, K_BENCH))
        else:
            idx, phi_sub = eval_dense_subset(src, tgt, q, K_BENCH)
            e = rel_err(phi[idx], phi_sub)
        bench_rows.append({"library": "pyfmmlib", "N": n, "k": K_BENCH,
                           "param": "iprec=2", "time_s": t, "rel_err": e})
        print(f"pyfmmlib     N={n:>6}  t={t:8.3f} s  rel_err={e:.2e}")

    # accuracy-knob sweep at fixed N
    src, tgt, q = make_sets(N_FIX)
    phi_ref = eval_dense(src, tgt, q, K_BENCH)
    pyfmmlib_pareto = []
    for ip in IPREC_SWEEP:
        t, phi = time_call(eval_pyfmmlib, src, tgt, q, K_BENCH, iprec=ip)
        pyfmmlib_pareto.append({"iprec": ip, "time_s": t, "rel_err": rel_err(phi, phi_ref)})
        print(f"pyfmmlib     N={N_FIX}  iprec={ip:>2}  t={t:7.3f} s  rel_err={pyfmmlib_pareto[-1]['rel_err']:.2e}")
else:
    pyfmmlib_pareto = []
    print("[SKIP] pyfmmlib benchmarks")
""")

CELL_23_BENCH_FMM2DPY = code(r"""if AVAILABLE.get("fmm2dpy") and CONVENTION_OK.get("fmm2dpy"):
    for n in N_SWEEP_FORTRAN:
        src, tgt, q = make_sets(n)
        t, phi = time_call(eval_fmm2dpy, src, tgt, q, K_BENCH, eps=1e-9)
        if n <= DENSE_MAX:
            e = rel_err(phi, eval_dense(src, tgt, q, K_BENCH))
        else:
            idx, phi_sub = eval_dense_subset(src, tgt, q, K_BENCH)
            e = rel_err(phi[idx], phi_sub)
        bench_rows.append({"library": "fmm2dpy", "N": n, "k": K_BENCH,
                           "param": "eps=1e-9", "time_s": t, "rel_err": e})
        print(f"fmm2dpy      N={n:>6}  t={t:8.3f} s  rel_err={e:.2e}")

    src, tgt, q = make_sets(N_FIX)
    phi_ref = eval_dense(src, tgt, q, K_BENCH)
    fmm2dpy_pareto = []
    for ep in EPS_SWEEP:
        t, phi = time_call(eval_fmm2dpy, src, tgt, q, K_BENCH, eps=ep)
        fmm2dpy_pareto.append({"eps": ep, "time_s": t, "rel_err": rel_err(phi, phi_ref)})
        print(f"fmm2dpy      N={N_FIX}  eps={ep:.0e}  t={t:7.3f} s  rel_err={fmm2dpy_pareto[-1]['rel_err']:.2e}")
else:
    fmm2dpy_pareto = []
    print("[SKIP] fmm2dpy benchmarks")
""")

CELL_24_BENCH_TABLE = code(r"""# Dense-baseline timings + consolidated benchmark table.
import pandas as pd

for n in [500, 1000, 2000, 4000]:
    src, tgt, q = make_sets(n)
    t, _ = time_call(eval_dense, src, tgt, q, K_BENCH, repeats=1)
    bench_rows.append({"library": "dense O(N^2)", "N": n, "k": K_BENCH,
                       "param": "-", "time_s": t, "rel_err": 0.0})
    print(f"dense        N={n:>6}  t={t:8.3f} s")

df_bench = pd.DataFrame(bench_rows)
df_bench.to_csv("results/benchmark_runs.csv", index=False)
print()
print(df_bench.to_string(index=False))
print("Saved: results/benchmark_runs.csv")
""")

# =============================================================================
# S5 - Visualization
# =============================================================================

CELL_25_PLOT_TIME = code(r"""import matplotlib.pyplot as plt

# Published single-core reference: Gimbutas & Greengard, hfmm2dpart, k=20,
# iprec=4 (~12 digits), Intel Xeon E5-2687W 3.10 GHz [GG15, Table 3].
FMMLIB_PAPER_TABLE3 = {10_000: 0.306, 100_000: 2.614, 1_000_000: 25.685}

fig, ax = plt.subplots(figsize=(7.5, 5))
mark = {"stablefmmpy": "o-", "stablefmmpy-ML": "d-", "pyfmmlib": "s-",
        "fmm2dpy": "^-", "dense O(N^2)": "k--"}
for lib, g in df_bench.groupby("library"):
    g = g.sort_values("N")
    ax.loglog(g["N"], g["time_s"], mark.get(lib, "x-"), label=lib, ms=5)

ax.loglog(list(FMMLIB_PAPER_TABLE3), list(FMMLIB_PAPER_TABLE3.values()), "*",
          color="gray", ms=12, label="hfmm2d paper, 1 core Xeon [GG15 T3]")
nn = np.array([250, 40000])
ax.loglog(nn, 2e-5 * nn, ":", color="gray")
ax.text(28000, 0.45, "O(N)", color="gray")
nn2 = np.array([250, 5000])
ax.loglog(nn2, 3.2e-8 * nn2**2, ":", color="gray")
ax.text(3600, 0.9, "O($N^2$)", color="gray")

ax.set_xlabel("N (sources = targets)"); ax.set_ylabel("wall time (s), setup + apply")
ax.set_title(f"Scalability: time vs N   (k = {K_BENCH:g})")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/time_vs_N.png", dpi=140, bbox_inches="tight")
plt.show()
print("Saved: figures/time_vs_N.png")
""")

CELL_26_ACC_VS_K = code(r"""# Accuracy vs wavenumber. The Fortran codes hold their tolerance fixed across
# the full K_SWEEP at N_FIX. stablefmmpy (pure Python) is measured on a smaller
# cloud (N_STABLE) and only up to k=25 with a hard time budget per configuration:
# at higher k its leaf-only wideband path degenerates to P2P fallbacks and a
# single solve can take tens of minutes - a cost, not an accuracy, limitation.
from time import perf_counter

N_STABLE = 500
STABLE_K_MAX = 25.0
STABLE_BUDGET_S = 150.0

ksweep_rows = []
src, tgt, q = make_sets(N_FIX)
src_s, tgt_s, q_s = make_sets(N_STABLE)
stable_alive = {25: True, 50: True}
for k in K_SWEEP:
    phi_ref = eval_dense(src, tgt, q, k)
    if (AVAILABLE.get("stablefmmpy") and CONVENTION_OK.get("stablefmmpy")
            and k <= STABLE_K_MAX):
        phi_ref_s = eval_dense(src_s, tgt_s, q_s, k)
        for r in (25, 50):
            if not stable_alive[r]:
                continue
            try:
                t0 = perf_counter()
                phi = eval_stablefmmpy(src_s, tgt_s, q_s, k, r=r)
                dt = perf_counter() - t0
                ksweep_rows.append({"library": f"stablefmmpy r={r} (N={N_STABLE})",
                                    "k": k, "rel_err": rel_err(phi, phi_ref_s)})
                if dt > STABLE_BUDGET_S:
                    stable_alive[r] = False
                    print(f"[INFO] stablefmmpy r={r} took {dt:.0f} s at k={k:g}; "
                          f"skipping higher k for this r")
            except Exception as exc:
                print(f"[WARN] stablefmmpy r={r} k={k}: {type(exc).__name__}")
    if AVAILABLE.get("pyfmmlib") and CONVENTION_OK.get("pyfmmlib"):
        phi = eval_pyfmmlib(src, tgt, q, k, iprec=3)
        ksweep_rows.append({"library": "pyfmmlib iprec=3", "k": k,
                            "rel_err": rel_err(phi, phi_ref)})
    if AVAILABLE.get("fmm2dpy") and CONVENTION_OK.get("fmm2dpy"):
        phi = eval_fmm2dpy(src, tgt, q, k, eps=1e-9)
        ksweep_rows.append({"library": "fmm2dpy eps=1e-9", "k": k,
                            "rel_err": rel_err(phi, phi_ref)})
    print(f"k = {k:>6g} done")

import pandas as pd
import matplotlib.pyplot as plt
df_k = pd.DataFrame(ksweep_rows)
df_k.to_csv("results/accuracy_vs_k.csv", index=False)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for lib, g in df_k.groupby("library"):
    g = g.sort_values("k")
    ax.loglog(g["k"], np.maximum(g["rel_err"], 1e-17), "o-", label=lib, ms=4)
ax.axhline(1e-14, color="gray", ls="--", lw=0.8)
ax.text(K_SWEEP[0], 1.6e-14, "machine precision", color="gray", fontsize=8)
ax.set_xlabel("wavenumber $k$"); ax.set_ylabel("relative error vs dense")
ax.set_title(f"Accuracy vs wavenumber  (N = {N_FIX})")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/accuracy_vs_k.png", dpi=140, bbox_inches="tight")
plt.show()
print("Saved: figures/accuracy_vs_k.png, results/accuracy_vs_k.csv")
""")

CELL_27_PARETO = code(r"""# What accuracy costs: achieved error vs wall time at fixed N, sweeping each
# library's own accuracy knob (r / iprec / eps).
import matplotlib.pyplot as plt

src, tgt, q = make_sets(N_FIX)
phi_ref = eval_dense(src, tgt, q, K_BENCH)

stable_pareto = []
if AVAILABLE.get("stablefmmpy") and CONVENTION_OK.get("stablefmmpy"):
    # pure-Python solver: measured on the smaller N_STABLE cloud with a time
    # budget - once a solve exceeds it, higher r values are skipped.
    src_s, tgt_s, q_s = make_sets(N_STABLE)
    phi_ref_s = eval_dense(src_s, tgt_s, q_s, K_BENCH)
    for r in R_PARETO:
        t, phi = time_call(eval_stablefmmpy, src_s, tgt_s, q_s, K_BENCH, r=r, repeats=1)
        stable_pareto.append({"r": r, "time_s": t, "rel_err": rel_err(phi, phi_ref_s)})
        print(f"stablefmmpy  r={r:>3}  t={t:7.3f} s  rel_err={stable_pareto[-1]['rel_err']:.2e}")
        if t > STABLE_BUDGET_S:
            print(f"[INFO] stablefmmpy r={r} took {t:.0f} s; stopping the r sweep")
            break

fig, ax = plt.subplots(figsize=(7, 4.8))
if stable_pareto:
    ax.loglog([d["time_s"] for d in stable_pareto],
              [max(d["rel_err"], 1e-17) for d in stable_pareto], "o-",
              label=f"stablefmmpy (r = 5...40, N = {N_STABLE})")
if pyfmmlib_pareto:
    ax.loglog([d["time_s"] for d in pyfmmlib_pareto],
              [max(d["rel_err"], 1e-17) for d in pyfmmlib_pareto], "s-",
              label=f"pyfmmlib (iprec = -1...5, N = {N_FIX})")
if fmm2dpy_pareto:
    ax.loglog([d["time_s"] for d in fmm2dpy_pareto],
              [max(d["rel_err"], 1e-17) for d in fmm2dpy_pareto], "^-",
              label=f"fmm2dpy (eps = 1e-3...1e-12, N = {N_FIX})")
ax.set_xlabel("wall time (s)"); ax.set_ylabel("achieved relative error")
ax.set_title(f"Accuracy-cost Pareto fronts  (k = {K_BENCH:g}; N noted per library)")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig("figures/pareto.png", dpi=140, bbox_inches="tight")
plt.show()
print("Saved: figures/pareto.png")
""")

CELL_28_STAB_MD = md(r"""## 5.4 Stability: balanced $10^{-14}$ errors where naive factors overflow

This is the core claim of [HK]/[M2D], reproduced with `stablefmmpy`'s
`BenchmarkSuite` on the papers' own single-level disc geometry ($\tau = 1/3$):

- **[HK] Table 6.1** — $k=100$, sets scaled by $0.0025$ ($k\delta = 0.25$, deep LF):
  the *stable* balance factor $\mathcal{B} = \|B\|_{max}$ stays $\approx 0.5287$ and the
  error $\approx 5.5\times10^{-15}$ for every $r$, while the *naive* $\mathcal{B}$ blows
  through $10^{190} \to 10^{296} \to$ `Inf` at $r \ge 180$ (error becomes `NaN`).
- **[M2D] Tables 6.1–6.2** — Cauchy kernel ($10^{-4}$ scaling, naive `Inf` from
  $r = 70$) and log kernel ($10^{2}$ scaling, naive `Inf` from $r = 90$), balanced
  errors $4.6\times10^{-15}$ / $1.3\times10^{-14}$.

`StabilityAnalyzer` additionally verifies the norm bound $\|U\|_{max} \le 1$
([HK] Theorem 2.7) and localizes the exact overflow threshold $r^*$.

**Calibration note.** `BenchmarkSuite` reproduces the papers on a *single-level*
disc geometry with $\tau = 1/3$, whereas the published tables are measured inside a
full $\tau = 0.6$, $L = 8$ hierarchical tree. The library's own design log records
the consequences: the stable balance factor lands $\sim$15–20% above the paper's
value (here $\approx 0.63$ vs $0.5287$ — the essential property is that it is
**constant in $r$**), the naive overflow threshold shifts earlier
($r^* \approx 90$–$120$ vs $180$), and the log-kernel naive path does *not*
overflow at a single level (that requires the full tree). The `[CHECK]` lines
below therefore test the library-level invariants, with the paper values printed
as reference.
""")

CELL_29_STAB_RUN = code(r"""if AVAILABLE.get("stablefmmpy"):
    from stablefmmpy import BenchmarkSuite, StabilityAnalyzer

    bs = BenchmarkSuite()
    print("Running [HK] Table 6.1 (k=100, scale=0.0025) ...")
    rows_hk61 = bs.run_helmholtz_table61(r_values=[120, 140, 160, 180, 200])
    print("Running [HK] Table 6.2 (k=10, scale=0.01) ...")
    rows_hk62 = bs.run_helmholtz_table62()
    print("Running [M2D] Table 6.1 (Cauchy, scale=1e-4) ...")
    rows_m2d61 = bs.run_multipole2d_table61()
    print("Running [M2D] Table 6.2 (log, scale=100) ...")
    rows_m2d62 = bs.run_multipole2d_table62()

    def show(rows, cols):
        hdr = "".join(f"{c:>14}" for c in cols)
        print(hdr); print("-" * len(hdr))
        for row in rows:
            line = ""
            for c in cols:
                v = row[c]
                line += f"{v:>14}" if isinstance(v, int) else (
                    f"{v:>14.4e}" if np.isfinite(v) else f"{'Inf':>14}")
            print(line)
        print()

    print("[HK] Table 6.1 reproduction:")
    show(rows_hk61, ["r", "stable_B", "stable_err", "naive_B", "naive_err"])
    print("[M2D] Table 6.1 reproduction (Cauchy):")
    show(rows_m2d61, ["r", "balanced_err", "naive_err"])
    print("[M2D] Table 6.2 reproduction (log):")
    show(rows_m2d62, ["r", "balanced_err", "naive_err"])

    # norm bounds + overflow threshold on the [HK] T6.1 geometry
    X61, Y61, q61 = BenchmarkSuite._make_helmholtz_clusters(50, 0.0025, 42)
    sa = StabilityAnalyzer(k=100.0, r=200)
    nb = sa.verify_norm_bounds(X61, Y61)
    r_star = sa.find_overflow_threshold(X61, Y61, r_max=200, step=10)
    print(f"||U||_max (balanced) = {nb['U_max_balanced']:.6f}   "
          f"theorem ||U||_max <= 1 satisfied: {nb['theorem_satisfied']}")
    ok_r = 60 <= r_star <= 140
    print(f"[CHECK] naive overflow threshold r* = {r_star} "
          f"(expected 80-130 for this single-level tau=1/3 geometry; "
          f"the paper's 180 belongs to the tau=0.6 L=8 tree): "
          f"{'PASS' if ok_r else 'FAIL'}")
    stable_errs = [row["stable_err"] for row in rows_hk61]
    ok_e = all(np.isfinite(e) and e <= 1e-12 for e in stable_errs)
    print(f"[CHECK] balanced err <= 1e-12 for all r in HK T6.1: "
          f"{'PASS' if ok_e else 'FAIL'} (max = {max(stable_errs):.2e})")

    import pandas as pd
    pd.DataFrame(rows_hk61).to_csv("results/stability_hk_t61.csv", index=False)
    pd.DataFrame(rows_m2d61).to_csv("results/stability_m2d_t61.csv", index=False)
    pd.DataFrame(rows_m2d62).to_csv("results/stability_m2d_t62.csv", index=False)
    print("Saved: results/stability_*.csv")
else:
    rows_hk61 = rows_hk62 = rows_m2d61 = rows_m2d62 = []
    print("[SKIP] stability experiments (stablefmmpy not available)")
""")

CELL_30_STAB_PLOTS = code(r"""if rows_hk61:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    r61 = [row["r"] for row in rows_hk61]
    ax1.semilogy(r61, [row["stable_err"] for row in rows_hk61], "o-",
                 color="tab:blue", label="[HK] T6.1 balanced")
    naive61 = [row["naive_err"] for row in rows_hk61]
    fin = [(r, e) for r, e in zip(r61, naive61) if np.isfinite(e)]
    inf_ = [r for r, e in zip(r61, naive61) if not np.isfinite(e)]
    if fin:
        ax1.semilogy(*zip(*fin), "s--", color="tab:red", label="[HK] T6.1 naive")
    if inf_:
        ax1.semilogy(inf_, [5e-1] * len(inf_), "rv", ms=10, label="naive overflow (Inf/NaN)")
    rm1 = [row["r"] for row in rows_m2d61]
    ax1.semilogy(rm1, [max(row["balanced_err"], 1e-16) for row in rows_m2d61], "d-",
                 color="tab:green", label="[M2D] T6.1 balanced (Cauchy)")
    nv = [row["naive_err"] for row in rows_m2d61]
    inf2 = [r for r, e in zip(rm1, nv) if not np.isfinite(e)]
    if inf2:
        ax1.semilogy(inf2, [2e-1] * len(inf2), "v", color="darkgreen", ms=8,
                     label="naive overflow (Cauchy)")
    ax1.axhline(1e-14, color="gray", ls=":", lw=0.8)
    ax1.set_xlabel("expansion order $r$"); ax1.set_ylabel("relative error")
    ax1.set_title("Balanced errors stay ~1e-14; naive overflows")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=7)

    ax2.semilogy(r61, [row["stable_B"] for row in rows_hk61], "o-",
                 color="tab:blue", label="stable  $||B||_{max}$")
    nb61 = [row["naive_B"] for row in rows_hk61]
    finB = [(r, b) for r, b in zip(r61, nb61) if np.isfinite(b)]
    infB = [r for r, b in zip(r61, nb61) if not np.isfinite(b)]
    if finB:
        ax2.semilogy(*zip(*finB), "s--", color="tab:red", label="naive  $||B||_{max}$")
    if infB:
        ax2.semilogy(infB, [1e300] * len(infB), "rv", ms=10, label="naive Inf")
    ax2.set_xlabel("expansion order $r$"); ax2.set_ylabel("balance factor $||B||_{max}$")
    ax2.set_title("[HK] T6.1: balance factor, stable vs naive")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("figures/stability_error_vs_r.png", dpi=140, bbox_inches="tight")
    plt.show()
    print("Saved: figures/stability_error_vs_r.png")
else:
    print("[SKIP] stability plots")
""")

CELL_31_M2L_MEMORY = code(r"""# M2L (B-matrix) memory footprint: dense (2r+1)^2 in the LF regime vs diagonal
# 2r+1 in the HF regime. stablefmmpy *stores* the HF B as a dense array with only
# the diagonal nonzero, so we report both as-stored and as-required bytes.
if AVAILABLE.get("stablefmmpy"):
    from stablefmmpy import HelmholtzKernel, LeafMatrices, PointSet

    rng = np.random.default_rng(SEED)
    N_M2L = 200
    mem_rows = []
    for r in [10, 20, 40, 80, 160]:
        for regime_name, kdelta in [("lf", 0.5), ("hf", 2.0 * r)]:
            delta = 0.05
            k = kdelta / delta
            X = PointSet(disk_points(N_M2L, 0.0j, delta, rng))
            Y = PointSet(disk_points(N_M2L, 6.0 * delta + 0.0j, delta, rng))
            lm = LeafMatrices(r, k)
            assert lm.regime(delta) == regime_name, (r, kdelta)
            U, B, V = lm.factorize(X, Y, balanced=True, regime=regime_name)
            q = rng.standard_normal(N_M2L) + 1j * rng.standard_normal(N_M2L)
            e = rel_err(U @ (B @ (V.T @ q)), HelmholtzKernel(k).matvec(X, Y, q))
            mem_rows.append({"r": r, "regime": regime_name, "kdelta": kdelta,
                             "B_stored_bytes": B.nbytes,
                             "B_required_bytes": 16 * int(np.count_nonzero(B)),
                             "rel_err": e})
            print(f"r={r:>4} {regime_name}  k*delta={kdelta:>6g}  stored={B.nbytes:>10,} B"
                  f"  required={16*int(np.count_nonzero(B)):>10,} B  rel_err={e:.1e}")

    import matplotlib.pyplot as plt
    import pandas as pd
    df_mem = pd.DataFrame(mem_rows)
    df_mem.to_csv("results/m2l_memory.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for regime_name, style, lbl in [("lf", "o-", "LF dense B: required = stored"),
                                    ("hf", "^-", "HF diagonal B: required")]:
        g = df_mem[df_mem.regime == regime_name]
        ax.loglog(g["r"], g["B_required_bytes"], style, label=lbl)
    g = df_mem[df_mem.regime == "hf"]
    ax.loglog(g["r"], g["B_stored_bytes"], "^--", alpha=0.5,
              label="HF B as stored (dense array) - implementation overhead")
    rr = np.array([10, 160])
    ax.loglog(rr, 16 * (2 * rr + 1.0) ** 2, ":", color="gray"); ax.text(90, 3.5e6, "$O(r^2)$", color="gray")
    ax.loglog(rr, 16 * (2 * rr + 1.0), ":", color="gray"); ax.text(100, 9e3, "$O(r)$", color="gray")
    ax.set_xlabel("expansion order $r$"); ax.set_ylabel("M2L operator size (bytes)")
    ax.set_title("M2L memory: dense LF vs diagonal HF translation matrix")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/m2l_memory.png", dpi=140, bbox_inches="tight")
    plt.show()
    print("Saved: figures/m2l_memory.png, results/m2l_memory.csv")
else:
    print("[SKIP] M2L memory experiment")
""")

CELL_32_SWITCHING = code(r"""# Switching level: fix r = 40 (boundary k*delta = r/e ~ 14.72), sweep k across it
# and factorize the same block three ways: auto, forced LF, forced HF.
if AVAILABLE.get("stablefmmpy"):
    from stablefmmpy import HelmholtzKernel, LeafMatrices, PointSet

    R_SW = 40
    BOUNDARY = R_SW / np.e
    rng = np.random.default_rng(SEED)
    X = PointSet(disk_points(200, 0.0j, 1.0, rng))
    Y = PointSet(disk_points(200, 6.0 + 0.0j, 1.0, rng))
    q = rng.standard_normal(200) + 1j * rng.standard_normal(200)

    switch_rows = []
    for k in [1, 2, 4, 8, 12, 14, 14.7, 16, 20, 32, 64]:
        lm = LeafMatrices(R_SW, float(k))
        phi_ref = HelmholtzKernel(float(k)).matvec(X, Y, q)
        row = {"kdelta": float(k), "auto_regime": lm.regime(1.0)}
        for regime in ["auto", "lf", "hf"]:
            try:
                U, B, V = lm.factorize(X, Y, balanced=True, regime=regime)
                row[f"err_{regime}"] = rel_err(U @ (B @ (V.T @ q)), phi_ref)
                row[f"bytes_{regime}"] = 16 * int(np.count_nonzero(B))
            except Exception:
                row[f"err_{regime}"] = np.nan
        switch_rows.append(row)
        print(f"k*delta={k:>6}  auto->{row['auto_regime']}  "
              f"err auto={row['err_auto']:.1e}  lf={row['err_lf']:.1e}  hf={row['err_hf']:.1e}")

    flip = [row["kdelta"] for row in switch_rows if row["auto_regime"] == "hf"]
    ok = flip and all(row["auto_regime"] == ("hf" if row["kdelta"] > BOUNDARY else "lf")
                      for row in switch_rows)
    print(f"[CHECK] regime flips lf->hf at k*delta = r/e = {BOUNDARY:.2f}: "
          f"{'PASS' if ok else 'FAIL'}")

    import matplotlib.pyplot as plt
    import pandas as pd
    df_sw = pd.DataFrame(switch_rows)
    df_sw.to_csv("results/switching_level.csv", index=False)
    kd = df_sw["kdelta"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axvspan(kd.min(), BOUNDARY, color="tab:blue", alpha=0.08)
    ax.axvspan(BOUNDARY, kd.max(), color="tab:red", alpha=0.08)
    ax.loglog(kd, df_sw["err_lf"], "o--", label="forced LF (Bessel)")
    ax.loglog(kd, df_sw["err_hf"], "^--", label="forced HF (plane waves)")
    ax.loglog(kd, df_sw["err_auto"], "k-", lw=2, label="auto (wideband switch)")
    ax.axvline(BOUNDARY, color="k", ls=":", lw=1.5)
    ax.text(BOUNDARY * 1.05, 3e-1, r"$k\delta = r/e$", rotation=90, va="top", fontsize=9)
    ax.set_xlabel(r"$k\delta$"); ax.set_ylabel("relative error of $UBV^T q$")
    ax.set_title(f"Switching level at r = {R_SW}: each regime fails on the wrong side")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("figures/switching_level.png", dpi=140, bbox_inches="tight")
    plt.show()
    print("Saved: figures/switching_level.png, results/switching_level.csv")
else:
    print("[SKIP] switching-level experiment")
""")

# =============================================================================
# S6 - MNN-H2 experiment
# =============================================================================

CELL_33_MNN_MD = md(r"""## 6. MNN-H2: the $\mathcal{H}^2$-matrix neural network, trained without MATLAB

[`mnn-H2`](https://github.com/ywfan/mnn-H2) (Fan, Feliu-Fabà, Lin, Ying,
Zepeda-Núñez [FFLYZ19]) hard-wires the FMM/$\mathcal{H}^2$ structure into a neural
network: restriction/prolongation convolutions play $V^{\mathsf T}$/$U$, banded
convolutions per level play $B$, and a local band plays the near field.

**The MATLAB scripts in that repo are only a data generator.** `testH2matrix.py`
reads two HDF5 files — `data/Input_<prefix>.h5` (dataset `Input`, shape
`(Nsamples, Nx)`) and `data/Output_<prefix>.h5` (dataset `Output`) — and everything
else is Keras. So we generate the same files **in Python** from our own kernel and
train **their** architecture (the translation-invariant `testH2matrix.py` network,
adapted verbatim: `CNNR`/`CNNK`/`CNNI` + periodic padding):

- Geometry: $N = 320 = 2^6 \cdot 5$ sources on the unit circle, targets on a
  concentric circle of radius $1.3$ — translation-invariant in the angle, exactly the
  symmetry the Conv1D architecture assumes, and singularity-free.
- Map to learn: $q \mapsto \phi = Kq$ with $K_{ij} = H_0^{(1)}(k\,|t_i - s_j|)$, $k=10$.
- $\phi$ is complex and the network is real, so two networks are trained (Re / Im).
- Deviations from their script (documented): their input normalization divides by the
  data mean (their NLSE inputs are positive); ours are zero-mean Gaussian, so we
  standardize by the standard deviation instead. Epochs cut 4000 → 400 (CPU budget;
  their published $\sim10^{-3}$–$10^{-4}$ errors use the full 4000),
  optimizer `Nadam` with `learning_rate` instead of the legacy `lr` attribute.

The point of the exercise: a **learned** compressed operator reaches percent-level
relative error in minutes of CPU training (and $\sim10^{-3}$ with the full epoch
budget) — useful where the kernel is unknown or data-driven — while the **analytic**
FMMs below sit at $10^{-12}$–$10^{-15}$ for the same operator, at comparable apply
cost. Compression is real, though: the two networks store $\sim$58 KB of weights vs
1.6 MB for the dense matrix.
""")

CELL_34_MNN_DATA = code(r"""# Generate the MNN-H2 training data (the role MATLAB plays in the original repo).
import h5py
from scipy.special import hankel1

NX_NN     = 320                 # = 2^6 * 5  -> k_grid = 7, m = 5 in their notation
K_NN      = 10.0
NSAMP     = 1200
N_TRAIN   = 1000

theta = 2 * np.pi * np.arange(NX_NN) / NX_NN
SRC_C = np.exp(1j * theta)                  # sources: unit circle
TGT_C = 1.3 * np.exp(1j * theta)            # targets: concentric circle (no singularity)
K_C = hankel1(0, K_NN * np.abs(TGT_C[:, None] - SRC_C[None, :]))   # 320 x 320, exact

rng = np.random.default_rng(SEED)
Q_ALL = rng.standard_normal((NSAMP, NX_NN))                        # real charges
PHI_ALL = Q_ALL @ K_C.T                                            # (NSAMP, NX) complex

with h5py.File("data/Input_helm2d.h5", "w") as f:
    f.create_dataset("Input", data=Q_ALL)
with h5py.File("data/Output_helm2d_re.h5", "w") as f:
    f.create_dataset("Output", data=PHI_ALL.real)
with h5py.File("data/Output_helm2d_im.h5", "w") as f:
    f.create_dataset("Output", data=PHI_ALL.imag)

with h5py.File("data/Input_helm2d.h5", "r") as f:                  # read-back check
    assert f["Input"].shape == (NSAMP, NX_NN)
print(f"Wrote data/Input_helm2d.h5 + Output_helm2d_{{re,im}}.h5  "
      f"({NSAMP} samples of length {NX_NN}) - no MATLAB involved.")
""")

CELL_35_MNN_TRAIN = code(r"""# Train the (adapted) MNN-H2 network of Fan et al. - architecture code follows
# mnn-H2/NLSE/testH2matrix.py (arXiv:1808.02376); see Section 6 text for the
# documented deviations.
nn_results = None
if AVAILABLE.get("tf_keras"):
    import tf_keras as keras
    from tf_keras import backend as Kb
    from tf_keras.layers import Add, Conv1D, Flatten, Input, Lambda, Reshape
    from tf_keras.models import Model
    from tf_keras.optimizers import Nadam
    import tensorflow as tf
    from time import perf_counter

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    ALPHA, K_GRID, N_CNN, EPOCHS, BATCH = 6, 7, 5, 400, 64
    L = K_GRID - 1                       # 6
    M_NN = NX_NN // 2**L                 # 5
    n_b_2, n_b_l, n_b_ad = 2, 3, 1

    def padding(x, size):                # periodic padding [testH2matrix.py]
        return Kb.concatenate([x[:, x.shape[1] - size // 2:x.shape[1], :],
                               x, x[:, 0:(size - size // 2 - 1), :]], axis=1)

    def CNNR(X, Nin, Nout, al_out, act):
        return Conv1D(al_out, Nin // Nout, strides=Nin // Nout, activation=act)(X)

    def CNNK(X, al_out, w, act):
        return Conv1D(al_out, w, activation=act)(Lambda(lambda x: padding(x, w))(X))

    def CNNI(X, al_out, act):
        return Conv1D(al_out, 1, activation=act)(X)

    def build_mnnh2():
        Ipt = Input(shape=(NX_NN, 1))
        # adjacent (near-field) part
        uad = Reshape((NX_NN // M_NN, M_NN))(Ipt)
        for _ in range(N_CNN - 1):
            uad = CNNK(uad, M_NN, 2 * n_b_ad + 1, "relu")
        uad = CNNK(uad, M_NN, 2 * n_b_ad + 1, "linear")
        uad = Flatten()(uad)
        # far-field part: V (restrictions), M (banded per level), U (prolongations)
        Vv_list = []
        Vv = CNNR(Ipt, NX_NN, 2**L, ALPHA, "linear")
        Vv_list.insert(0, Vv)
        for ll in range(L - 1, 1, -1):
            Vv = CNNR(Vv, 2**(ll + 1), 2**ll, ALPHA, "linear")
            Vv_list.insert(0, Vv)
        MVv_list = []
        for ll in range(2, L + 1):
            MVv = Vv_list[ll - 2]
            w = 2 * n_b_2 + 1 if ll == 2 else 2 * n_b_l + 1
            for _ in range(N_CNN):
                MVv = CNNK(MVv, ALPHA, w, "relu")
            MVv_list.append(MVv)
        chi = None
        for ll in range(2, L):
            chi = MVv_list[ll - 2] if chi is None else Add()([chi, MVv_list[ll - 2]])
            chi = CNNI(chi, 2 * ALPHA, "linear")
            chi = Reshape((2**(ll + 1), ALPHA))(chi)
        chi = Add()([chi, MVv_list[L - 2]])
        chi = CNNI(chi, M_NN, "linear")
        chi = Flatten()(chi)
        model = Model(inputs=Ipt, outputs=Add()([chi, uad]))
        model.compile(loss="mean_squared_error", optimizer=Nadam(learning_rate=1e-3))
        return model

    import h5py
    with h5py.File("data/Input_helm2d.h5", "r") as f:
        Q_data = f["Input"][:]
    s_in = Q_data.std()
    Xd = (Q_data / s_in).reshape(NSAMP, NX_NN, 1)

    class EveryN(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 50 == 0:
                print(f"    epoch {epoch + 1:>4}/{EPOCHS}  loss = {logs['loss']:.3e}")

    nets, scales, train_times = {}, {}, {}
    for part in ["re", "im"]:
        with h5py.File(f"data/Output_helm2d_{part}.h5", "r") as f:
            Yd = f["Output"][:]
        s_out = Yd.std(); scales[part] = (s_in, s_out)
        model = build_mnnh2()
        if part == "re":
            print(f"MNN-H2 parameters per network: {model.count_params():,}")
        print(f"Training {part.upper()} network ({EPOCHS} epochs, batch {BATCH}) ...")
        t0 = perf_counter()
        model.fit(Xd[:N_TRAIN], (Yd / s_out)[:N_TRAIN], batch_size=BATCH,
                  epochs=EPOCHS, verbose=0, callbacks=[EveryN()])
        train_times[part] = perf_counter() - t0
        nets[part] = model

    # test evaluation on the 200 held-out samples
    Xtest = Xd[N_TRAIN:]
    pred_re = nets["re"].predict(Xtest, verbose=0) * scales["re"][1]
    pred_im = nets["im"].predict(Xtest, verbose=0) * scales["im"][1]
    phi_hat = pred_re + 1j * pred_im
    phi_true = PHI_ALL[N_TRAIN:]
    errs = np.linalg.norm(phi_hat - phi_true, axis=1) / np.linalg.norm(phi_true, axis=1)

    t0 = perf_counter()
    nets["re"].predict(Xtest, verbose=0); nets["im"].predict(Xtest, verbose=0)
    t_apply = (perf_counter() - t0) / len(Xtest)

    nn_results = {"mean_rel_err": float(errs.mean()), "max_rel_err": float(errs.max()),
                  "params_total": int(nets["re"].count_params() * 2),
                  "train_time_s": sum(train_times.values()),
                  "apply_time_s": float(t_apply)}
    # NOTE: TF CPU training is not bit-deterministic across runs even with fixed
    # seeds (thread scheduling); at this 400-epoch budget the test error varies
    # roughly between 2e-2 and 6e-2. The original paper reaches ~1e-3-1e-4 with
    # 4000 epochs.
    ok = nn_results["mean_rel_err"] < 1e-1
    print()
    print(f"[CHECK] mnnh2 test rel_err = {nn_results['mean_rel_err']:.2e} "
          f"(expect < 1e-1 at this CPU epoch budget; observed range ~2e-2-6e-2 "
          f"across runs): {'PASS' if ok else 'FAIL'}")
    print(f"train time {nn_results['train_time_s']:.0f} s, "
          f"apply {t_apply*1e3:.2f} ms/matvec, "
          f"params (both nets) {nn_results['params_total']:,}")
else:
    print("[SKIP] MNN-H2 training (tf_keras not available)")
""")

CELL_36_MNN_COMPARE = code(r"""# Learned vs analytic operators on the SAME 320-point circle geometry.
import pandas as pd
from time import perf_counter

rows_nn = []
q_c = Q_ALL[N_TRAIN].astype(complex)
phi_exact = K_C @ q_c

t0 = perf_counter(); _ = K_C @ q_c; t_dense = perf_counter() - t0
rows_nn.append({"operator": "dense matvec", "rel_err": 0.0,
                "apply_s": t_dense, "memory_bytes": K_C.nbytes})

if AVAILABLE.get("stablefmmpy") and CONVENTION_OK.get("stablefmmpy"):
    try:      # multilevel mode: faster and more accurate where available
        t, phi = time_call(eval_stablefmmpy, SRC_C, TGT_C, q_c, K_NN, r=25,
                           multilevel=True, repeats=1)
        label = "stablefmmpy (r=25, multilevel)"
    except TypeError:
        t, phi = time_call(eval_stablefmmpy, SRC_C, TGT_C, q_c, K_NN, r=25, repeats=1)
        label = "stablefmmpy (r=25)"
    rows_nn.append({"operator": label, "rel_err": rel_err(phi, phi_exact),
                    "apply_s": t, "memory_bytes": None})
if AVAILABLE.get("pyfmmlib") and CONVENTION_OK.get("pyfmmlib"):
    t, phi = time_call(eval_pyfmmlib, SRC_C, TGT_C, q_c, K_NN, iprec=3)
    rows_nn.append({"operator": "pyfmmlib (iprec=3)", "rel_err": rel_err(phi, phi_exact),
                    "apply_s": t, "memory_bytes": None})
if AVAILABLE.get("fmm2dpy") and CONVENTION_OK.get("fmm2dpy"):
    t, phi = time_call(eval_fmm2dpy, SRC_C, TGT_C, q_c, K_NN, eps=1e-9)
    rows_nn.append({"operator": "fmm2dpy (eps=1e-9)", "rel_err": rel_err(phi, phi_exact),
                    "apply_s": t, "memory_bytes": None})
if nn_results:
    rows_nn.append({"operator": "MNN-H2 (learned)", "rel_err": nn_results["mean_rel_err"],
                    "apply_s": nn_results["apply_time_s"],
                    "memory_bytes": nn_results["params_total"] * 4})

df_nn = pd.DataFrame(rows_nn)
df_nn.to_csv("results/mnnh2_comparison.csv", index=False)
print(df_nn.to_string(index=False))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4.4))
for _, row in df_nn.iterrows():
    e = max(row["rel_err"], 1e-16)
    ax.scatter(row["apply_s"], e, s=90)
    ax.annotate(row["operator"], (row["apply_s"], e), textcoords="offset points",
                xytext=(8, 4), fontsize=9)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("apply time per matvec (s)")
ax.set_ylabel("relative error")
ax.set_title(f"Learned vs analytic compressed operators  (N = {NX_NN}, k = {K_NN:g})")
ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig("figures/mnnh2_compare.png", dpi=140, bbox_inches="tight")
plt.show()
print("Saved: figures/mnnh2_compare.png, results/mnnh2_comparison.csv")
""")

# =============================================================================
# S7 - Qualitative comparison + paper tables
# =============================================================================

CELL_37_QUAL_MD = md(r"""## 7. The full library landscape

Two of the six requested libraries cannot run this 2D Helmholtz benchmark numerically,
for reasons verified directly in their source trees (clones under
`../librerias_reps/`):

- **ScalFMM v3.1** (Inria) — kernel-independent interpolation FMM (Chebyshev /
  equispaced+FFT / barycentric). Its kernel collection
  (`include/scalfmm/matrix_kernels/`) contains Laplace (`one_over_r`, `ln_2d`),
  Gaussian and scalar test kernels — **no Helmholtz kernel exists in the release**,
  and the experimental pybind11 module is hard-coded to 2D Laplace with real
  `double` values. Adding $H_0^{(1)}$ would mean writing and validating a new C++
  matrix kernel with complex arithmetic.
- **FastMMLib v1.0** (Darrigrand, Lafranche, Rais) — clean generic C++ design, but
  the shipped kernels are **3D only** (Helmholtz via Gegenbauer series, Laplace via
  spherical harmonics), and there are **no Python bindings** (plain `g++` + `ar`
  build, results written to text files).
- **mnn-H2** — *is* exercised numerically in Section 6, by replacing its MATLAB data
  generator with our Python one.
""")

CELL_38_QUAL_TABLE = code(r"""import pandas as pd

LIB_MATRIX = [
    {"library": "stablefmmpy 0.1.0", "language": "Python", "dim": "2D",
     "kernels": "Helmholtz H0, Cauchy, log",
     "helmholtz_2d": "yes",
     "low_rank_scheme": "balanced Bessel/Hankel (LF) + equispaced DFT (HF) [HK]",
     "python_api": "native", "license": "MIT", "parallelism": "none (pure Python)",
     "stability": "lambda balancing, ||U||max<=1, backward error O(log N)",
     "in_this_benchmark": "numeric (all sections)"},
    {"library": "pyfmmlib 2024.1.1", "language": "Fortran + f2py", "dim": "2D/3D",
     "kernels": "Laplace, Helmholtz, Cauchy, difference",
     "helmholtz_2d": "yes (low-frequency FMM only)",
     "low_rank_scheme": "analytic multipole/local + plane-wave translations [GG15]",
     "python_api": "wrappers (fmm_part)", "license": "MIT (wrappers) / GPL-2 (fmmlib)",
     "parallelism": "OpenMP (_vec routines)",
     "stability": "classical scaling; k in upper half-plane; no wideband HF pass",
     "in_this_benchmark": "numeric (Sections 4-6)"},
    {"library": "fmm2d (Flatiron)", "language": "Fortran", "dim": "2D",
     "kernels": "Laplace, Helmholtz, biharmonic, Stokes",
     "helmholtz_2d": "yes",
     "low_rank_scheme": "multipole/local + diagonal plane-wave forms",
     "python_api": "fmm2dpy (f2py)", "license": "Apache-2.0",
     "parallelism": "OpenMP",
     "stability": "production code, tolerance-driven (eps)",
     "in_this_benchmark": "numeric (Sections 4-6)"},
    {"library": "ScalFMM 3.1", "language": "C++17", "dim": "1-3D (generic)",
     "kernels": "Laplace 1/r, ln_2d, Gaussian - NO Helmholtz",
     "helmholtz_2d": "no",
     "low_rank_scheme": "Chebyshev / equispaced+FFT / barycentric interpolation + ACA/SVD recompression",
     "python_api": "experimental pybind11 (2D Laplace only)", "license": "CeCILL-C",
     "parallelism": "OpenMP tasks, MPI, StarPU",
     "stability": "interpolation-based (well-conditioned)",
     "in_this_benchmark": "qualitative"},
    {"library": "FastMMLib 1.0", "language": "C++11", "dim": "3D",
     "kernels": "Helmholtz 3D (Gegenbauer), Laplace 3D (spherical harmonics)",
     "helmholtz_2d": "no (3D only)",
     "low_rank_scheme": "Gegenbauer-series plane-wave expansions",
     "python_api": "none", "license": "CeCILL v2.1",
     "parallelism": "none documented",
     "stability": "auto-tuned expansion lengths",
     "in_this_benchmark": "qualitative"},
    {"library": "mnn-H2", "language": "Python (Keras) + MATLAB datagen", "dim": "1D/2D grids",
     "kernels": "learned operator (orig.: NLSE, RTE)",
     "helmholtz_2d": "learned here (Section 6)",
     "low_rank_scheme": "H2 structure hard-wired as CNN restriction/band/prolongation [FFLYZ19]",
     "python_api": "training scripts", "license": "MIT (research code)",
     "parallelism": "TensorFlow",
     "stability": "trained; error plateau ~1e-3 - 1e-4",
     "in_this_benchmark": "numeric via Python-generated data (no MATLAB)"},
]
df_lib = pd.DataFrame(LIB_MATRIX)
df_lib.to_csv("results/library_matrix.csv", index=False)
pd.set_option("display.max_colwidth", 60)
df_lib[["library", "dim", "helmholtz_2d", "low_rank_scheme", "in_this_benchmark"]]
""")

CELL_39_PAPER_TABLES = code(r"""# Reproduction check. The library-level INVARIANTS are tested (see the
# calibration note in Section 5.4); the papers' exact values - measured inside a
# tau=0.6, L=8 tree rather than this single-level tau=1/3 geometry - are printed
# as reference.
PAPER = {
    "hk61_stable_B": 0.52867, "hk61_stable_err": 5.5118e-15, "hk61_naive_inf_r": 180,
    "m2d61_plateau": 4.6e-15, "m2d61_naive_inf_r": 70,
    "m2d62_plateau": 1.3e-14, "m2d62_naive_inf_r": 90,
}

if rows_hk61:
    Bs = [row["stable_B"] for row in rows_hk61]
    ok_const = (max(Bs) - min(Bs)) < 1e-6          # the invariant: B independent of r
    ok_range = 0.4 <= Bs[0] <= 0.8                 # same order as the paper's 0.5287
    first_inf = next((row["r"] for row in rows_hk61 if not np.isfinite(row["naive_B"])), None)
    ok_inf = first_inf is not None                 # naive DOES overflow at finite r
    errs_ok = all(np.isfinite(row["stable_err"]) and row["stable_err"] <= 1e-12
                  for row in rows_hk61)
    print(f"[HK] T6.1  stable_B = {Bs[0]:.5f}, constant in r: {ok_const}"
          f"  (paper: {PAPER['hk61_stable_B']} in the L=8 tree)")
    print(f"[HK] T6.1  naive B overflows from r = {first_inf}"
          f"  (paper: {PAPER['hk61_naive_inf_r']}; earlier here because tau=1/3"
          f" widens the naive factors)")
    print(f"[HK] T6.1  balanced err <= 1e-12 for all r: {errs_ok}"
          f"  (paper: {PAPER['hk61_stable_err']:.1e})")
    ok_hk = ok_const and ok_range and ok_inf and errs_ok
    print(f"[CHECK] HK-T6.1 reproduction (library invariants): {'PASS' if ok_hk else 'FAIL'}")
    print()

    plateau61 = min(row["balanced_err"] for row in rows_m2d61)
    finf61 = next((row["r"] for row in rows_m2d61 if not np.isfinite(row["naive_err"])), None)
    ok61 = plateau61 <= 1e-13 and finf61 is not None
    print(f"[M2D] T6.1 (Cauchy)  balanced plateau {plateau61:.1e} "
          f"(paper {PAPER['m2d61_plateau']:.1e}); naive Inf from r={finf61} "
          f"(paper {PAPER['m2d61_naive_inf_r']})")
    print(f"[CHECK] M2D-T6.1 reproduction: {'PASS' if ok61 else 'FAIL'}")

    plateau62 = min(row["balanced_err"] for row in rows_m2d62)
    finf62 = next((row["r"] for row in rows_m2d62 if not np.isfinite(row["naive_err"])), None)
    ok62 = plateau62 <= 1e-13
    note62 = ("no naive overflow at single level - consistent with the design log: "
              "the paper's r=90 threshold for the log kernel requires the full L=8 tree"
              if finf62 is None else f"naive Inf from r={finf62}")
    print(f"[M2D] T6.2 (log)  balanced plateau {plateau62:.1e} "
          f"(paper {PAPER['m2d62_plateau']:.1e}); {note62}")
    print(f"[CHECK] M2D-T6.2 reproduction: {'PASS' if ok62 else 'FAIL'}")
else:
    print("[SKIP] paper-table reproduction (stablefmmpy not available)")
""")

# =============================================================================
# S8 - Reproducibility + references
# =============================================================================

CELL_40_FOOTER = code(r"""import json
import platform
from importlib import metadata
from time import perf_counter

env = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "seed": SEED,
    "available": AVAILABLE,
    "convention_gate": CONVENTION_OK,
    "elapsed_total_s": round(perf_counter() - T0, 1),
    "versions": {},
}
for dist in ["numpy", "scipy", "matplotlib", "pandas", "h5py", "numba",
             "stablefmmpy", "pyfmmlib", "tensorflow-cpu", "tf-keras"]:
    try:
        env["versions"][dist] = metadata.version(dist)
    except metadata.PackageNotFoundError:
        env["versions"][dist] = None

with open("results/environment.json", "w") as f:
    json.dump(env, f, indent=2)

print(json.dumps(env, indent=2))
print()
print(f"Total notebook wall time: {env['elapsed_total_s']:.0f} s")
""")

CELL_41_REFERENCES = md(r"""## 8. References

**Core papers (the method under study)**

- **[HK]** M. Michelle, X. Ou, J. Xia, *A Stable Matrix Version of the Wideband Fast
  Multipole Method for the 2D Helmholtz Kernel* (preprint). See also
  [Purdue Math profile](https://www.math.purdue.edu/~xiaj/).
- **[M2D]** X. Ou, M. Michelle, J. Xia, *A Stable Matrix Version of the 2D Fast
  Multipole Method*, SIAM J. Matrix Anal. Appl. 46(1), 2025.
  [doi:10.1137/24M1636953](https://doi.org/10.1137/24M1636953)

**Low-rank approximation**

- **[HMT11]** N. Halko, P.-G. Martinsson, J. A. Tropp, *Finding Structure with
  Randomness*, SIAM Review 53(2), 2011. [doi:10.1137/090771806](https://doi.org/10.1137/090771806)
- **[CGMR05]** H. Cheng, Z. Gimbutas, P.-G. Martinsson, V. Rokhlin, *On the
  Compression of Low Rank Matrices*, SIAM J. Sci. Comput. 26(4), 2005.
  [doi:10.1137/030602678](https://doi.org/10.1137/030602678) — the ID used by
  [`scipy.linalg.interpolative`](https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html).
- **[Beb00]** M. Bebendorf, *Approximation of boundary element matrices*,
  Numer. Math. 86, 2000. [doi:10.1007/PL00005410](https://doi.org/10.1007/PL00005410)
- **[CETW22]** Y. Chen, E. N. Epperly, J. A. Tropp, R. J. Webber, *Randomly pivoted
  Cholesky*, 2022. [arXiv:2207.06503](https://arxiv.org/abs/2207.06503)
- **[GW26]** M. A. Gilles, H. Wilber, *Low-Rank Approximation by Randomly Pivoted LU*,
  2026. [arXiv:2601.22344](https://arxiv.org/abs/2601.22344)
- **[FD09]** W. Fong, E. Darve, *The black-box fast multipole method*,
  J. Comput. Phys. 228(23), 2009. [doi:10.1016/j.jcp.2009.08.031](https://doi.org/10.1016/j.jcp.2009.08.031)
- **[CD13]** C. Cecka, E. Darve, *Fourier-Based Fast Multipole Method for the
  Helmholtz Equation*, SIAM J. Sci. Comput. 35(1), 2013.
  [doi:10.1137/11085774X](https://doi.org/10.1137/11085774X)

**Surveys mined for Section 2**

- **[Kai25]** S. Kailasa, *Modern Research Software for Fast Multipole Methods*,
  PhD thesis, University College London, 2025 (§2.3 low-rank schemes, §2.5.1 Helmholtz
  rank growth, Table 2.5 software matrix). [UCL Discovery](https://discovery.ucl.ac.uk/)
- **[Bla17]** P. Blanchard, *Fast hierarchical algorithms for the low-rank
  approximation of matrices*, PhD thesis, Université de Bordeaux, 2017.
  [HAL tel-01534930](https://theses.hal.science/tel-01534930)

**Libraries**

- `stablefmmpy` — [PyPI](https://pypi.org/project/stablefmmpy/) ·
  [GitHub](https://github.com/pinpacho/stablefmmpy)
- `pyfmmlib` — [GitHub](https://github.com/inducer/pyfmmlib), wrapping
  [`fmmlib2d`](https://github.com/zgimbutas/fmmlib2d). Reference paper **[GG15]**:
  Z. Gimbutas, L. Greengard, *Computational software: Simple FMM libraries for
  electrostatics, slow viscous flow, and frequency-domain wave propagation*,
  Comm. Comput. Phys. 18(2), 2015. [doi:10.4208/cicp.150215.260615sw](https://doi.org/10.4208/cicp.150215.260615sw)
- `fmm2d` / `fmm2dpy` — [GitHub](https://github.com/flatironinstitute/fmm2d) ·
  [docs](https://fmm2d.readthedocs.io/)
- ScalFMM — [Inria GitLab](https://gitlab.inria.fr/solverstack/ScalFMM)
  (`include/scalfmm/utils/low_rank.hpp` for the pACA + SVD pipeline)
- FastMMLib — [PLM GitLab](https://plmlab.math.cnrs.fr/fastmmlib/fastmmlib)
  (É. Darrigrand, Y. Lafranche, R. Rais, [HAL hal-05679634](https://hal.science/hal-05679634))
- mnn-H2 — [GitHub](https://github.com/ywfan/mnn-H2). Reference paper **[FFLYZ19]**:
  Y. Fan, J. Feliu-Fabà, L. Lin, L. Ying, L. Zepeda-Núñez, *A multiscale neural
  network based on hierarchical nested bases*, Res. Math. Sci. 6, 2019.
  [arXiv:1808.02376](https://arxiv.org/abs/1808.02376)

---
*Generated by `gen_fmm2d_helmholtz_bench.py`; part of the `pruebas_claude` FMM
research workspace. Environment built by `setup.sh` (WSL/Linux, full) or
`setup.ps1` (Windows, degraded).*
""")


if __name__ == "__main__":
    cells = [
        CELL_01_TITLE,
        CELL_02_SETUP_MD,
        CELL_03_AVAILABILITY,
        CELL_04_PARAMS,
        CELL_05_RESEARCH_MD,
        CELL_06_READ_NOTES,
        CELL_07_RANK_GEOM,
        CELL_08_RANK_SVD,
        CELL_09_ALGEBRAIC_MD,
        CELL_10_ALGEBRAIC_IMPL,
        CELL_11_ALGEBRAIC_COMPARE,
        CELL_12_RPCHOL,
        CELL_13_ANALYTIC,
        CELL_14_DEFS_MD,
        CELL_15_LAMBDA_PLOT,
        CELL_16_REGIME_MAP,
        CELL_17_HARNESS_MD,
        CELL_18_ADAPTERS,
        CELL_19_NUMBA_ASIDE,
        CELL_20_CONVENTION_CHECK,
        CELL_21_BENCH_STABLE,
        CELL_22_BENCH_PYFMMLIB,
        CELL_23_BENCH_FMM2DPY,
        CELL_24_BENCH_TABLE,
        CELL_25_PLOT_TIME,
        CELL_26_ACC_VS_K,
        CELL_27_PARETO,
        CELL_28_STAB_MD,
        CELL_29_STAB_RUN,
        CELL_30_STAB_PLOTS,
        CELL_31_M2L_MEMORY,
        CELL_32_SWITCHING,
        CELL_33_MNN_MD,
        CELL_34_MNN_DATA,
        CELL_35_MNN_TRAIN,
        CELL_36_MNN_COMPARE,
        CELL_37_QUAL_MD,
        CELL_38_QUAL_TABLE,
        CELL_39_PAPER_TABLES,
        CELL_40_FOOTER,
        CELL_41_REFERENCES,
    ]
    write_notebook(cells, OUT)
