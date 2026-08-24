# Scalable Kernel Learning

**Effective dimension as a computational budget: statistical and computational
tradeoffs in approximate kernel learning.**

## Problem

Kernel methods make linear algorithms nonlinear by replacing inner products with a
kernel function. As a result, the resulting model is defined by an $n \times n$
matrix. Kernel ridge regression requires solving the following expression

$$(K + \lambda n I)\,\alpha = y,$$

which is $O(n^2)$ in memory and $O(n^3)$ in time. At $n = 50{,}000$ that is
20 GB in double precision before doing any factorization. Exact kernel methods
therefore are no longer viable for sample sizes where they would be most interesting.

The standard workaround is approximation. The **Nyström method** builds a
low-rank substitute from $m \ll n$ sampled columns. **Random Fourier features**
avoid $K$ entirely by building an explicit $D$-dimensional feature map
whose inner products approximate the kernel. Both work, but the more interesting
and common question is which one of these is more accurate at a fixed budget - an
empirical comparison between both methods.

We ask a different question: **how much approximation is enough, and can that be
known before running anything?**

## Hypothesis

The theory offers a sensible answer to this question. Define the effective dimension

$$d_{\text{eff}}(\lambda) = \mathrm{tr}\left(K(K + \lambda n I)^{-1}\right) = \sum_{i=1}^{n} \frac{\sigma_i}{\sigma_i + \lambda n}.$$

This counts the directions in the data whose signal survives regularization: components
with $\sigma_i \gg \lambda n$ contribute close to 1 to the sum, components with
$\sigma_i \ll \lambda n$ contribute close to 0. It is the statistical degrees of
freedom of the kernel ridge estimator.

Additionally, it is the number of Nyström landmarks that provably suffices
for the approximate estimator to have the same risk rate as the exact one (Rudi,
Camoriano & Rosasco, 2015). The same quantity dominates both the statistics and the
computation, which leads us in a direction worth exploring and testing.

If the identification is empirically tight, the consequence is more practical: the
computational budget is a function of the kernel spectrum, and *readable in advance*.
We hypothesize that sampling below this scale produces measurable statistical
degradation, while sampling substantially above it provides diminishing returns
relative to computational cost.

A second question also follows from this, and the literature has much less to say
about it. $d_{\text{eff}}$ is derived for the risk of the *mean* predictor. A Gaussian
process also produces a posterior *variance*. Does the same budget buy reasonable
uncertainty, or does the variance require more?

## Methodology

Every approximation is benchmarked against an fp64 baseline computed in this
repository (exact KRR via Cholesky, and exact GP regression with marginal likelihood
hyperparameter optimization). The comparison set is:

- Nyström with uniform, ridge-leverage-score, and recursive leverage-score sampling
- Random Fourier features
- Two sparse GP methods (SoR/DTC and the Titsias variational bound).

The primary experiments are **synthetic, with prescribed kernel spectra**
(polynomial and exponential decay), because that is the only setting where
$d_{\text{eff}}$ is known in closed form, not estimated. We use this case so that we
can properly test the hypothesis before moving to non-synthetic datasets. Standard UCI
regression benchmarks provide the secondary, realistic check.

We use the following metrics for evaluation, taken together on every run:

- **Statistical** — test RMSE, negative log predictive density, empirical
  coverage of 95 % predictive intervals, and KL divergence from the exact
  posterior where it can be computed.
- **Computational** — wall-clock time (fit and predict separated), peak memory,
  analytic FLOP counts.
- **Diagnostic** — $d_{\text{eff}}$, spectral decay exponent, condition number.

## Implementation

- **Python** — all estimators, diagnostics, experiment orchestration, and
  evaluation. Every method reported is implemented here; scikit-learn and
  GPyTorch appear only in the test suite, as correctness oracles.
- **C++ (optional)** — profiled numerical bottlenecks exposed through pybind11,
  primarily kernel matrix construction, which is $O(n^2 d)$ and
  memory-bandwidth-bound.
- **CUDA (optional)** — tiled kernel construction on the GPU, benchmarked against
  a cuBLAS formulation.

For reproducibility, `src/` falls back to the NumPy path when the native extension is
unavailable, so the repository runs and reproduces every statistical result on a
machine with no compiler and no GPU.

Numerical implementations for the synthetic spectrum generator and the randomized
range finder are reused from [rnla](https://github.com/daviddavilad/randomized-numerical-linear-algebra).

## Reproducing

```bash
uv sync
pytest                                     # correctness suite
python experiments/run.py configs/synthetic/poly2.yaml
python experiments/figures.py              # results/ -> paper/figures/
```

## Tech stack

### Python (for research)

```
python 3.12
numpy, scipy          exact linear algebra, fp64 throughout
pytorch               GPU tensors, autograd for hyperparameter opt
gpytorch              ORACLE ONLY - never in the reported pipeline
scikit-learn          ORACLE ONLY
matplotlib            figures
pandas / pyarrow      results storage (parquet)
pytest                tests
hydra-core or yaml    experiment configs
uv                    environment + lockfile
```

### C++ (for numerical computations)

```
pybind11 + Eigen (or raw BLAS)
CMake + Ninja, vcpkg 
GoogleTest or hand-rolled checks
```

Targets, in priority order - **choose by profiling**:
1. **Kernel matrix construction** ($O(n^2 d)$, exhibits parallelism, memory-bandwidth-bound, and it is not optimized much in sklearn).
2. **RFF feature map generation.** Exhibits parallelism.
3. **Recursive RLS sampling.** Algorithmic work, no good reference implementation.

### CUDA (for the GPU)

Custom kernel-construction kernel, benchmarked against a cuBLAS-based formulation. Runs on Nvidia RTX5060 Ti GPU, or CARC, depending on needs.

### Compute

- **Nvidia RTX5060 Ti GPU**
- **CARC** (carc.unm.edu). Free to students. Easley: L40s + H100s. Hopper: A100s.
- MacBook M2 chip for development. PyTorch MPS works for fp32 prototyping; **no CUDA.**

### Writing

LaTeX, following research standards. Results → figures via a single `make figures` path from stored parquet, so nothing in the paper is hand-copied.

---

## Repository structure

```
scalable-kernel-learning/
├── README.md                      # Problem, hypothesis, method, repo instructions, Randomized Numerical Linear Algebra library use
├── pyproject.toml
├── uv.lock
├── CMakeLists.txt
├── configs/
│   ├── synthetic/                 # one YAML per spectrum × n × λ sweep
│   └── real/
├── src/
│   └── skl/
│       ├── __init__.py
│       ├── kernels/               # exact KRR, exact GP
│       │   ├── exact.py
│       │   └── gp.py
│       ├── approx/
│       │   ├── nystrom.py         # uniform + RLS + recursive RLS
│       │   ├── rff.py
│       │   └── sparse_gp.py       # SoR/DTC + Titsias VFE
│       ├── diagnostics/
│       │   ├── effective_dim.py
│       │   └── spectrum.py
│       ├── data/
│       │   ├── synthetic.py       # prescribed-spectrum generator
│       │   └── uci.py
│       ├── eval/
│       │   ├── metrics.py         # RMSE, NLPD, coverage, KL
│       │   └── timing.py
│       └── cpp/
│           ├── kernel_matrix.cpp  # pybind11 module
│           ├── kernel_matrix.cu   # CUDA path
│           └── bindings.cpp
├── experiments/
│   ├── run.py                     # config → parquet
│   └── figures.py                 # parquet → PDF figures
├── tests/
│   ├── test_exact.py              # vs. sklearn/GPyTorch to 1e-10
│   ├── test_approx.py             # m = n recovers exact
│   ├── test_effective_dim.py      # vs. closed form on known spectra
│   └── test_cpp.py                # C++/CUDA paths agree with NumPy to tol
├── results/                       # parquet outputs, gitignored; index.csv tracked
├── docs/                          # general design and project notes
├── paper/
│   ├── main.tex
│   └── figures/
└── .github/workflows/ci.yml       # pytest on CPU
```

**Test suite:**
- Every approximation reduces to the exact method at $m = n$ or $D \to \infty$.
- C++ and CUDA paths agree with the NumPy reference to fp64 tolerance.
- $d_{\text{eff}}$ matches its closed form on synthetic spectra.

---

## Status and roadmap

Phase 0 (foundation), started 15 Aug 2026. Full plan, hypotheses, and roadmap in
[`project-map.md`](project-map.md).

| Phase | Content | Target Deliverable | Date |
|---|---|---|---|
| **0. Foundation** | Repo scaffold, environment, CI, RNLA pinned as a dependency. Base readings. | Repo builds clean, CI green, `make figures` path exists end-to-end on a dummy result. | End of August |
| **1. Exact baselines** | Exact KRR (Cholesky) and exact GP regression, fp64. $d_{\text{eff}}$ estimator. Synthetic spectrum generator. | Matches sklearn/GPyTorch to 1e-10. $d_{\text{eff}}$ matches closed form on known spectra. | Mid-September |
| **2. Approximations** | Nyström (uniform, RLS, recursive RLS) and RFF. Synthetic sweeps. | **H1 and H2 tested.** Figures 1, 2 and 4 exist in draft form. | End of September |
| **3. Uncertainty** | Sparse GP (SoR/DTC and Titsias VFE). Calibration harness. | **H3 tested.** Figure 3 exists. Interim write-up (~4 pages) drafted. | Mid-October |
| **4. C++ layer** | Kernel matrix construction via pybind11, blocked and profiled. RFF map. | Benchmarks reproducible; C++ agrees with NumPy to fp64 tolerance. | Mid-November |
| **5. CUDA layer** | Tiled and streamed kernel construction on the RTX 5060 Ti, benchmarked against a cuBLAS formulation. | Figure 6 exists. GPU agrees with CPU to tolerance. | Mid-December |
| **6. Precision and scale** | H4 accuracy study locally; fp64 timing study on CARC. Real datasets at full sweep. | **H4 tested.** Figure 5 exists. Results frozen — no new experiments after this gate. | Winter break |
| **7. Write-up** | Full paper, LaTeX. | Draft to Prof. Martinez-Ramon; review. | Spring 2027 |

**Note. The Python research must be complete and correct before starting the C++ and CUDA implementations.**

## References

- **Rahimi & Recht (2007)**, *Random Features for Large-Scale Kernel Machines*, NIPS.
- **Williams & Seeger (2001)**, *Using the Nyström Method to Speed Up Kernel Machines*, NIPS.
- **Rudi, Camoriano & Rosasco (2015)**, *Less is More: Nyström Computational Regularization*, NIPS.
- **Alaoui & Mahoney (2015)**, *Fast Randomized Kernel Ridge Regression with Statistical Guarantees*, NIPS.
- **Musco & Musco (2017)**, *Recursive Sampling for the Nyström Method*, NIPS.
- **Yang, Li, Mahdavi, Jin & Zhou (2012)**, *Nyström Method vs Random Fourier Features*, NIPS.
- **Titsias (2009)**, *Variational Learning of Inducing Variables in Sparse GPs*, AISTATS.
- **Quiñonero-Candela & Rasmussen (2005)**, *A Unifying View of Sparse Approximate GP Regression*, JMLR.
- **Bach (2013)**, *Sharp Analysis of Low-Rank Kernel Matrix Approximations*, COLT.
- **Caponnetto & De Vito (2007)**, *Optimal Rates for Regularized Least-Squares*, FoCM.
- **Rasmussen & Williams**, *GPML*, chs. 2, 3, 5, 8.
- **Shawe-Taylor & Cristianini**, *Kernel Methods for Pattern Analysis*, chs. 2–3.
- **Rojo-Álvarez, Martínez-Ramón, Muñoz-Marí & Camps-Valls (2018)**, *Digital Signal Processing with Kernel Methods*, Wiley.
- **Halko, Martinsson & Tropp (2011)**, *Finding Structure with Randomness*.
- **Gardner et al. (2018)**, *GPyTorch: Blackbox Matrix-Matrix GP Inference*.