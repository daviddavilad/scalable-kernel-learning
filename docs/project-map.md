# ECE 517 Project Map - Scalable Kernel Learning

**Status:** Tentatively locked, scope and direction to be confirmed with Prof. Martinez-Ramon
**Author:** David Dávila
**Date:** 15 Aug 2026

---

## 1. Title

### Working titles (tentative)

1. *Effective Dimension as a Computational Budget: Statistical and Computational Tradeoffs in Approximate Kernel Learning* — descriptive, accurate, a bit long.
2. *How Much Computation Does a Kernel Learning Problem Actually Require?* - better hook, can be used in conversation.
3. *Beyond the Mean: Approximation Error in the Posterior Variance of Scalable Gaussian Processes* - use this if the H3 result (below) turns out to be the strongest finding.

Start with (1) for now and reevaluate choice later on once the result / findings are known.

### The research question

> Approximation error in kernel learning is mainly dictated by the **effective dimension** of the problem, not by the sample size. If this is true, then the required computational budget is predictable *a priori* from the kernel spectrum - and uniform sampling systematically wastes it.

The governing quantity:

$$d_{\text{eff}}(\lambda) = \operatorname{tr}\!\left(K(K + \lambda n I)^{-1}\right) = \sum_{i=1}^{n} \frac{\sigma_i}{\sigma_i + \lambda n}$$

This expression simultanously represents (a) the statistical degrees of freedom of the KRR fit and (b) the theoretically sufficient number of Nyström landmarks. That coincidence *is* the thesis.

### Project complementarity and integrity

Both projects are designed to complement each other with regards to purpose and skills developed. In this sense:

> **MATH 471:** given this computational problem, how do we scale its solution?
> **ECE 517:** given this learning problem, how much computation is actually necessary?

---

## 2. Hypotheses

| # | Hypothesis | Potential falsification proof |
|---|---|---|
| **H1** | Nyström KRR with $m \gtrsim d_{\text{eff}}(\lambda)$ matches exact KRR test error; below that threshold error degrades sharply. | Error degrades smoothly with no threshold, or the threshold doesn't track $d_{\text{eff}}$. |
| **H2** | Ridge-leverage-score sampling reaches the threshold at smaller $m$ than uniform sampling, and the gap **widens** as the spectrum flattens. | Gap is constant, or reverses. |
| **H3** | Posterior **variance** degrades at a larger $m$ than the posterior **mean** — i.e. $d_{\text{eff}}$ predicts the mean budget but *underestimates* the calibration budget. | Mean and variance degrade together. |
| **H4** | Required numerical precision tracks the conditioning implied by $\lambda$ and the spectral decay: fp32 becomes unusable for small $\lambda$ on fast-decaying spectra. | fp32 is fine throughout, or fails independently of $\lambda$. |

H3 is the strongest claim and a potential path for a research contribution. The literature is heavily focused on the mean, so the variance failure mode is underexamined and maps directly onto sparse GPs and compact models.

---

## 3. Scope

### In scope

**Exact baselines (fp64, the ground truth everything is measured against)**
- Kernel ridge regression via Cholesky of $K + \lambda n I$
- Exact GP regression: marginal likelihood, predictive mean and variance, hyperparameter optimisation

**Approximations**
- Nyström, uniform landmark sampling
- Nyström, ridge-leverage-score sampling (Alaoui–Mahoney)
- Recursive RLS sampling (Musco–Musco) — develop own implementation; no good off-the-shelf version exists
- Random Fourier Features (Rahimi–Recht)
- Orthogonal Random Features (Yu et al. 2016) — *stretch only*
- Sparse GP: Subset of Regressors / DTC (the direct GP analogue of Nyström)
- Sparse GP: Titsias VFE (the principled variational bound) — the interesting comparison for H3, since VFE is known to repair FITC's variance pathologies

**Diagnostics**
- $d_{\text{eff}}(\lambda)$ (exact on synthetics, estimated on real data)
- Spectral decay rate, condition number of $K + \lambda n I$

### Potential extensions

If time-permitting, the following paths are natural extensions of the work:

| Extension | Explanation |
|---|---|
| **Neural tangent kernel** | In the infinite-width limit a neural network trains like kernel regression under the NTK, so the same $d_{\text{eff}}$ method applies to a deep model. Involves deriving the NTK for a simple architecture, computing its spectrum, and checking whether the H1 threshold behaves the same way. Deep learning extension. |
| **Structured random features** | Fastfood and SORF replace the dense random projection in RFF with structured transforms (Hadamard, diagonal, permutation), cutting the feature map from $O(nD)$ to $O(nD\log d)$ and reducing variance through orthogonality. Extends the RFF section of figures 2 and 4. |
| **Randomized SVD landmark selection** | Rather than *sampling* landmarks, choose the subspace directly via a randomized range finder, i.e. sketch $K\Omega$ and orthogonalize. Tests whether sketching beats sampling at a fixed budget, it uses Randomized Numerical Linear Algebra methods. |
| **Non-Gaussian likelihoods** | Extend the GP section to classification via the Laplace approximation or variational inference. The posterior is already approximate before any subsampling is applied, and the two sources of error compound. |
| **Multiple kernel learning under a budget** | Learn a conic combination $k = \sum_i \beta_i k_i$ subject to a fixed total computational budget, and ask whether it is better spent on more landmarks for one kernel or fewer across several. A convex problem in $\beta$. |

### Out of scope

- **Iterative solvers of any kind.** No CG, no Lanczos, no BBMM. This is the collision point with MATH 471 and it is also off-thesis: the idea is to *shrink the problem*, not *solve the same problem cleverly*. Cite Gardner et al. (2018) as related work.
- **Rewriting Cholesky.** Extremely complicated to beat LAPACK, losing to LAPACK is not a result.
- **Deep kernel learning / neural nets.** Different project.
- **Classification.** Regression only, so that calibration is measurable on a continuous scale.

---

## 4. Experimental design

### Datasets

**Synthetic (primary - given that the theory is testable, because $d_{\text{eff}}$ is known exactly)**

Reuse the infrastructure and approaches from the Randomized Numerical Linear Algebra (RNLA) repository. For example, `make_test_matrix`: build $A = U \Sigma V^\top$ with prescribed singular values.

| Spectrum | Form | Purpose |
|---|---|---|
| Polynomial, slow | $\sigma_j \sim j^{-1}$ | large $d_{\text{eff}}$, approximation should struggle |
| Polynomial, medium | $\sigma_j \sim j^{-2}$ | the interesting middle regime |
| Exponential | $\sigma_j \sim e^{-cj}$ | tiny $d_{\text{eff}}$, "less is more" should be dramatic |

Sweep $n \in \{2^{10}, \ldots, 2^{15}\}$, $\lambda$ across 5 decades.

**Real (secondary)**

Use the UCI sets the sparse-GP literature uses: `kin40k` ($n{=}40{,}000$, $d{=}8$), `protein` ($n{\approx}45{,}700$, $d{=}9$), `elevators` ($n{\approx}16{,}600$, $d{=}18$).

**Alternatives:** Solar irradiance or electric load forecasting data (Prof. Martínez-Ramón's research focus).

### Metrics

*Statistical:* test RMSE; negative log predictive density; empirical coverage of 95 % predictive intervals; reliability diagram; KL divergence from the exact posterior where it can be computed.

*Computational:* wall-clock (fit / predict, separated); peak memory; analytic FLOP count; throughput.

*Diagnostic:* $d_{\text{eff}}$, spectral decay exponent, condition number.

### Important research figures (deliverables)

1. Test error vs. $m/d_{\text{eff}}$, one line per spectrum - **the money plot.** If H1 holds, curves collapse onto each other.
2. Error vs. wall-clock time (Pareto frontier), all methods overlaid.
3. Interval coverage vs. $m/d_{\text{eff}}$, overlaid on figure 1 - **the H3 plot**, showing the variance curve lagging the mean curve.
4. Required $m$ for 1 % excess error: RLS vs. uniform, as a function of decay exponent.
5. fp32 vs. fp64 error vs. $\lambda$, with the Cholesky breakdown point marked.
6. GPU/CPU/Python throughput for kernel construction vs. $n$ - systems figure, for performance analysis.

---

## 5. Tech stack

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

**Rule:** every estimator is implemented in the repository. GPyTorch and sklearn can be used as benchmarks/tools to verify correctness in the test suite, to be disclosed in the report and materials.

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

Custom kernel-construction kernel, benchmarked against a cuBLAS-based formulation. Runs on Nvidia RTX5060 GPU, or CARC, depending on needs.

### Compute

- **Nvidia RTX5060 Ti GPU**
- **CARC** (carc.unm.edu). Free to students. Easley: L40s + H100s. Hopper: A100s.
- MacBook M2 chip for development. PyTorch MPS works for fp32 prototyping; **no CUDA.**

### Writing

LaTeX, following research standards. Results → figures via a single `make figures` path from stored parquet, so nothing in the paper is hand-copied.

---

## 6. Repository structure

```
scalable-kernel-learning/
├── README.md                  # thesis in 3 sentences, repo instructions, Randomized Numerical Linear Algebra library use
├── pyproject.toml
├── uv.lock
├── CMakeLists.txt
├── configs/
│   ├── synthetic/             # one YAML per spectrum × n × λ sweep
│   └── real/
├── src/
│   ├── kernels/               # exact KRR, exact GP
│   │   ├── exact.py
│   │   └── gp.py
│   ├── approx/
│   │   ├── nystrom.py         # uniform + RLS + recursive RLS
│   │   ├── rff.py
│   │   └── sparse_gp.py       # SoR/DTC + Titsias VFE
│   ├── diagnostics/
│   │   ├── effective_dim.py
│   │   └── spectrum.py
│   ├── data/
│   │   ├── synthetic.py       # prescribed-spectrum generator
│   │   └── uci.py
│   ├── eval/
│   │   ├── metrics.py         # RMSE, NLPD, coverage, KL
│   │   └── timing.py
│   └── cpp/
│       ├── kernel_matrix.cpp  # pybind11 module
│       ├── kernel_matrix.cu   # CUDA path
│       └── bindings.cpp
├── experiments/
│   ├── run.py                 # config → parquet
│   └── figures.py             # parquet → PDF figures
├── tests/
│   ├── test_exact.py          # vs. sklearn/GPyTorch to 1e-10
│   ├── test_approx.py         # m = n recovers exact
│   ├── test_effective_dim.py  # vs. closed form on known spectra
│   └── test_cpp.py            # C++/CUDA paths agree with NumPy to tol
├── results/                   # parquet outputs, gitignored; index.csv tracked
├── docs/                      # general design and project notes
├── paper/
│   ├── main.tex
│   └── figures/
└── .github/workflows/ci.yml   # pytest on CPU
```

**Test suite:**
- Every approximation reduces to the exact method at $m = n$ or $D \to \infty$.
- C++ and CUDA paths agree with the NumPy reference to fp64 tolerance.
- $d_{\text{eff}}$ matches its closed form on synthetic spectra.

---

## 7. Roadmap

| Phase | Content | Target Deliverable | Date |
|---|---|---|---|
| **0. Foundation** | Repo scaffold, environment, CI, RNLA pinned as a dependency. Base readings. | Repo builds clean, CI green, `make figures` path exists end-to-end on a dummy result. | End of August |
| **1. Exact baselines** | Exact KRR (Cholesky) and exact GP regression, fp64. $d_{\text{eff}}$ estimator. Synthetic spectrum generator. | Matches sklearn/GPyTorch to 1e-10. $d_{\text{eff}}$ matches closed form on known spectra. | Mid-September |
| **2. Approximations** | Nyström (uniform, RLS, recursive RLS) and RFF. Synthetic sweeps. | **H1 and H2 tested.** Figures 1, 2 and 4 exist in draft form. | End of September |
| **3. Uncertainty** | Sparse GP (SoR/DTC and Titsias VFE). Calibration harness. | **H3 tested.** Figure 3 exists. Interim write-up (~4 pages) drafted. | Mid-October |
| **4. C++ layer** | Kernel matrix construction via pybind11, blocked and profiled. RFF map. | Benchmarks reproducible; C++ agrees with NumPy to fp64 tolerance. | Mid-November |
| **5. CUDA layer** | Tiled and streamed kernel construction on the RTX 5060, benchmarked against a cuBLAS formulation. | Figure 6 exists. GPU agrees with CPU to tolerance. | Mid-December |
| **6. Precision and scale** | H4 accuracy study locally; fp64 timing study on CARC. Real datasets at full sweep. | **H4 tested.** Figure 5 exists. Results frozen — no new experiments after this gate. | Winter break |
| **7. Write-up** | Full paper, LaTeX. | Draft to Prof. Martínez-Ramón; revise; publish repo. | Spring 2027 |

**Note. The Python research must be complete and correct before starting the C++ and CUDA implementations.**

### Parallel track: Studying ML and preparing for exams

Machine Learning is a complex topic that needs careful preparation. Throughout the semester, the following are mandatory while simultaneously working on the repository:

- Attend office hours with any questions about course content or the implementation. Office Location: ECE237b, Department of Electrical and Computer Engineering. Time: Wednesdays 1 - 2 pm (confirmed with Prof. Martinez-Ramon, outside of the regular time, contact ahead of time)
- Study the 3 course books (goal - be able to describe each algorithm/idea by the Feynman Technique, prioritize understanding):
    - The Elements of Statistical Learning, T. Hastie et al., Springer, 2009. (Already have physical copy)
    - Kernel Methods for Pattern Analysis, J. Shawe-Taylor, N. Cristianini, Cambridge University Press, 2004. (Available through UNM library)
    - Gaussian Processes for Machine Learning, C. Rasmussen et al., MIT Press, 2006 (Available online, free access)
- Submit homework and assignments on time
- Start review for exams one week before
- For the repository itself, understand the following deeply (good preparation):
    - SVM primal → dual derivation, KKT conditions, the role of $C$
    - Representer theorem
    - Mercer's theorem and positive-definiteness conditions
    - GP marginal likelihood and its gradients w.r.t. hyperparameters
    - VC dimension and structural risk minimisation

---

## 8. Readings and Resources

### Base understanding

- **Rahimi & Recht (2007)**, *Random Features for Large-Scale Kernel Machines*, NIPS.
- **Williams & Seeger (2001)**, *Using the Nyström Method to Speed Up Kernel Machines*, NIPS.
- **Rudi, Camoriano & Rosasco (2015)**, *Less is More: Nyström Computational Regularization*, NIPS. **Important**
- **Alaoui & Mahoney (2015)**, *Fast Randomized Kernel Ridge Regression with Statistical Guarantees*, NIPS.

### Core readings

- **Musco & Musco (2017)**, *Recursive Sampling for the Nyström Method*, NIPS.
- **Yang, Li, Mahdavi, Jin & Zhou (2012)**, *Nyström Method vs Random Fourier Features*, NIPS. **Important**
- **Titsias (2009)**, *Variational Learning of Inducing Variables in Sparse GPs*, AISTATS.
- **Quiñonero-Candela & Rasmussen (2005)**, *A Unifying View of Sparse Approximate GP Regression*, JMLR.
- **Bach (2013)**, *Sharp Analysis of Low-Rank Kernel Matrix Approximations*, COLT.
- **Caponnetto & De Vito (2007)**, *Optimal Rates for Regularized Least-Squares*, FoCM. - Where $d_{\text{eff}}$ comes from statistically.

### Background readings for working on the implementations and Other

- **Rasmussen & Williams**, *GPML*, chs. 2, 3, 5, 8. (Free.)
- **Shawe-Taylor & Cristianini**, *Kernel Methods for Pattern Analysis*, chs. 2–3.
- **Rojo-Álvarez, Martínez-Ramón, Muñoz-Marí & Camps-Valls (2018)**, *Digital Signal Processing with Kernel Methods*, Wiley **Important**
- **Halko, Martinsson & Tropp (2011)**, *Finding Structure with Randomness*. — Connection with Randomized Numerical Linear Algebra.
- **Gardner et al. (2018)**, *GPyTorch: Blackbox Matrix-Matrix GP Inference*. — Cite as related work.

---

## 9. Questions for Prof. Martinez-Ramon

1. **Approval and scope.** Is this the right size? Is there a piece you would cut or extend? Other extensions worth exploring?
2. **Check-in cadence.** Weekly if possible, biweekly is also okay. 20 minutes is a good range.
3. **CARC sponsorship.** Ask for permission to be added to CARC, if possible.

And one offer: **Would you like an experiment run on a dataset from your own research?**

Show curiosity and genuine thoughtfulness.

---

## 10. First task

Draft a one-page proposal with the research question, the method, the proposed implementation, deliverables and extensions on LaTeX, commit to the project repository alongside the ´project-map.md´.

Optional Title: **Scalable Kernel Learning: Effective Dimension as a Computational Budget.**