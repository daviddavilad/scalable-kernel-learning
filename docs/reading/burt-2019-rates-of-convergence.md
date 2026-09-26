# Rates of convergence for sparse variational GP regression

**Burt, D. R., Rasmussen, C. E., & van der Wilk, M. (2019).** Rates of convergence for sparse variational Gaussian process regression. In *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97:862–871. [PMLR](https://proceedings.mlr.press/v97/burt19a.html) · [arXiv](https://arxiv.org/abs/1903.03571)

Read: 2026-09-25 (pass 3)

## Problem

Exact GP regression costs $O(N^3)$ time and $O(N^2)$ memory. Sparse GPs with $M \ll N$ inducing variables reduce this to $O(NM^2 + M^3)$ time and $O(NM + M^2)$ memory, which looks almost linear in $N$. But that hides the real question:

$$\boxed{\text{How must } M \text{ grow with } N \text{ to maintain a good approximation?}}$$

If $M$ must eventually grow proportionally to $N$, sparse GPs are not truly scalable. The paper derives conditions under which $\mathrm{KL}(Q\|\hat P) \to 0$ as $N \to \infty$ while $M \ll N$, where $Q$ is the variational posterior and $\hat P$ the exact GP posterior.

## Approach

### 1. Sparse variational GP setup

With $K_n = K_{ff} + \sigma_n^2 I$, the exact log marginal likelihood is

$$\mathcal{L} = -\tfrac12 y^\top K_n^{-1} y - \tfrac12 \log|K_n| - \tfrac N2 \log(2\pi).$$

Introducing $M$ inducing inputs $Z = \{z_m\}$ with $u_m = f(z_m)$ gives the Nyström approximation

$$Q_{ff} = K_{fu}K_{uu}^{-1}K_{uf}, \qquad Q_n = Q_{ff} + \sigma_n^2 I.$$

After optimizing the variational distribution (Titsias, 2009), the ELBO is

$$\mathcal{L}_{\mathrm{lower}} = -\tfrac12 y^\top Q_n^{-1} y - \tfrac12 \log|Q_n| - \tfrac N2 \log(2\pi) - \frac{t}{2\sigma_n^2}, \qquad t = \mathrm{Tr}(K_{ff} - Q_{ff}),$$

and the gap is exactly the KL divergence:

$$\boxed{\mathcal{L} = \mathcal{L}_{\mathrm{lower}} + \mathrm{KL}(Q\|\hat P).}$$

### 2. What they bound

Let $\widetilde K_{ff} = K_{ff} - Q_{ff}$ with largest eigenvalue $\widetilde\lambda_{\max}$. The **a posteriori** bound is

$$\mathrm{KL}(Q\|\hat P) \leq \frac{1}{2\sigma_n^2}\left[t + \frac{\widetilde\lambda_{\max}\|y\|^2}{\sigma_n^2 + \widetilde\lambda_{\max}}\right] \leq \frac{t}{2\sigma_n^2}\left(1 + \frac{\|y\|^2}{\sigma_n^2 + t}\right).$$

For outputs drawn from the GP prior, the average-case bound is much cleaner:

$$\boxed{\frac{t}{2\sigma_n^2} \leq \mathbb{E}_y\left[\mathrm{KL}(Q\|\hat P)\right] \leq \frac{t}{\sigma_n^2}.}$$

So up to a factor of two, $\mathbb{E}_y[\mathrm{KL}] \propto t$.

### 3. How the trace term enters

The diagonal of the residual is the conditional variance of $f(x_i)$ left after conditioning on the inducing variables,

$$[K_{ff} - Q_{ff}]_{ii} = k(x_i,x_i) - k_{x_i u}K_{uu}^{-1}k_{u x_i},$$

so $t$ is the total covariance left unexplained by the sparse approximation. It appears explicitly in the ELBO as the penalty $-t/(2\sigma_n^2)$ and controls the KL - it simultaneously governs the objective and the approximation error.

### 4. From spectral features to a priori bounds

If the inducing features are aligned with the leading eigenvectors of $K_{ff} = W\Lambda W^\top$, then $Q_{ff}$ is the optimal rank-$M$ approximation and

$$t_{\mathrm{opt}} = \sum_{m=M+1}^{N} \lambda_m(K_{ff}), \qquad \widetilde\lambda_{\max} = \lambda_{M+1}(K_{ff}).$$

To get a priori results, they move to the covariance operator $(\mathcal{K}g)(x') = \int g(x)k(x,x')p(x)\,dx$ with Mercer expansion $k(x,x') = \sum_m \lambda_m \phi_m(x)\phi_m(x')$, and define eigenfunction inducing variables $u_m = \int f(x)\phi_m(x)p(x)\,dx$. The expected trace error becomes

$$\boxed{\mathbb{E}_X[t] = N\sum_{m=M+1}^{\infty}\lambda_m.}$$

This is the bridge from finite-matrix error to population spectral decay. With $C = N\sum_{m>M}\lambda_m$, the high-probability bounds scale as $\mathrm{KL} \lesssim C/\sigma_n^2$: increasing $N$ makes approximation harder, increasing $M$ shrinks the tail, and the required growth of $M$ is set by eigenvalue decay.

### 5. Back to ordinary inducing points

For inducing points chosen from the data, $Q_{ff}$ is a Nyström approximation. Sampling $Z$ from a k-DPP, $P(Z) \propto \det(K_{ZZ})$, gives (Belabbas & Wolfe, 2009)

$$\boxed{\mathbb{E}_Z\left[\mathrm{Tr}(K_{ff} - Q_{ff})\right] \leq (M+1)\sum_{m=M+1}^{N}\lambda_m(K_{ff}),}$$

so determinant-based selection is within a factor $M+1$ of the ideal spectral error in expectation. The k-DPP favours inducing sets that are well dispersed in kernel space. With an $\varepsilon$-approximate k-DPP the bound gains an additive $2Nv\varepsilon$ term.

### 6. How they choose M in practice

The common rule - increase $M$ until the ELBO stops improving - is only a **necessary** condition, not a sufficient one. A proper a posteriori certificate uses an upper bound on the marginal likelihood:

$$\boxed{\mathrm{KL}(Q\|\hat P) \leq \mathcal{L}_{\mathrm{upper}} - \mathcal{L}_{\mathrm{lower}}.}$$

If the gap is small, the approximation is certified. But the paper's main contribution is **a priori growth laws**, not a practical adaptive rule for $M$.

## Results

**Squared exponential kernel, Gaussian inputs.** Eigenvalues decay geometrically, $\lambda_m \propto B^{m-1}$ with $0 < B < 1$, so $\sum_{m>M}\lambda_m = O(B^M)$. This gives

$$\boxed{M = O(\log N) \text{ in 1D}, \qquad M = O(\log^D N) \text{ in } D \text{ dimensions.}}$$

**Matérn-$(k+\tfrac12)$ kernels.** Eigenvalues decay polynomially, $\lambda_m \asymp m^{-2k-2}$, so $\sum_{m>M}\lambda_m = O(M^{-2k-1})$. With $M = N^\alpha$, convergence requires $\alpha > 1/(2k+1)$ for eigenfunction features but only $\alpha > 1/(2k)$ for ordinary inducing points, because of the extra $M+1$ factor. For Matérn-3/2 that is $\alpha > 1/3$ versus $\alpha > 1/2$. The authors note it is unclear whether this gap comes from the proof, the k-DPP initialization, or is inherent to inducing points.

**Qualitative conclusion.** Smoother kernels and more concentrated inputs mean faster eigenvalue decay and fewer inducing variables. Rough kernels such as Matérn-1/2 need substantially larger $M$.

**Posterior mean and variance also converge.** If $\mathrm{KL} \to 0$, pointwise posterior mean and variance converge to those of the exact GP, so the guarantees extend to prediction and uncertainty, not just the objective.

## Questions and gaps

1. **Scaling law, not an online rule.** The results prescribe global growth rates such as $M = O(\log^D N)$ or $M = N^\alpha$. They do not answer: given the current inducing set $U_t$, should I add another inducing point?

2. **Computability.** The ideal criteria depend on $\sum_{m>M}\lambda_m$, the leading eigensystem of $K_{ff}$, or the population operator. Computing these exactly may defeat the purpose of sparse inference.

3. **Can $t$ be monitored incrementally?** Since $t$ controls KL, a natural criterion is the marginal improvement from adding one point:

$$\Delta t(z \mid U) = t(U) - t(U \cup \{z\}).$$

Can this be computed or approximated cheaply enough online?

4. **Adding versus deleting.** The asymptotic picture assumes $M$ only grows. Under nonstationarity, previously useful inducing points may become redundant, so the problem is closer to dynamic dictionary management than to a growth rate.

5. **Fixed input distribution.** The a priori theory assumes i.i.d. inputs from a fixed $p(x)$. If $p_t(x)$ changes, so do the covariance operator and its spectrum.

## Open problems / extensions of the work

- **Adaptive $M$.** A rule $M_t = M_{t-1} + \Delta M_t$ driven by the current data, posterior, or residual covariance, rather than a predetermined $M(N)$.
- **Marginal-value criterion.** Add $z$ if and only if its statistical benefit exceeds its computational cost, $\Delta_{\mathrm{stat}}(z \mid U) > \eta\,\Delta_{\mathrm{comp}}(M)$. The right definition of each side is open.
- **Incremental residual-trace updates** via rank-one updates of $K_{uu}^{-1}$, Schur complements, or Woodbury identities, to turn the trace criterion into an online mechanism.
- **Other selection schemes.** Ridge leverage scores, pivoted Cholesky, greedy posterior variance, RNLA methods. The paper notes that better Nyström guarantees translate directly into better sparse-GP guarantees.
- **Non-Gaussian likelihoods**, which the authors name as future work.

## Relevance to my work

This paper answers the theoretical side of the question behind my project - how many inducing variables do we actually need - but through **sufficient a priori growth rates** for $M$, not by choosing an optimal $M$. That distinction matters: my project is about deciding $M$ adaptively from the current model state, which this paper does not attempt.

What it does establish are the ingredients that make that question well-posed:

1. sparse GP quality is governed by Nyström approximation quality;
2. Nyström error is controlled by the discarded spectrum;
3. the residual trace $t$ controls the KL divergence to the exact posterior;
4. spectral decay determines the required budget;
5. ordinary inducing points achieve controlled error if selected well.

This converts "do I have enough inducing points?" into a precise question: how much posterior accuracy do I gain by increasing $M$ by one, i.e. how large is $\Delta t(z \mid U)$?

There is also a connection to effective dimension. Burt et al. measure what is left unexplained through the tail $\sum_{m>M}\lambda_m$, while

$$d_{\mathrm{eff}}(\lambda) = \sum_i \frac{\lambda_i}{\lambda_i + \lambda}$$

measures how many directions are statistically active at a given noise scale. Different quantities, same idea: not all eigen-directions of the kernel matter equally. Whether $M$ should track an efficiently estimated $d_{\mathrm{eff}}$, the residual spectral mass, or the trace error is open.

The binding constraint is computational. The research problem is therefore not just finding the optimal $M$, but

$$\boxed{\text{finding a cheap statistic that tells us when increasing } M \text{ is still worth the computational cost.}}$$

That is where this paper leaves a genuine opening for the research direction.

## Future reads

- **Titsias (2009)**, *Variational Learning of Inducing Variables in Sparse Gaussian Processes* - the variational construction behind $Q_{ff}$ and the ELBO.
- **Csató & Opper (2002)**, *Sparse On-Line Gaussian Processes* - existing online sparsification criteria.
- **Alaoui & Mahoney (2015)**, *Fast Randomized Kernel Ridge Regression with Statistical Guarantees* - ridge leverage scores, effective dimension, and Nyström sample complexity.
- **Li, Jegelka & Sra (2016)**, *Fast DPP Sampling for Nyström with Application to Kernel Methods* - the approximate k-DPP initialization Burt et al. build on.
- **Huggins et al. (2019)** - translating KL control into guarantees on posterior mean and variance.