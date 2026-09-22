# Learning to forget: recurrent sparse spectrum signature GPs

**Tóth, C., Adachi, M., Osborne, M. A., & Oberhauser, H. (2025).** Learning to forget: Bayesian time series forecasting using recurrent sparse spectrum signature Gaussian processes. In *Proceedings of The 28th International Conference on Artificial Intelligence and Statistics*, PMLR 258:4654–4662. [PMLR](https://proceedings.mlr.press/v258/toth25b.html) · [arXiv preprint](https://arxiv.org/abs/2412.19727)

Read: 2026-09-19 (pass 2)

## Problem

The signature kernel method has found many applications in Machine Learning, such as covariance functions for Gaussian Processes. 

The main advantage of using this approach is that it provides a structured global description of the time series. However, it can quickly become a disadvantage when faced with a nonstationary problem, since this characteristic will change the description of the time series over time, thus resulting in poor model performance.

The tradeoff between accuracy and scalability is real. Computing the exact signature kernel requires O(N^2), while simultaneously existing approximating methods are able to reduce the complexity to O(N), although they come with the tradeoff of poor performance for large datasets and the non-adaptability problem. 

## Mapping of the literature

The authors identify two main approaches for reducing the quadratic computational cost of signature kernels.

1. **Subsampling / low-rank approximations.** Previous work has used sequence subsampling through Nyström approximation, inter-domain inducing points, and diagonalization of the Gram matrix. These approaches improve scalability but can produce crude approximations and deteriorate on large datasets.

2. **Random projections.** Random Fourier Features (RFFs) provide an explicit finite-dimensional approximation of stationary kernels. Tóth et al. previously extended this idea to Random Fourier Signature Features (RFSFs), which approximate signature kernels with probabilistic guarantees.

The important limitation is that these methods still treat the signature as a **global feature representation**. Information is accumulated over the entire history of the sequence, so the representation has no principled mechanism for deciding that older information is no longer relevant.

The GP literature provides another route to sequential modeling through specialized sequence kernels and GP state-space models. Sparse-spectrum GPs are particularly relevant here because Random Fourier Features allow the GP to be formulated directly in a finite-dimensional weight space rather than through the full kernel matrix.

The contribution of this paper therefore combines:

$$
\text{signature methods}
+
\text{Random Fourier Features}
+
\text{adaptive forgetting}
+
\text{variational Bayesian inference}.
$$

## Approach

The authors first construct Random Fourier Signature Features (RFSFs), which provide a finite-dimensional random approximation to the signature kernel.

For a stationary base kernel, Random Fourier Features approximate

$$
k(x,y)
\approx
\frac{2}{D}
\sum_{i=1}^{D}
\cos(\omega_i^\top x+b_i)
\cos(\omega_i^\top y+b_i),
$$

where the frequencies $\omega_i$ are sampled from the kernel's spectral measure and

$$
b_i \sim U(0,2\pi).
$$

The RFF construction is then combined with signature features. For signature level $m$, the resulting Random Fourier Signature Feature map is denoted

$$
\Phi_m(x).
$$

Using the first $M$ signature levels gives the complete feature representation

$$
\Phi(x)
=
\left(
1,
\frac{\Phi_1(x)}{\|\Phi_1(x)\|_2},
\ldots,
\frac{\Phi_M(x)}{\|\Phi_M(x)\|_2}
\right)
\in \mathbb{R}^{MD}.
$$

Here, importantly,

- $M$ is the **signature truncation level**;
- $D$ is the **number of Random Fourier Features per signature level**.

Both determine the dimensionality of the feature representation.

### Learning to forget

Standard signature features continuously accumulate information from the sequence. They therefore provide a global description of the complete history but have no built-in mechanism for forgetting observations that have become irrelevant.

The authors introduce **Random Fourier Decayed Signature Features (RFDSFs)** by adding channel-wise exponential decay parameters

$$
\lambda \in \mathbb{R}^D.
$$

Rather than applying a forgetting factor directly to the likelihood contribution of old observations, as in adaptive GP approaches such as AGP, the decay is applied inside the recursive **signature feature representation itself**.

The recursion takes the general form

$$
\Phi_m(x_{0:l})
=
\lambda^{\odot m}\odot\Phi_m(x_{0:l-1})
+
\text{contribution from the new increment},
$$

where $\odot$ denotes element-wise operations.

Unrolling the recursion shows that historical feature contributions are multiplied by exponentially decreasing powers of $\lambda$. Therefore, information from the past gradually disappears from the feature representation.

This distinction is important:

$$
\boxed{
\text{AGP: decay the influence of old observations}
}
$$

whereas

$$
\boxed{
\text{RFDSF: decay historical information inside the feature representation}.
}
$$

The decay factors are **channel-specific**, meaning different Random Fourier feature channels can operate on different effective time scales. Some channels can preserve long-term information while others focus primarily on recent observations.

Consequently, the phrase **"dynamically adapt its context length"** refers to learning how quickly information is forgotten, not changing the dimensionality of the model.

In other words,

$$
\text{adaptive context length}
\neq
\text{adaptive feature budget}.
$$

### Recurrent Sparse Spectrum Signature GP

The RFDSF representation is used as the feature map of a GP formulated in weight space. Predictions take the form

$$
y_l
=
w^\top\Phi(x_{0:l})
+
\epsilon_l,
$$

where

$$
w\in\mathbb{R}^{MD}
$$

and

$$
\epsilon_l\sim\mathcal{N}(0,\sigma_y^2).
$$

A Gaussian prior is placed over the weights,

$$
p(w)=\mathcal{N}(0,I),
$$

making this a Bayesian linear model over the RFDSF feature representation and, equivalently, a finite-dimensional GP approximation.

The model also treats the Random Fourier frequencies $\Omega$ and phases $B$ probabilistically. Variational distributions are introduced over

$$
w,\qquad\Omega,\qquad B,
$$

and the model is trained using variational inference.

For example,

$$
q(w)
=
\mathcal{N}(\mu_w,\Sigma_w).
$$

The predictive distribution therefore remains probabilistic rather than producing only point forecasts.

The resulting model is called the **Recurrent Sparse Spectrum Signature Gaussian Process (RS$^3$GP)**. Its variational version additionally learns distributions over the random covariance parameters.

Crucially, the model learns the **decay parameters** controlling the effective memory of the feature channels. Thus the context length is learned from data rather than being specified as a fixed historical window.

### What is actually adaptive?

This distinction is particularly important for comparison with adaptive sparse GPs.

The paper adapts:

$$
\boxed{\text{effective temporal memory / context length}}
$$

through learned decay parameters.

It also learns parameters associated with the RFF representation through variational inference.

However, it does **not** dynamically adapt the number of Random Fourier Features.

The RFF dimension

$$
D
$$

is fixed in advance, as is the signature truncation level

$$
M.
$$

For example, in the synthetic experiment the authors explicitly use

$$
D=200,
\qquad
M=5.
$$

Therefore, the dimensionality of the feature representation

$$
MD
$$

remains fixed during training.

This means that "dynamically adapting context length" should not be interpreted as dynamically adapting model capacity or computational budget.

## Results

The authors evaluate the method on a synthetic multi-sinusoidal dataset specifically constructed to require reasoning across both short and long time horizons.

They compare:

- VRS$^3$GP;
- RS$^3$GP;
- an SVGP with an RBF kernel and a fixed, hand-tuned context length.

The synthetic experiment shows that the learned decay mechanism allows the model to capture both short- and long-horizon temporal structure without manually specifying a fixed context window.

The method is then evaluated on eight real-world time-series datasets using the Continuous Ranked Probability Score (CRPS). The baselines include classical statistical models, deep-learning forecasting models, diffusion models, SVGP, and deep-kernel GPs.

RS$^3$GP and VRS$^3$GP outperform the GP baselines on several datasets and achieve performance comparable to the deep-learning baselines while generally requiring substantially less training time.

The computational advantage becomes particularly relevant for long sequences because the RFDSF representation can be recursively evaluated and parallelized rather than requiring the quadratic computation of the exact signature kernel.

## Questions and gaps

- The paper learns the **effective context length**, but why should the number of Random Fourier Features $D$ remain fixed?

- Since $D$ controls the accuracy of the Monte Carlo approximation to the kernel, could the required $D$ change as the data distribution changes?

- Is there a principled way to determine the required RFF dimension from quantities such as effective dimension, spectral decay, or approximation error rather than selecting $D$ as a fixed hyperparameter?

- The decay is channel-specific:

  $$
  \lambda\in\mathbb{R}^D.
  $$

  Does learning different decay rates effectively make some random features much more useful than others? If so, could low-value feature channels be removed and replaced dynamically?

- Conversely, if the current feature representation becomes insufficient under distribution shift, could new Fourier features be introduced online?

- How sensitive is performance to the initially chosen $D$? Could a model with an unnecessarily large $D$ waste substantial computation even after the learned decay mechanism determines that many channels contribute little?

- The authors learn temporal relevance through $\lambda$, but temporal relevance and representational complexity are different problems. Can these two forms of adaptation be learned jointly?

## Open problems / extensions of the work

The authors explicitly identify several limitations. Their current formulation assumes a Gaussian likelihood, which may be inappropriate for heavy-tailed or non-Gaussian data. They also suggest that more sophisticated forgetting mechanisms may improve the ability to model complex dependencies in non-stationary time series, and mention extension to multivariate forecasting as future work.

A separate potential extension is **adaptive model capacity**.

The proposed method adapts

$$
\lambda
\quad\Rightarrow\quad
\text{how much history the representation remembers},
$$

but keeps

$$
D
\quad\Rightarrow\quad
\text{number of Random Fourier Features}
$$

fixed.

Therefore there are potentially two distinct adaptation problems:

$$
\boxed{
\text{When should information be forgotten?}
}
$$

and

$$
\boxed{
\text{How large should the feature representation be?}
}
$$

The paper directly addresses the first but not the second.

An interesting extension would therefore be an online mechanism that can increase, remove, or replace Random Fourier Features according to the current complexity of the kernel representation.

## Relevance to my work

This paper strengthens rather than eliminates the question I identified when reading the adaptive sparse GP literature.

In AGP, the model adapts the locations of the inducing points but keeps the number of inducing points fixed:

$$
M_{\mathrm{inducing}}=\text{constant}.
$$

In RS$^3$GP, the model learns how quickly historical information should decay, but the number of Random Fourier Features is also fixed:

$$
D_{\mathrm{RFF}}=\text{constant}.
$$

These are different approximation mechanisms,

$$
\text{inducing points}
\qquad\text{vs.}\qquad
\text{random Fourier features},
$$

but both choose the **approximation budget in advance**.

This suggests a broader research question:

$$
\boxed{
\text{Can scalable kernel methods adapt their approximation budget online?}
}
$$

Rather than choosing a fixed $M$ or $D$, the model could potentially use information about the current kernel operator -- such as spectral decay, effective dimension, ridge leverage scores, or approximation error -- to determine how much computational budget is actually necessary.

For non-stationary problems this is particularly interesting because the complexity of the effective kernel representation may itself change over time.

The desired model would therefore adapt along two separate axes:

$$
\boxed{
\text{temporal adaptation}
+
\text{representational adaptation}.
}
$$

The first determines **which historical information remains relevant**.

The second determines **how much model capacity is required to represent the current problem**.

RS$^3$GP provides a principled mechanism for the first. The second appears to remain open in this formulation.