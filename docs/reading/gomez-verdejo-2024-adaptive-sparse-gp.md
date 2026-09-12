# Adaptive sparse Gaussian process

**Gómez-Verdejo, V., Parrado-Hernández, E., & Martínez-Ramón, M. (2024).** Adaptive sparse Gaussian process. *IEEE Transactions on Neural Networks and Learning Systems*, 35(11), 16383–16395. [doi](https://doi.org/10.1109/TNNLS.2023.3294089) · [arXiv preprint](https://arxiv.org/abs/2302.10325)

Read: 2026-09-09 (pass 3)

## Problem

Common machine learning and data analysis tasks share a common problem: non-stationarity of the data or distribution. Adaptive learning is thus necessary for cases in which the learning machine needs to forget past data distribution.

However, these updates to the model sometimes require a large amount of compute, thus making computational cost a burden for coming up with the proper machine learning model. As a result, efficient algorithms require a rather simple, low-effort model update so as not to grow in computational burden with the incoming data, and with the lowest possible computational cost for online parameter updating. Hence their contribution of the adaptive sparse GP model follows.

In the paper, this approach is applied to signal processing applications in general, given their non-stationary nature and the need to update as new data comes. Some of these SP applications include:

- Spectral estimation
- Signal modeling
- Adaptive filtering
- Array processing

In this sense, Bayesian models and in particular Gaussian processes (GP) seem to be an ideal solution, since each time a new sample arrives the predictive posterior can be updated using the previous posterior as the new prior of the model and multiplying it by the likelihood of the new data. Recall the relationship

$$p(\theta \mid y) \propto p(y \mid \theta)\,p(\theta).$$

The idea can also be extended to models with hidden latent variables, which require learning algorithms such as expectation maximization. This is known in the literature as online variational Bayes. However, the main problem of these approaches is that the predictive posterior parameters usually depend on all the training data, thus making them computationally expensive. Not only that, but as new samples arrive their complexity tends to grow with $O(N^3)$, where $N$ is the number of samples. This is exactly the biggest burden for these models, if not combined with pruning methods.

This is one of the biggest problems in adaptive learning. As models grow in complexity and training samples, their complexity does not grow linearly, it does so **cubically**. Developing a model with a rather parsimonious update that is computationally cheap is thus the main focus that the literature seems to be getting at.

## Mapping of the literature

To solve these issues, a possible solution is to use low computational cost GP versions, or more specifically, to resort to sparse GPs (SGP). In these approaches, the model solution only depends on $M \ll N$ points in the observation space, called **inducing points**. This makes the model much more cost-friendly and avoids the complexity of the model growing as new samples come.

The authors cite the following models as attempts trying to balance a solution that does not grow in complexity with one that is computationally efficient:

- **L. Csató and M. Opper, "Sparse on-line Gaussian processes".** They proposed a compact Bayesian online algorithm based on an approximation to the real predictive posterior. Cost $O(NM^2)$. Drawback: the model hyperparameters (kernel parameters and noise variance) remain fixed, which is unrealistic in nonstationary problems.
- **T. Evans and P. Nair, "Scalable Gaussian processes with grid-structured eigenfunctions (GP-GRIEF)".** They approximate the kernel matrix using $K$ eigenfunctions to obtain a fast computation of the likelihood derivatives.
- **S. Stanton, W. Maddox, I. Delbridge, and A. G. Wilson, "Kernel interpolation for scalable online Gaussian processes".** They interpolate the kernel with order $K$ to reduce the computational cost of updating the predictive distribution. Cost $O(MK^2)$. Limitation: need to predefine (and keep fixed) the set of inducing points, and the obtained posterior is an *approximation* (because it is based on SGPs). Personal note: is this related to RNLA? In particular, the approximation sounds like an rSVD computed on the kernel.

The fix to these drawbacks is variational SGP algorithms, where including a variational distribution over the inducing points avoids the error of approximating the posterior. For example:

- **T. N. Hoang, Q. M. Hoang, and B. K. H. Low, "A unifying framework of anytime sparse Gaussian process regression models with stochastic variational inference for big data".**
- **C.-A. Cheng and B. Boots, "Incremental variational sparse Gaussian process regression".**

Both approaches propose an incremental learning algorithm for variational SGPs, although the hyperparameters are fixed in training.

- **T. D. Bui, C. Nguyen, and R. E. Turner, "Streaming sparse Gaussian process approximations".** They propose to update the variational bound with an online variational Bayesian scheme. The advantage is having a variational bound expression to be optimized with respect to all the model hyperparameters, including the inducing points.

We can also find approaches in the variational SGP space:

- **J. Hensman, N. Fusi, and N. D. Lawrence, "Gaussian processes for big data".**

Bui et al. and Hensman et al. use variational stochastic inference (VSI) to obtain online versions, because using VSI allows one to work with small batches of data, thus becoming closer to the online update idea. Drawback: the stochastic optimization assumes the sampling is random (which is not necessarily true in nonstationary problems) and the optimization usually requires that each minibatch is processed in several iterations (computationally expensive).

The authors then claim that these solutions are not focused on nonstationary datasets, which makes them unsuitable for signal processing applications.

### Prior approaches at reducing the computational cost

- **M. W. Seeger, C. K. Williams, and N. D. Lawrence, "Fast forward selection to speed up sparse Gaussian process regression".** They propose replacing $K_{xx}$ with a Nyström approximation $K_{xu}K_{uu}^{-1}K_{ux}$, with $K_{xu}$ and $K_{uu}$ being the kernel matrix of the inducing points with the training data and with themselves. Personal note: Nyström (part of my RNLA research).
- **E. Snelson and Z. Ghahramani, "Sparse Gaussian processes using pseudo-inputs".** An evolution of the previous method. Corrects the approximated matrix with the term $\mathrm{diag}(K_{xx}) - \mathrm{diag}(K_{xu}K_{uu}^{-1}K_{ux})$ so that the diagonal of $K_{xx}$ is exact.

The main drawback of these two: they do not tend to the exact GP, since they start from an approximation. Also, the inducing inputs are additional parameters to be inferred, so overfitting might become a problem.

Variational sparse GPs (VSGPs) overcome these issues by minimizing the divergence between the exact GP posterior and a variational approximation where the inducing points are modeled as variational parameters. This way, by minimizing the divergence with respect to these inducing points, the SGP tends to the exact GP. Introducing a variational prior over the inducing variables reduces the overfitting risk.

## Approach

The adaptive sparse GP model specification follows these steps:

1. First, reformulate a variational sparse Gaussian process (VSGP) algorithm to make it adaptive through a **forgetting factor**.
2. Next, to make model inference as simple as possible, update a **single inducing point** of the sparse GP model together with the remaining model parameters every time a new sample arrives.

This essentially makes the algorithm achieve fast convergence of the inference process, which allows it to keep an efficient model update (a single inference iteration) even with highly non-stationary data.

### Variational sparse GP background

The standard GP regression model assumes

$$y_n = f_n + \epsilon_n, \qquad \epsilon_n \sim \mathcal{N}(0,\sigma^2),$$

with a GP prior over the latent function $f$. For $N$ training observations, exact GP inference requires operations on the kernel matrix

$$K_{xx} \in \mathbb{R}^{N\times N},$$

which leads to the well-known $O(N^3)$ computational cost.

To obtain a sparse representation, introduce a set of $M \ll N$ **inducing inputs**

$$U = [u_1,\ldots,u_M]$$

and their corresponding latent function values

$$f_u = [f(u_1),\ldots,f(u_M)]^\top.$$

The main idea is to make $f_u$ summarize the information contained in the full latent vector $f$. Assuming that $f_u$ sufficiently represents $f$,

$$p(f_* \mid f_u,f) \approx p(f_* \mid f_u).$$

The true posterior over the inducing variables is then approximated by a Gaussian variational distribution

$$q(f_u) = \mathcal{N}(f_u \mid \mu,A),$$

where

$$\mu = \mathbb{E}_q[f_u], \qquad A = \mathrm{Cov}_q(f_u).$$

This gives the approximate predictive posterior

$$q(f_*) = \int p(f_* \mid f_u)q(f_u)\,df_u = \mathcal{N}(f_* \mid m_*,v_*),$$

with

$$m_* = k_{u*}^\top K_{uu}^{-1}\mu$$

and

$$v_* = k_{**} - k_{u*}^\top K_{uu}^{-1}k_{u*} + k_{u*}^\top K_{uu}^{-1}AK_{uu}^{-1}k_{u*}.$$

Thus, once $U$, $\mu$, and $A$ have been learned, prediction depends on $M$ inducing points rather than all $N$ observations.

The inducing locations $U$ are learned by maximizing the **variational lower bound (ELBO)**. Define the Nyström-type approximation

$$Q_{xx} = K_{xu}K_{uu}^{-1}K_{ux}.$$

The collapsed variational bound can then be written as

$$\mathcal{F}_V(U) = \log \mathcal{N}\left(y \mid 0, \sigma^2I + Q_{xx}\right) - \frac{1}{2\sigma^2}\mathrm{tr}\left(K_{xx}-Q_{xx}\right).$$

The first term measures how well the sparse GP explains the observations, while the trace term penalizes covariance information lost by replacing $K_{xx}$ with its sparse approximation $Q_{xx}$. The bound is optimized with respect to the inducing locations $U$, kernel hyperparameters $\theta$, and noise variance $\sigma^2$.

For fixed $U$, $\theta$, and $\sigma^2$, the optimal variational distribution has a closed form. Define

$$B = \left(K_{uu} + \sigma^{-2}K_{ux}K_{xu}\right)^{-1}.$$

Then

$$\mu = \sigma^{-2}K_{uu}BK_{ux}y, \qquad A = K_{uu}BK_{uu}.$$

Hence, the three main quantities have the following interpretation:

- $U$ determines **where the function is summarized**.
- $\mu$ determines **what the posterior believes the function values are at those locations**.
- $A$ determines **the uncertainty and covariance associated with those inducing values**.

### The forgetting factor

The first contribution of the paper is to make the VSGP adaptive to non-stationary data by introducing an exponential **forgetting factor** $0 < \lambda \leq 1$. At time $t$, an observation obtained at time $t'$ receives weight

$$\lambda^{t-t'}.$$

Thus, recent observations have greater influence than older observations,

$$1,\lambda,\lambda^2,\lambda^3,\ldots$$

as we move backward through the data. When $\lambda = 1$, no forgetting occurs and the formulation reduces to the ordinary VSGP.

Define the diagonal forgetting matrix

$$\Lambda_t = \mathrm{diag}\left(\lambda^{t-1}, \lambda^{t-2}, \ldots, \lambda, 1\right).$$

The standard VSGP quantities are then replaced by exponentially weighted versions. In particular,

$$B_\lambda = \left(K_{uu} + \sigma^{-2}K_{ux}\Lambda_t K_{xu}\right)^{-1},$$

and

$$\mu_\lambda = \sigma^{-2}K_{uu}B_\lambda K_{ux}\Lambda_t y, \qquad A_\lambda = K_{uu}B_\lambda K_{uu}.$$

The resulting adaptive predictive posterior remains Gaussian,

$$q_\lambda(f_*) = \mathcal{N}\left(f_* \mid m_{\lambda,*},v_{\lambda,*}\right),$$

where

$$m_{\lambda,*} = \sigma^{-2}k_{u*}^{\top}B_\lambda K_{ux}\Lambda_t y$$

and

$$v_{\lambda,*} = k_{**} + k_{u*}^{\top}\left(B_\lambda-K_{uu}^{-1}\right)k_{u*}.$$

A useful interpretation of the forgetting factor is that old observations are assigned increasing effective noise. An observation $t-t'$ steps old behaves as if its noise variance were approximately

$$\frac{\sigma^2}{\lambda^{t-t'}}.$$

Thus, rather than abruptly deleting old observations, the model gradually becomes less confident that old information still represents the current data-generating process.

Most importantly for online learning, the historical data can be compressed into fixed-size quantities. Define

$$S_t = K_{ux}\Lambda_t K_{xu}, \qquad r_t = K_{ux}\Lambda_t y.$$

When a new observation $(x_{t+1},y_{t+1})$ arrives, define its kernel vector against the inducing points as

$$k_{u,t+1} = [k(u_1,x_{t+1}), \ldots, k(u_M,x_{t+1})]^\top.$$

The sufficient quantities can then be recursively updated as

$$S_{t+1} = \lambda S_t + k_{u,t+1}k_{u,t+1}^{\top}, \qquad r_{t+1} = \lambda r_t + k_{u,t+1}y_{t+1}.$$

Since $S_t \in \mathbb{R}^{M\times M}$ and $r_t \in \mathbb{R}^{M}$, their dimensions do not increase as new samples arrive. This provides the paper's computationally cheap **fast-AGP** update when the inducing points and model hyperparameters are held fixed.

### Single inducing point update

The recursive update above is computationally cheap, but keeping $U$, $\theta$, and $\sigma^2$ fixed can be inadequate under strong non-stationarity. As the underlying distribution changes, inducing locations that were previously representative may become obsolete, and the optimal kernel and noise parameters may also change.

The full **AGP** therefore allows model inference to continue online. Rather than reoptimizing all $M$ inducing locations after every new observation, the authors update only **one inducing point** together with the kernel hyperparameters and noise variance.

The representativeness of an input $x_*$ by the current inducing set is related to the conditional variance

$$\delta(x_*) = k_{**} - k_{u*}^{\top}K_{uu}^{-1}k_{u*}.$$

Since this quantity is a conditional variance, $\delta(x_*) \geq 0$. A small value indicates that the feature representation of $x_*$ is already well captured by the inducing set, whereas a large value indicates information that is poorly represented by the current inducing subspace.

To maintain a fixed model size $M$, the algorithm evaluates the relevance $R_m$ of the existing inducing points and identifies the least relevant one,

$$m^* = \arg\min_m R_m.$$

That inducing point can then be replaced by a new candidate associated with the incoming data. Conceptually,

$$U_{\mathrm{new}} = \left(U_{\mathrm{old}} \setminus \{u_{m^*}\}\right) \cup \{x_{t+1}\}.$$

The new location provides an initialization for the subsequent inference step. The selected inducing point is then optimized together with the kernel parameters $\theta$ and noise variance $\sigma^2$, while the remaining $M-1$ inducing locations are held fixed.

Updating only one inducing point substantially reduces the dimensionality of the optimization problem and allows the authors to use only a small number of inference iterations, typically a single iteration.

Changing an inducing location creates an important complication. If $u_m^{\mathrm{old}} \rightarrow u_m^{\mathrm{new}}$, then all kernel evaluations $k(u_m,x_i)$ associated with that inducing point change. Consequently, the previous fixed-size summaries $S_t$ and $r_t$ cannot in general be updated exactly without revisiting data.

The authors therefore retain an optional sliding window containing the $T$ most recent observations and use it during parameter inference. This produces an update cost of approximately $O(TM^2)$, where $T$ can be chosen much smaller than the complete number of observations $N$.

The two proposed operating modes can therefore be summarized as:

**Option 1: fast-AGP.** $U$, $\theta$, and $\sigma^2$ are fixed, while the predictive posterior is updated recursively as new observations arrive. This is the computationally cheapest option, but it cannot fully adapt the model representation to large changes in the underlying distribution.

**Option 2: AGP.** The predictive posterior is updated recursively, while the model additionally updates $u_{m^*}$, $\theta$, and $\sigma^2$.

Thus, the full AGP combines

$$\boxed{\text{forget old information} + \text{learn new information} + \text{replace obsolete inducing points} + \text{adapt the kernel}}$$

while attempting to keep the computational cost of each online update low.

## Results

Experimental results demonstrate the capabilities of the proposed algorithm and its performance in modeling the predictive posterior, in both mean and confidence interval estimation, compared to state-of-the-art approaches.

The authors compare their two methods (adaptive VSGP without inference - fast-AGP, and adaptive VSGP with inference - AGP) against the following approaches:

- An adaptive VSGP that uses VSI (AGP-VSI) to update the model parameters in each iteration on a window of $T$ (same $T$).
- Windowed VSGP (w-VSGP) that uses a sliding window of $T$.
- Online SGP. Taken from PyTorch rather than implemented by them, using the parameters that the original authors recommend.
- Kernel interpolation for scalable online GP (WISKI), using the authors' implementation.

They exclude any approaches related to local approximations, and sparse methodologies whose strategy is based on an online reduction of the kernel matrix size through similarity criteria, since the first option is not a sparse methodology and the second does not involve optimizing a set of inducing points to be statistically significant. Thus, the scopes are not really comparable.

### Synthetic dataset

Experiments were first run on a synthetic dataset, a sinusoidal signal

$$y(t) = A_t \sin(2\pi f t) + \epsilon_t.$$

Table I reports MSE, 95% CI, and training time, showing that AGP is the clear winner in accuracy, and it is also faster than the rest of the methods (excluding fast-AGP, of course).

Figure 2 plots the evolution of the MSE during online learning. The results are quite surprising: AGP is the only one that is truly stable in terms of MSE - every other algorithm's MSE radically worsens after $t = 3$ seconds, including fast-AGP, as is expected since it stops learning the $U$ matrix.

Figure 3 plots the mean real signal, predictive mean, and two-sigma uncertainty for each of the methods. Again, AGP tracks the signal excellently throughout the plotted runtime, while the rest of the methods tend to break down after 3 seconds. The uncertainty of w-VSGP widens dramatically.

Finally, Figure 4 shows the evolution of the AGP MSE for different values of $\lambda$. The most stable values are 0.994, 0.933, 0.912, 0.983, 0.912, although the distribution of the data suggests that these measurements might be noisy. In particular, $\lambda = 1$ has lower MSE than $\lambda = 0.891$. The authors stress the importance of choosing the forgetting factor $\lambda$, and this table also shows the robustness of the algorithm with respect to that parameter choice.

### Real dataset

Results using a real dataset follow the same pattern in terms of performance. Table II reports the same performance measures (MSE, 95% CI, and training time) for the different methodologies.

From Table II, fast-AGP and AGP show the best performance overall. fast-AGP attains the lowest MSE, but AGP is the only method that estimates the 95% CI accurately - the authors attribute fast-AGP's apparent advantage in the mean estimate to slight overfitting. AGP's coverage lands around 96%, perhaps slightly conservative, while the remaining methods sit near 90%.

Table III reports the performance measures for a different problem. Again fast-AGP gives the best MSE with the fastest implementation, while AGP gives the best confidence interval estimate. On the OR dataset, WISKI achieves the lowest MSE but fails to estimate the 95% CI accurately.

## Questions and gaps

Concepts I have not covered yet:

- Hidden latent variables
- Online variational Bayes

Questions:

- Don't you have to take the absolute value of $k_{**}-k_{u*}^{\top}K_{uu}^{-1}k_{u*}$ to determine similarity? **Answered.**
- What is $G(f_u, y)$ in the ELBO formulation? What does it denote?
- When defining the relevance criterion,

$$R_m = \sum_{t'=1}^{t}\lambda^{t-t'}k_{mm}^{-1}k_{mt'}^2,$$

what does $k_{mm}^{-1}k_{mt'}^2$ denote?

## Open problems / extensions of the work

The authors state that past contributions only partially cover the needs of:

- A model update that does not grow in computational cost with the incoming data, and
- The lowest possible computational cost for updating the parameters of the model.

However, the introduction of a field of solutions to this problem has been motivated with the publication of this paper.

One issue with the paper's contribution is that under option 2 (model update with inference over the model parameters) we change an inducing point,

$$u_m^{\mathrm{old}} \rightarrow u_m^{\mathrm{new}},$$

and therefore all of its kernel evaluations $k(u_m,x_i)$ change. So you can't perfectly update everything using only the previous $M\times M$ summary anymore. This is why the authors optionally maintain a sliding window of $T$ recent observations: when they optimize an inducing point or hyperparameters, they can use those recent observations to recompute the relevant quantities. That gives a full parameter-update step complexity of approximately

$$\boxed{O(TM^2).}$$

The authors emphasize that $T$ can be much smaller than the total number of historical observations $N$. This is somewhat of a rustic approach, since the new inference will never be perfect, especially if we want to keep the cost low with a small window $T$. Hence there must be a better way of solving this problem somehow. Good question to raise.

Option 1 (model update without inference over the model parameters) is cheap, but it might not fully reflect changes in the data, since we keep $U$, $\theta$, and $\sigma^2$ fixed and only update the posterior recursively.

## Relevance to my work

S. Stanton, W. Maddox, I. Delbridge, and A. G. Wilson, "Kernel interpolation for scalable online Gaussian processes", might be related in some way to RNLA. You can apply NLA methods to kernels to make matrix operations computationally efficient and ensure stability of solutions. This sounds like an interesting direction to explore, and the paper deserves a read to understand the method. If I can find a way of making the set of inducing points dynamical (to improve stability and accuracy in nonstationary datasets), that could be a promising direction.

Overall, the literature that is focused on executing approximation on kernels *before* solving the problem tends to be inaccurate, or does not approach the exact GP. There must be some way of reducing the cost of the problem without sacrificing robustness.

$U$ is the really interesting object from a scalability perspective. Once $U$ is fixed, Gaussian conjugacy gives us beautiful algebra for $\mu$ and $A$. Finding a small $U$ that preserves the important information in a huge kernel operator is where approximation, sampling, effective dimension, leverage, optimization, and GP inference start colliding. Another opportunity.