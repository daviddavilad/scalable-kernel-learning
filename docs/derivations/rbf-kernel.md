# RBF Kernel

For $x,z\in\mathbb{R}^d$, the radial basis function (RBF) kernel is

$$k(x,z) = \exp\left(-\frac{\|x-z\|_2^2}{2\ell^2}\right), \qquad \ell>0.$$

For

$$X\in\mathbb{R}^{n\times d}, \qquad Z\in\mathbb{R}^{m\times d},$$

the kernel matrix $K\in\mathbb{R}^{n\times m}$ has entries

$$K_{ij}=k(x_i,z_j).$$

## Pairwise squared distances

Direct computation gives

$$\|x_i-z_j\|_2^2 = \sum_{r=1}^d (x_{ir}-z_{jr})^2.$$

Broadcasting this expression requires an intermediate array of shape

$$(n,m,d),$$

so instead expand the norm:

$$
\begin{aligned}
\|x-z\|_2^2
&=(x-z)^\top(x-z)\\
&=x^\top x-x^\top z-z^\top x+z^\top z\\
&=x^\top x-2x^\top z+z^\top z\\
&=\|x\|_2^2-2x^\top z+\|z\|_2^2.
\end{aligned}
$$

Therefore,

$$D_{ij} = \|x_i\|_2^2 -2x_i^\top z_j +\|z_j\|_2^2.$$

Define

$$a_i=\|x_i\|_2^2, \qquad b_j=\|z_j\|_2^2.$$

Then the full squared-distance matrix is

$$D = a\mathbf{1}_m^\top -2XZ^\top +\mathbf{1}_n b^\top.$$

Therefore,

$$\boxed{K = \exp\left(-\frac{D}{2\ell^2}\right)}$$

where the exponential is applied elementwise.

This avoids constructing an $(n,m,d)$ array.

The saving is substantial. At $n=m=10^4$ and $d=10$, the broadcast intermediate
occupies

$$10^4\cdot10^4\cdot10\cdot8\ \text{bytes}=8\ \text{GB},$$

while the result itself occupies

$$10^4\cdot10^4\cdot8\ \text{bytes}=800\ \text{MB}.$$

The dominant operation is instead the matrix multiplication
$XZ^\top\in\mathbb{R}^{n\times m}$, which is handled by BLAS.

## Floating-point cancellation

Mathematically,

$$D_{ij}=\|x_i-z_j\|_2^2\geq0.$$

However, the expanded form

$$\|x_i\|_2^2 -2x_i^\top z_j +\|z_j\|_2^2$$

can suffer cancellation when $x_i\approx z_j$.

For example, even when $x_i=z_j$,

$$D_{ii}=0$$

mathematically, while floating-point arithmetic may produce

$$\widehat D_{ii}=-\varepsilon, \qquad \varepsilon>0$$

for very small $\varepsilon$.

Therefore the implementation uses

$$\widehat D_{ij} \leftarrow \max(\widehat D_{ij},0).$$

This restores the mathematical constraint

$$D_{ij}\geq0.$$

Taking an absolute value would instead map

$$-\varepsilon\mapsto\varepsilon,$$

introducing a positive distance where the intended value is zero.

The expanded form is checked against a direct broadcast implementation in
`test_rbf_kernel_matches_naive`.

## Basic properties

### Unit diagonal

For the self-kernel $K(X,X)$,

$$\|x_i-x_i\|_2^2=0,$$

so

$$\boxed{K_{ii}=1}.$$

Asserted in `test_rbf_kernel_unit_diagonal`.

### Symmetry

Since

$$\|x_i-x_j\|_2^2 = \|x_j-x_i\|_2^2,$$

we have

$$\boxed{K=K^\top}.$$

Asserted in `test_rbf_kernel_symmetry`.

### Bounds

Since $D_{ij}\geq0$, we have $-\frac{D_{ij}}{2\ell^2}\leq0$, and therefore

$$0<K_{ij}\leq1$$

mathematically. Accounting for underflow, the computed matrix satisfies

$$\boxed{0\leq\widehat K_{ij}\leq1}.$$

Asserted in `test_rbf_kernel_is_bounded`.

In fp64 the exponential underflows to zero once its argument falls below
approximately $-745$, that is when

$$\|x-z\|_2^2>1490\,\ell^2.$$

Beyond that point the true kernel value has decayed below the smallest
representable double, so the loss is numerically negligible. Asserted in
`test_rbf_kernel_underflows_to_zero_at_distance`.

### Positive semidefiniteness

Positive semidefiniteness follows from Bochner's theorem: $k$ is shift-invariant,
and the Fourier transform of a Gaussian is a positive Gaussian. Hence

$$\boxed{c^\top Kc\geq0\quad\text{for all }c\in\mathbb{R}^n,}$$

so $K\succeq0$ and $\lambda_i(K)\geq0$ up to floating-point error. See
Rasmussen & Williams, *GPML*, §4.2.

Asserted in `test_rbf_kernel_is_positive_semidefinite`.

### Lengthscale

For fixed $x\neq z$, let

$$r^2=\|x-z\|_2^2>0.$$

Then

$$k_\ell(x,z) = \exp\left(-\frac{r^2}{2\ell^2}\right).$$

Differentiating with respect to $\ell$,

$$\frac{\partial k_\ell(x,z)}{\partial \ell} = \exp\left(-\frac{r^2}{2\ell^2}\right)\frac{r^2}{\ell^3}.$$

Since

$$\exp\left(-\frac{r^2}{2\ell^2}\right)>0, \qquad r^2>0, \qquad \ell^3>0,$$

we obtain

$$\boxed{\frac{\partial k_\ell(x,z)}{\partial\ell}>0} \qquad (x\neq z).$$

Thus increasing the lengthscale increases the similarity between distinct
points. Asserted in `test_rbf_kernel_increases_with_lengthscale`.

## Connection to effective dimension

For the self-kernel,

$$K=K(X,X),$$

with eigenvalues

$$\sigma_1,\ldots,\sigma_n\geq0.$$

The effective dimension used in this project is

$$\boxed{d_{\mathrm{eff}}(\lambda) = \sum_{i=1}^n \frac{\sigma_i}{\sigma_i+\lambda n}}.$$

Thus the pipeline is

$$X \longrightarrow K(X,X) \longrightarrow \{\sigma_i\}_{i=1}^n \longrightarrow d_{\mathrm{eff}}(\lambda).$$