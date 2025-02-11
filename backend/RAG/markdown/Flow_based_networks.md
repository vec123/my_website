## Flow Based Networks

![A normalizing flow](../images/flow/flow.jpg) A normalizing flow. _Source:[Tutorial on Normalizing Flows](https://creatis-myriad.github.io/tutorials/2023-01-05-tutorial_normalizing_flow.html)_

Flow based Models are probabilistic Models which enable computation of the maximal likelihood estimation directly. This contrasts GANS, where the MLE-problem is replaced by a minimax game and solved only implicitly. In VAEs and Diffusion Models, the MLE-problem is replaced by a lower-bound instead, made necessary by the problem structure and the intractability of \\(p(x) = \int_{z} p(x,z) dz\\). 

In contrast to these methods, flow based models perform a series of invertible computations which make the Maximum Likelihood Problem tractable. 

Let \\(p(x)\\) be the density function of the data distribution \\(X \in \mathbb{R}^d\\). Let \\(p(f(x))\\) be a transformed density function of a transformed latent distribution \\(Z \in \mathbb{R}^d\\). This mapping \\(f: X \rightarrow Z\\) is by design invertible and can be used to generate new data-points by \\(x = f^{-1}(z)\\). 

Given such an invertible function \\(f\\) exists, we can write with some abuse of notation: 

\\[ \int_{-\infty}^{+\infty} p( x ) dx = \int_{-\infty}^{+\infty} p( z ) dz = \int_{-\infty}^{+\infty} p( f(x) ) dx. \\] 

Constricting to a subspace \\(\alpha \in X\\) and \\(\hat{\alpha} \in Z\\), we can write: 

\\[ \int_{\alpha} p(x) dx = \int_{\hat{\alpha}} p(z) dz = \int_{\alpha} p( f(x) ) dx. \\] 

Consider a single region only: 

\\[ p( x ) dx = p( z) dz = p( f(x) ) dx. \\] 

\\[ p( x ) \left| \frac{dx}{dz} \right| = p( x ) \left| \frac{dx}{df(x)} \right| = p( z ). \\] 

\\[ p( x ) = p( z ) \left| \frac{dz}{dx} \right| = p( z ) \left| \det\left( \frac{df(x)}{dx} \right) \right|. \\] 

These two relations describe the transformation law between the two densities. Since the density is always positive, we apply the absolute norm to the Jacobian term. Generalization to multivariate distributions results in: 

\\[ p( x ) = p( z ) \left| \det\left( \frac{df(x)}{dx} \right) \right|. \\] 

This can be written as the log-likelihood: 

\\[ \log( p( x ) ) = \log( p( z ) ) + \log\left( \det\left( \frac{df(x)}{dx} \right) \right). \\] 

Assume: 

\\[ z = f(x) = f_n \circ \ldots \circ f_1(x), \\] 

where \\(f(x)\\) is the concatenation of \\(n\\) invertible functions. The log-likelihood becomes: 

\\[ \log( p( x ) ) = \log( p( z ) ) + \sum_{i=1}^{n}\log\left( \det\left( \frac{df_i(x)}{df_{i-1}(x)} \right) \right), \\] 

where \\(f_0 = x\\). In theory, \\(f_{i}\\) can be any diffeomorphism (bijective function). However, sensible requirements are that its inverse and the determinant of its Jacobian can be easily computed. The difficulty lies in finding such functions which are simple enough yet capable of simplifying the MLE-problem sufficiently while maintaining high expressive power. 

### Planar and Radial Flows

Planar flows apply: 

\\[ f(z) = z + u\sigma(w^T z + b), \\] 

where \\(\lambda = \\{w \in \mathbb{R}^D, u \in \mathbb{R}^D, b \in \mathbb{R}\\}\\) and \\(\sigma\\) is a differentiable nonlinear function. Since: 

\\[ z = f(z) - u\sigma(w^T z + b), \\] 

the function is invertible. Furthermore, the Jacobian determinant is described by: 

\\[ \left|\det\left( \frac{\partial f(z)}{\partial z} \right)\right| = 1 + u^T \sigma'(w^T z + b)w, \\] 

and can be computed efficiently. 

A related family of flows, the radial flow, is described by: 

\\[ f(z) = z + \beta \sigma(\alpha, r)(z + z_0), \\] 

where \\(r = |z - z_0|\\), \\(\sigma(\alpha, r) = \frac{1}{\alpha + r}\\), and \\(\lambda = \\{z_0 \in \mathbb{R}^D, \alpha \in \mathbb{R^+}, \beta \in \mathbb{R}\\}\\). The determinant of the Jacobian in this case is: 

\\[ \left|\det\left( \frac{\partial f(z)}{\partial z} \right)\right| = \left[1 + \beta \sigma(\alpha, r)\right]^{d-1} \left[1 + \beta \sigma(\alpha, r) + \beta \sigma'(\alpha, r)\right]. \\] 

### Coupling Layers

A family of normalizing flows is called coupling layers. 

The input is split into two parts \\(x_{1:D} = [x_{1:d}; x_{d+1:D}]\\). The output is then obtained separately by operating on these parts: 

\\[ z_{1:d} = x_{1:d}, \quad z_{d+1:D} = g(z_{d+1:D}, m(z_{1:d})). \\] 

The inverse is obtained by: 

\\[ x_{1:d} = z_{1:d}, \quad x_{d+1:D} = g^{-1}(z_{d+1:D}, m(z_{1:d})). \\] 

The Jacobian matrix is: 

\\[ \begin{pmatrix} I_d & 0 \\\ \frac{\partial g(z_{d+1:D}, m(z_{1:d}))}{\partial z_{0:d}} & \frac{\partial g(z_{d+1:D}, m(z_{1:d}))}{\partial z_{d+1:D}} \end{pmatrix} \\] 

The advantage of this approach is that \\(\frac{\partial g(z_{d+1:D}, m(z_{1:d}))}{\partial z_{0:d}}\\) is not part of its determinant, and therefore \\(m(z_{1:d})\\) does not need to be inverted for the transformation between densities. It can be represented by any non-invertible function, in the most general case by a Transformer. However, \\(g(.)\\) must be chosen in an easily differentiable and invertible way. 

Since coupling layers only transform part of the input, they are often stacked and alternated with permutations. 

![A normalizing flow](../images/flow/flow.jpg)