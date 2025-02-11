# Score Matching

vic-bayer, October 2024

## Introduction

Score-matching is a method for likelihood maximization applied to models of the form:

$$p_\theta(x) = \frac{\tilde{p}_\theta(x)}{Z_\theta}$$ 

where \\( Z_\theta = \int \tilde{p}_\theta(x) \, dx \\) is a generally intractable normalization constant. An example of such a model is an energy-based model, which enjoys a very flexible parameterization as a Boltzmann distribution (also called Gibbs distribution) with the form:

$$p_\theta(x) = \frac{e^{-f_\theta(x)}}{\int e^{-f_\theta(x)} \, dx}$$ 

Here, \\( f_\theta(x) \\) denotes the energy of the distribution in analogy to the Boltzmann distribution. In the context of energy-based model optimization, \\( \int e^{-f_\theta(x)} \, dx \\) is often approximated using Monte Carlo methods, or more specifically, Langevin dynamics. This is generally computationally expensive, although in low dimensions, it might be amenable to parallelization via rejection sampling.

## Score Matching Methodology

The score refers to:

$$\nabla_x \log p(x, \theta)$$ 

which is different from the score \\( \nabla_\theta \log p(x, \theta) \\) used in statistical literature. The idea is that:

$$\nabla_x \log p(x, \theta) = \nabla_x \log \left( \frac{e^{p_\theta(x)}}{\int e^{p_\theta(x)} \, dx} \right) = \nabla_x p_\theta(x) - \nabla_x Z_\theta = \nabla_x p_\theta(x)$$ 

In essence, the score function does not depend on the intractable integral while still containing information about the distribution. As the name suggests, score matching aims at matching the gradient of the empirical data density \\( q(\cdot) \\) with the gradient of the model density \\( p(\cdot, \theta) \\), leading to an optimization of the form:

$$\theta_{\text{opt}} = \operatorname*{argmin}_{\theta} \frac{1}{2} \mathbb{E}_{q} \left\| \nabla_x \log q(x) - \nabla_x \log p(x, \theta) \right\|_2^2$$ 

## Further Analysis

Consider the expression:

$$\frac{1}{2} \left\| \nabla_x \log q(x) - \nabla_x \log p(x, \theta) \right\|_2^2 = \frac{1}{2} (\nabla_x \log q(x))^2 + \frac{1}{2} (\nabla_x \nabla_x \log p(x, \theta))^2 - \frac{1}{2} \nabla_x \log q(x) \nabla_x \log p(x, \theta)$$ 

The first term does not depend on \\( \theta \\), so it is not relevant for the optimization problem.

## Expectation of the Cross-Term

Now consider:

$$\mathbb{E}_{q} \left[ -\nabla_x \log q(x) \nabla_x \log p(x, \theta) \right] = \int -\nabla_x \log q(x) \nabla_x \log p(x, \theta) q(x) \, dx$$ 

This simplifies using integration by parts to:

$$\int -\nabla_x q(x) \nabla_x \log p(x, \theta) \, dx = -\lim_{b \to \infty} \nabla_x \log p(b, \theta) q(b) + \lim_{a \to -\infty} \nabla_x \log p(a, \theta) q(a) + \int \nabla_x^2 p(x, \theta) q(x) \, dx$$ 

The last equality follows from integration by parts. Here, Hyvärinen et al. make the regularity assumptions that:

$$\nabla_x \log p(x, \theta) q(x) \to 0 \quad \text{as} \quad \|x\|_2 \to \infty$$ 

## Limitations and Regularity Assumptions

These assumptions do not hold for general point processes (such as the Poisson process or Hawkes process), which do not necessarily decay sufficiently fast at the boundary. To alleviate this, a weight function enforcing decay can be introduced, as discussed in "Is Score Matching Suitable for Estimating Point Processes?" Furthermore, nominal score matching is not suitable for autoregressive models, which is also addressed in the same work.

## Optimization Problem

Assuming the regularity assumptions hold, the optimization problem can be rewritten as:

$$\theta_{\text{opt}} = \operatorname*{argmin}_{\theta} \mathbb{E}_{q} \left[ \text{Tr} \left( \nabla_x^2 p(x, \theta) \right) - \frac{1}{2} \left\| \nabla_x \log p(x, \theta) \right\|_2^2 \right]$$ 

If the data distribution is in the model class, this optimization yields the optimal value.

