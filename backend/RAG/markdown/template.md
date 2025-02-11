# Score Matching

vic-bayer, October 2024

## Introduction

Score-matching is a method for likelihood maximization applied to models of the form:

pθ(x) = _𝑡𝑖𝑙𝑑𝑒_(x) / Zθ

where _Z θ = ∫𝑡𝑖𝑙𝑑𝑒(x) dx_ is generally intractable normalization. Energy models have promise here, approximating the integral via Langevin.

## Problem Setup

Score refers to _∇ log p(x, θ)_. Regularity assumptions assume _∇ log q→0 |x|_2→∞_. 

## Energy-Based Models

An example of such a model is an energy-based model which enjoys a very flexible parameterization as a Boltzmann distribution (also called Gibbs distribution) with the form:

pθ(x) = e-fθ(x) / ∫ e-fθ(x) dx

Here, _f θ(x)_ denotes the energy of the distribution in analogy to the Boltzmann distribution. In the context of energy-based model optimization, _∫ e -fθ(x) dx_ is often approximated using Monte Carlo methods, or more specifically, Langevin dynamics. This is generally computationally expensive, although in low dimensions, it might be amenable to parallelization via rejection sampling.

## Score Matching Methodology

The idea is that:

∇x log p(x, θ) = ∇x log [epθ(x) / ∫ epθ(x) dx] = ∇x pθ(x) - ∇x Zθ = ∇x pθ(x)

In essence, the score function does not depend on the intractable integral while still containing information about the distribution. As the name suggests, score matching aims at matching the gradient of the empirical data density _q(·)_ with the gradient of the model density _p(·, θ)_ , leading to an optimization of the form:

θopt = argminθ (1/2) Eq ||∇x log q(x) - ∇x log p(x, θ)||22

## Further Analysis

Consider the term:

(1/2) ||∇x log q(x) - ∇x log p(x, θ)||22 = (1/2) (∇x log q(x))2 \+ 

## Energy-Based Models

An example of such a model is an energy-based model which enjoys a very flexible parameterization as a Boltzmann distribution (also called Gibbs distribution) with the form:

pθ(x) = e-fθ(x) / ∫ e-fθ(x) dx

Here, _f θ(x)_ denotes the energy of the distribution in analogy to the Boltzmann distribution. In the context of energy-based model optimization, _∫ e -fθ(x) dx_ is often approximated using Monte Carlo methods, or more specifically, Langevin dynamics. This is generally computationally expensive, although in low dimensions, it might be amenable to parallelization via rejection sampling.

## Score Matching Methodology

The idea is that:

∇x log p(x, θ) = ∇x log [epθ(x) / ∫ epθ(x) dx] = ∇x pθ(x) - ∇x Zθ = ∇x pθ(x)

In essence, the score function does not depend on the intractable integral while still containing information about the distribution. As the name suggests, score matching aims at matching the gradient of the empirical data density _q(·)_ with the gradient of the model density _p(·, θ)_ , leading to an optimization of the form:

θopt = argminθ (1/2) Eq ||∇x log q(x) - ∇x log p(x, θ)||22

## Further Analysis

Consider the term:

(1/2) ||∇x log q(x) - ∇x log p(x, θ)||22 = (1/2) (∇x log q(x))2 \+ (1/2) (∇x ∇x log p(x, θ))2 \- (1/2) ∇x log q(x) ∇x log p(x, θ)

The first term does not depend on _θ_ , so it is not relevant for the optimization problem.

## Expectation of the Cross-Term

Now consider:

Eq [-∇x log q(x) ∇x log p(x, θ)] = ∫ -∇x log q(x) ∇x log p(x, θ) q(x) dx

This simplifies using integration by parts to:

∫ -∇x q(x) ∇x log p(x, θ) dx = -limb→∞ ∇x log p(b, θ) q(b) + lima→-∞ ∇x log p(a, θ) q(a) + ∫ ∇x2 p(x, θ) q(x) dx

The last equality follows from integration by parts. Here, Hyvärinen et al. make the regularity assumptions that:

∇x log p(x, θ) q(x) → 0 when ||x||2 → ∞

## Limitations and Regularity Assumptions

These assumptions do not hold for general point processes, such as the Poisson process or Hawkes process, which do not necessarily decay sufficiently fast at the boundary. To alleviate this, a weight function enforcing decay can be introduced, as discussed in "Is Score Matching Suitable for Estimating Point Processes?" Furthermore, nominal score matching is not suitable for autoregressive models, which is also addressed in the same work.

## Optimization Problem

Assuming the regularity assumptions hold, the optimization problem can be rewritten as:

θopt = argminθ Eq [Tr(∇x2 p(x, θ)) - (1/2) ||∇x log p(x, θ)||22]

If the data distribution is in the model class, this optimization yields the optimal value.

