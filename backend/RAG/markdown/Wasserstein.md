# Wasserstein distance and algorithms

## Introduction

The Wasserstein distance is a distance between shapes based on optimal transport. Consider two probability densities \\(p(x)\\) and \\(q(x)\\) defining the distributions \\(P\\) and \\(Q\\), so that \\( x \sim P \\) and \\( y \sim Q \\) with domains \\( x \in \Omega_x\\) and \\( y \in \Omega_y\\) . Its original formulation is the Monge-Kantorovich formulation which considers the optimal transport cost by 

$$ W_p(P,Q) =\inf_T \int_{Omega_x} c(x,T(x)) dp(x) = \inf_T \int c(x,T(x)) p(x) dx. $$ 

The \\(p\\) Waserstein distance is the optimal transport cost with cost function \\( c(x,y) = ||x-y||^p\\). 

$$ W_p(P,Q) = \inf_T \int_{Omega_x} || x - T(x) ||^p dP(x). $$. 

Here \\( dP(x) \\) is the probability measure and \\( T(x) \\) is the transport map. If the minimizer \\( T^*(x) \\) exists it is the optimal transport map. However, in this formulation, a solution is not guaranteed to exist. For example, if \\(p(x)\\) corresponds to a point-mass and \\(q(x)\\) corresponds to multiple point-masses, then, due to the usage of equal probability measure for both distributions, the mass can not be splitted. The non-existance of a solution is resolved by the Kantorovich formulation, which allows for splitting of mass and furthermore enables the usage of a dual formulation for computing the distance. When both probability distributions are continous or an one-to-one mapping exists, then \\( T^*(x) \\) exists and is unique. Due to the simplicity of the Monge formulation and the explicit transport map, it is often used in practice. In many cases a one-to-one mapping exists, so that the Monge formulation can be used.   
  
An alternative formulation is the soft formulation, leading to the Kantorovich's formulation of the Wasserstein distance. The optimal transport problem might be written as 

$$ W_p(P,Q) =\inf_{\gamma(x,y) \ in \Gamma } \int_{Omega_x \times Omega_y} c(x,y)d\gamma(x,y), $$ 

where \\( \gamma(x,y) \\) can be interpreted as the amount of mass that is moved from \\(x\\) to \\(y\\) and \\( \Gamma = \\{ \gamma(x,y) : \int_{Omega_x} \gamma(x,y) dx = P(y) \text{and} \int_{Omega_y} \gamma(x,y) dz = P(x)\\} \\) is the set of all allowed transport plans subject to the constraint. This allows for mass splitting and is more general than the Monge formulation. If an optimal transport map exists, the corresponding optimal transport map can be obtained by 

$$ \gamma(\mathcal{A},\mathcal{B}) = \int_{x \in \mathcal{A}: T(x) in \mathcal{B}} p(x) dx, $$ 

where \\( \mathcal{A} \subset \Omega_x \\) and \\( \mathcal{B} \subset \Omega_y \\) are the domain in which the transport map is considered. The Wasserstein distance in the Kantorovich formulation is 

$$ W_p(P,Q) = (\inf_{\gamma(x,y) \ in \Gamma } \int_{Omega_x \times Omega_y} ||x-y||^p \gamma(x,y) )^\frac{1}{p} $$ 

under the constraint that \\( \int_y \gamma(x,y) dy = P(x) \\) and \\(\int_x \gamma(x,y) dx = Q(y) \\). The minimizer \\( \gamma^*(x,y) \\) is called the optimal transport plan and is guaranteed to exist. If an optimal transport map exists, it can be related to an optimal transport plan by \\( \gamma(x,y) = \int_{x \in \mathcal{X}: T(x) in \mathcal{Y}} p(x) dx \\), where \\( \mathcal{X} \subset \Omega_{x} \\) and \\( \mathcal{Y} \subset{\Omega_y} \\) are the input and output spaces. If \\(\gamma(x,y) \\) is one-to-one, i.e. no splitting of mass occurs (input and output measure coincide), there exists an optimal transport map. For many practical applications this is the case. Note that minimizing the Wasserstein distance returns the minimal cost, as well as a corresponding transport plan. As an example, consider two greyscale images in \\(2D\\) which can be interpreted as probability distributions by normalization of the intensities. Then the optimal transport map is a \\(2D\\) vector-field, where each point receives a \\(2D\\)-vector with the corresponding displacement. ****In a discrete setting we might have samples \\( \\{x_i\\}_{i=1,...,N} \\) and \\( \\{y_i\\}_{i=1,...,M}\\) which occur with probability \\( p_i \\) and \\( q_j \\) respectively, defining PDFs of the form \\(p(x_i) = \sum_{i=1}^N p_i \delta(x -x_i) \\) and \\(q(y_i) = \sum_{i=1}^N q_i \delta(y -y_i) \\). The Wasserstein distance between the discretely sampled distributions can be written as

$$ W_p(P,Q) = \inf_{\gamma} \sum_i \sum_j ||x_i-y_j||^p \gamma(x_i,y_j) $$ 

so that \\( \sum_j \gamma(x_i, y_j) = p_i \\), and \\( \sum_i \gamma(x_i, y_j) = q_j \\) and \\( \gamma(x_i,y_j) \geq 0 \\). In general, this problem does not have an optimal transport map, but an optimal transport plan. It is a convex (but not strictly convex) optimization problem.   
  
Interesting cases are \\( p=1 \\) (also called Earth-Mover distance) and \\( p=2 \\). For \\( p=2 \\), it can be shown that there exists a unique convex function \\( \phi \\) so that \\( \gamma^*(x,y) = \nabla \phi \\). Furthermore, for \\( p= 2\\) the metric space \\( P(\Omega), W_p \\) has the structure of a Riemannian manifold. This might (maybe) also hold for other values of \\( p \\) (see more [Wasserstein Riemannian geometry of Gaussian densities ](https://link.springer.com/article/10.1007/s41884-018-0014-4)).   
  
Computing the Kantorvich Wasserstein distance is a convex optimization problem (with linear transport cost) and can be written in a dual formulation. Consider that if \\( \gamma \in \Gamma \\) (as defined above), then 

$$ \sup_{\Phi, \Psi} \int_{\Omega_x} \Phi(x)dp(x) + \int_{\Omega_y} \Psi(y)dq(y) - \int_{\Omega_x \times \Omega_y} (\Phi(x) - \Psi(y)) d\gamma(x,y) = 0, $$ 

whereas the same quanitiy becomes \\( \infty \\) if \\( \gamma \notin \Gamma \\) (see here on duality). By adding this constraint to the objective by the well-known lagrangian \\(\lambda\\) method in convex optimization, we obtaine 

$$ W_p(P,Q) = \inf_{\gamma} \int{\Omega_x \times \Omega_y} c(x,y) d\gamma(x,y) + \sup_{\Phi, \Psi} \int_{\Omega_x} \Phi(x)dp(x) + \int_{\Omega_y} \Psi(y)dq(y) - \int_{\Omega_x \times \Omega_y} (\Phi(x) - \Psi(y)) d\gamma(x,y). $$. 

Exchanging supremum and infimum as commonly done to establish the Dual, we obtain 

$$ W_p(P,Q) = \sup_{\gamma \in \Gamma} \inf_{\Phi, \Psi} \int_{\Omega_x} \Phi(x)dp(x) + \int_{\Omega_y} \Psi(y)dq(y) + \inf_{\gamma} \int{\Omega_x \times \Omega_y} c(x,y) - \int_{\Omega_x \times \Omega_y} (\Phi(x) - \Psi(y)) d\gamma(x,y). $$. 

In many cases strict duality holds, we refer to here for more details and proofs. We can rewrite the infimum 

$$ \inf_{\gamma} \int{\Omega_x \times \Omega_y} c(x,y) - \int_{\Omega_x \times \Omega_y} (\Phi(x) - \Psi(y)) d\gamma(x,y) = 0. $$ 

if \\( \Phi(x) - \Psi(y) \leq c(x,y) \\) and \\( -\infty \\) otherwise. This leads to the dual formulation 

$$ W_p(P,Q) = \sup_{\Phi, \Psi} \int_{y} \Phi(y)dQ(y) + \int_{x} \Psi(x)dP(x) $$ 

so that \\( \Phi(x) - \Psi(y) \leq c(x,y) \\). For \\( p = 1 \\) this can be transformed into 

$$ W_p(P,Q) = \sup_{f} \int_{x} f(x)dq(x) - \int_{x} f(x)dP(x), $$ 

with \\(f(x) \in \mathcal{F}:\\{ f(x) - f(y) \leq c(x,y) \\}\\). This corresponds to the loss-term of Wasserstein GANs. Another important fact is that the computing the Wasserstein distance is expensive, unless the probabilities \\( p(x) \\) and \\( q(x) \\) are one dimensional. In the one-dimensional case the distance has a closed form 

$$ W_p(P,Q) = (\int_{\Omega_x }| \mathcal{F}^{-1}(x) - \mathcal{G}^{-1}(x) |^p dx)^{\frac{1}{p}}. $$ 

where \\( \mathcal{F} \\) and \\( \mathcal{G} \\) are the cumulative distribution functions of \\( P \\) and \\( Q \\) respectively. This relation (in \\(1D\\)) to the cumulative distribution function is mentioned again in the cumulative distribution transform section on eulcidean embeddings. Consider the cumulative distribution functions of the respective densities 

$$ P(x) = \int_{\inf(\mathcal{X})}^x p(x)dx. $$ 

The more general pseudo-inverse inverse is defined as 

$$ P(z)^{-1} = \inf\\{ x \in \Omega_x : P(x) \geq z \\}. $$ 

If an inverse exist the pseudo-inverse is equal to the inverse. The optimal transport map (if it exists) can be obtained from the cumulative distribution functions and their (pseudo-)inverses as 

$$ T(X) = Q^{-1}( P(x)). $$ 

This result holds, because in \\(1D\\) both distributions are supported on the real line \\( \Omega_x, \Omega_y \in \mathcal{R}\\), allowing for a natural ordering. In this case one can state the requirement for the optimal transport map (which exists if a one-to-one mapping is possible) as 

$$ \int_{x: T_t^*(x) \in \Omega_x } P(x)dx = \int_{\Omega_y} L(y)dy, $$ 

i.e. the transport map is so that the cumulative distribution is preserved. If the mapping is not one-to-one, which can happen for discrete PDFs (whose CDFs are step functions), splitting of mass occurs. i.e. some mass from \\(P\\) is moved to multiple points \\(Q\\). We will assume that the optimal transport map exists and that it is Lebesque integrable, i.e. \\(T(x) \in L^2(\Omega_x)\\) with 

$$ (\int_{\Omega_x} |T(x)|^2 dx)^{0.5} \leq \infty, $$ 

and \\( \lambda \\) being the Lebesque measure. The optimal transport problem in \\(1D\\) and correspondingh \\(p-\\)Wasserstein distance are especially interesting cases because they have a closed form. This result is especially interesting when considering the high computational complexity of the Wasserstein distance in higher dimensions. Sliced Wasserstein approximate distance and transport plan by projection onto \\(1D\\) distributions (see [here](https://proceedings.neurips.cc/paper_files/paper/2019/file/f0935e4cd5920aa6c7c996a5ee53a70f-Paper.pdf) and [here](https://arxiv.org/abs/2410.12176)).   
  
In general the Wasserstein space is a metric space, with the Wasserstein distance as a metric. It is furthermore a geodesic space which becomes a Riemannian manifold in the \\(2-\\)Wasserstein case. This allows for the definition of shortest paths between probability distributions and signals which can be interpreted as such (i.e. images). More on this in the section on geometry and geodesics.   
The section on the cumulative distribution transform shows how projection onto \\(1D\\) can define embeddings whose euclidean distance (inner product) corresponds to the \\(2-\\)Wasserstein distance. The cumulative distribution transform is isometric to the \\(2-\\)Wasserstein distance. Furthermore, certain non-linear transformations (such as scaling, translation and composition) become linear in the transform space, enabling the application of linear methods. Additionally, the transform is invertible. These results are related to kernel methods, with the transform acting as a kernel. Contrary to popular kernel methods which work with the inner-product directly nad ignore the underlying transform, the CDT explicitely defines the kernel transform. 

## Geometry and Geodesics

The \\(p-)\ Wasserstein distance (remember \\( c(x,y) = |x-y|^p \\)) is a metric on \\( P(\Omega)\\), i.e. it satisfies the triangle inequality, symmetry and non-negativity. Here \\( P(\Omega)\\) represents the set of probability densities on the domain \\( \Omega \\). The metric space ( \\( P(\Omega), W_p \\) ) is the p-Wasserstein space. Furthermore (see [here](https://ieeexplore.ieee.org/document/7974883)) the metric space \\( P(\Omega), W_p \\) is a geodesic space, i.e. there exists a continous path between any two points /(p(x), q(y) \in \P(\Omega)/) in the space. When a unique optimal transport map \\(T(x)^*) from \\( P \\) to \\( Q \\) exists, the geodesic is obtained by moving the mass at constant speed from \\(x\\) to \\(T(x)^* \\). For \\( t \in [0,1] \\) the location of the mass at time \\(t\\) and position \\(x\\) is given by \\( T_t^*(x) = (1-t)x + t T(x)^* \\). Its velocity is given by \\( \dot{T}_t^*(x) = T(x)^* - x \\). Pushing forward the mass through \\( T_t^*(x) \\) is subject to the condition 

$$ \int_{x: T_t^*(x) \in \Omega_z } P(x)dx = \int_{\Omega_z} L(z)dz, $$ 

(where \\(P(x)\\) and \\(L(z)\\) are the respective distributions) which can be written in differential form (see here ) as 

$$ \det(D T_t^*(x)) L(T_t^*(x)) = P(x) $$ 

(where \\(D T_t^*(x)\\) is the Jacobian of the transport map) so that 

$$ L(T_t^*(x)) = \frac{P(x)}{\det(D T_t^*(x))}. $$ 

This defines a path \\(T_t^*(x) = I(x,t)\\) in the Wasserstein space with tangent vector \\( s(x,t) = \frac{ \partial I(x,t) }{ \partial t(x,t) }\\). Especially interesting is \\( P(\Omega), W_2 \\) which is a Riemannian manifold with a formal Riemannian metric (see [here](https://www.tandfonline.com/doi/full/10.1081/PDE-100002243)). For this case a vector field \\(v(x,t)\\) can be defined so that 

$$ s(x,t) = -\nabla (I(x,t)v(x,t)). $$ 

The inner product of the Riemannian manifold is defined as 

$$ \langle s, s \rangle = \min \int |v(x,t)|^2 I(x,t )dx . $$ 

Utilizing this, the 2-Wasserstein metric can be reformulated as 

$$ W_2^2 (P,Q) = \inf_{T,v} \int_0^1 \int_{\Omega} |v(x,t)|^2 I(x,t) dx dt. $$ 

so that \\( \partial_t T + \nabla(Pv) = 0\\) (this is the continuity equation) and \\( T(x,0) = p(x) \\) and \\( T(x,1) = q(x) \\). This formulation is interesting because it allows for interpolation between shapes by moving along a geodesic in the 2-Wasserstein space. 

## The Cummulative Distribution Transform 

Consider the measures \\(p \in \Omega_p \subset \mathcal{R} \\) and \\(q \in \Omega_q \subset \mathcal{R} \\) with probability spaces \\( (\mathcal{X}, \Sigma(\mathcal{X}), p) \\) and \\( (\mathcal{Y}, \Sigma(\mathcal{Y}), q) \\) respectively. Denote the density functions \\( dp(x) = p(x)dx \\) and \\( dq(x) = q(x)dx \\). A transport map \\( T: \Omega_p \rightarrow\Omega_ \\) that pushes /( p(x) /) onto /( q(x) /) can be expressed as 

$$ T(X) = P^{-1}( Q(x)). $$ 

and 

$$ Q(X) = P^(T(x)). $$ 

which through the derivative becomes 

$$ q(X) = T(x)' p^(T(x)). $$ 

We assume that \\(T\\) is an element of the \\(L^2\\) function space, i.e. it is square integrable 

$$ |T|_2 = ( \int_{\mathcal{X} } |T|^2 d\lambda)^{0.5} \leq \infty, $$ 

where \\(\lambda\\) is the Lebesgue measure in \\( \mathcal{X} \\). Consider a function \\( \hat{p}: \mathcal{X} \rightarrow \mathbb{R} \\) as 

$$ \hat{q}(x) = (T(x) - x)\sqrt{p(x)}. $$ 

This defines the Cumulative Distribution Transform (CDT) of \\(q\\) with respect to \\(p\\). Note, that the \\(L^2\\) norm of the CDT \\( \hat{q}(x) \\) corresponds to the \\(2-\\)Wasserstein distance between the densities \\(p(x)\\) and \\(q(x)\\). In fact the CDT is isometric to the \\(2-\\)Wasserstein distance. The inverse of the Cumulative Distribution Transform (CDT) of \\( \hat{q}(x) \\) is defined as 

$$ q(y) = T^{-1}(y)'p( T^{-1}(y) ) $$. 

See [here](https://www.sciencedirect.com/science/article/pii/S1063520317300076) for more details and examples on the uniform distribution and gaussian distribution, defined by \\(p(x)\\) and \\(q(x)\\) respectively. Figure three and five show how overlapping gaussian distributions and non-convex (but disjoint) sets can be seperated linearly after application of the transform. The CDT is a non-linear transformation. Certain non-linear transformations (such as scaling, translation and composition) become linear in the transform space, enabling the application of linear methods. These properties are listed [here](https://www.sciencedirect.com/science/article/pii/S1063520317300076) and are not repeated. Find a video [here](https://www.youtube.com/watch?v=khkSOleeEno) on the CDT. 

## Slicing, expected Transport Plans, CDT in multiple dimensions

Sections above showed that the Wasserstein distance can be expensive to compute, especially in high dimensions. However the one-dimensional case has a closed form solution and allows for the definition of the cumulative distribution transform. Slicing probability distributions into one-dimensional distributions can be used to approximate the Wasserstein distance, its transport plan and apply the CDF in higher dimensions. For projection the Radon transform is used. The Radon transform is a linear bijective transformation. It is defined by 

$$ Rf(\theta, t) = \int_{\mathcal{R}^d} f(x) \delta(t - \langle x, \theta \rangle) dx, $$ 

where \\( f(x) \in \mathcal{L_1} \\), \\( (t,\theta) \in \mathcal{R} \times \mathcal{S}^{d-1}\\) (\\( \mathcal{S}^{d-1} \\) is the unit sphere) and \\( \delta \\) is the one-dimensional Dirac delta function. A hyperplane can be written as 

$$ H_{\theta,t} = \\{ x \in \mathcal{R}^d : \langle x, \theta \rangle = t \\}. $$ 

For a fixed \\( \theta \\) the Radon transform is a projection of the function \\( f(x) \\) onto the hyperplane \\( H_{\theta,t} \\), so that \\( Rf(\theta, \cdot) : \mathcal{R} \rightarrow \mathcal{R} \\). Its inverse is defined as 

$$ f(x) = R^{-1}(Rf(\theta, t)) = \int_{\mathcal{S}^{d-1}} Rf(\theta, t) * \nu( \langle x, \theta \rangle) d\theta, $$ 

where \\( \nu \\) is a one-dimensional high-pass filter and * is the convolutional operator. This inverse is also known as filtered back-projection and used in medical tomography for image reconstruction. The integration of the filtered back-projection is usually replaced by a monte carlo approximation (i.e. a finite sum over projections). This conclueds the introduction on the Radon transform and its inverse. To gain a more intuitive and visual understanding of the Radon transform, i recommend [this](https://www.youtube.com/watch?v=f0sxjhGHRPo) video.   
  


##  Notation 

  
  
A notational pecularity is \\( T_{\\#}P \\), which corrsponds to a new probability distribution induced by \\( P \\) through the mapping \\( T \\). Consider the optimal transport map \\( T: \mathbb{R}^d \rightarrow \mathbb{R}^d \\), where the distribution of \\( T(X) \\) is called push-forward ( \\( T_{\\#}P \\) ) of \\(P\\), so that 

$$ T_{\\#}P (A) = P(\\{ x: T(x) \in A \\}) = P(T^{-1}(A)). $$ 

This is also called the push-forward /( T_{\\#}P /) of /(P/). 

This is useful for high-dimensional distributions where exact numerical computation can be too expensive. By representing shapes and images as probability distributions, one can use the Wasserstein distance to compare shapes, compute mean shapes and interpolate with transport plans.   
  
Sources:   
  
[Optimal Transport and Wasserstein Distance - Carnegie university](https://stat.cmu.edu/~larry/=sml/Opt.pdf),   
[Optimal Mass Transport - Signal processing and machine-learning applications](https://ieeexplore.ieee.org/document/7974883),   
[Sliced-Wasserstein Autoencoder: An Embarrassingly Simple Generative Model](https://arxiv.org/abs/1804.01947),   
[An Invitation to Statistics in Wasserstein Space](https://link.springer.com/book/10.1007/978-3-030-38438-8),   
[Optimal Transport for Applied Mathematicians](https://link.springer.com/book/10.1007/978-3-319-20828-2),   
[Sliced Wasserstein Kernels for Probability Distributions](https://ieeexplore.ieee.org/document/7780937),   
[Sliced Wasserstein Distance for Learning Gaussian Mixture Models](https://arxiv.org/abs/1711.05376),   
[Introduction to Optimal Transport](https://www.damtp.cam.ac.uk/research/cia/files/teaching/Optimal_Transport_Notes.pdf),   
[The cumulative distribution transform and linear pattern classification](https://www.sciencedirect.com/science/article/pii/S1063520317300076),   
[A continuous linear optimal transport approach for pattern analysis in image datasets](https://www.sciencedirect.com/science/article/pii/S0031320315003507),   
[Github Repo with Implementations](https://github.com/skolouri), 

