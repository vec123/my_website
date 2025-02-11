## Information Theoretical "Distances" and Geometric Notions

The field of information geometry aims at using the mathematical tools of differential geometry to the study of information science. In the words of publication, "information sciences seek methods to distill information from data to models". Information geometry develops a geometric perspective on information science. While formal descriptions involve too much differential geometry to fit into this work, we provide an intuitive description that relates ubiquitous quantities with geometrical intuitions, such as distances and lines. 

Often, one considers the set of distributions \\( p(x, \theta) \\). Here, \\( \theta \\) are the model parameters, and \\( x \\) is the observed data, defining an empirical distribution \\( q(x) \\). Usually, one aims to maximize the likelihood \\( p(x, \theta) \\) or log-likelihood \\( \log (p(x, \theta)) \\). From the perspective of information geometry, this corresponds to a minimization of the divergence or "distance" between \\( p(x, \theta) \\) and \\( q(x) \\). 

An interesting quantity is the score, which corresponds to the derivative of \\( \log (p(x, \theta)) \\) with respect to the model parameters: 

\\( s(x, \theta) = \frac{\partial \log (p(x, \theta))}{\partial \theta} \\). 

It can be shown that under regularity conditions, the score of the true model \\( \tilde{\theta} \\) is zero. In general, the score indicates the model-parameter sensitivity. This notion of a score-based gradient has been exploited successfully in generative modeling, usually under the name of score matching. 

For each \\( \theta \\), it is possible to define the Fisher Information Matrix as the covariance of the score: 

\\[ \mathbf{I}(\theta) = \mathbb{E}\left[ \left( \frac{\partial \log p(x, \theta)}{\partial \theta} \right) \left( \frac{\partial \log p(x, \theta)}{\partial \theta} \right)^T \right] = -\mathbb{E}\left[ \frac{\partial^2 \log p(x|\theta)}{\partial \theta \partial \theta^T} \right] = \int_{x} \frac{\partial}{\partial \theta} \log(p(x, \theta)^2) p(x, \theta) \, dx \\] 

This corresponds to the variance of the score and thus defines the curvature at \\( \theta \\). Note that the Fisher Information Matrix can be viewed as the Hessian of the Shannon entropy \\( H(p(x, \theta)) = - \int p(x, \theta) \log (p(x, \theta)) \, dx \\). It is not a function of any data point \\( x \\) due to the integral over it. 

The Cramer-Rao bound states that the Fisher Information Matrix is a lower bound on the variance of any unbiased estimator \\( \hat{\theta} \\), so that: 

\\( \text{Cov}(\hat{\theta}) \geq \mathbf{I}(\theta)^{-1} \\). 

A high Fisher Information Matrix corresponds to a strong "curvature" and thus a high sensitivity to distribution parameters. It quantifies how much information a parameter \\( \theta \\) carries. If the corresponding Fisher Information is low, then it does not carry much information (if it is zero, it can be changed to arbitrary values without impacting the likelihood) and vice-versa. 

Furthermore, \\( \mathbf{I}(\theta) \\) is a special Riemannian metric that provides the tangent space of \\( \theta \\) with an inner product. As such, it imposes a Riemannian structure onto the set of distributions \\( p(\theta) \\), which can be used to compute geodesic distances on a statistical manifold. These distances are called Fisher-Rao distances. The distance between two distributions, defined by their parameters \\( \theta_p \\) and \\( \theta_q \\), is defined as the shortest length in parameter space of a geodesic connecting the two. Importantly, the Riemannian structure of the manifold means that the space is locally linear. 

Since computing geodesic distances involves solving, possibly intractable, ordinary differential equations, quadratic form approximations are often used. These approximations are often divergences, such as the KL-divergence. A divergence corresponds to a generalized distance, which does not fulfill the symmetry or triangle-inequality property. However, it is always positive and only zero when both distributions coincide. Divergences are often used to quantify the dissimilarities between points on manifolds. 

A very important divergence is the Kullback-Leibler divergence (or relative entropy): 

\\[ D_{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx. \\] 

By its relation to the Shannon entropy, it measures an information-theoretical distance between two distributions. For small displacements, it can be shown that: 

\\[ D_{KL}(p(x|\theta) \| p(x|\theta + d\theta)) \approx \frac{1}{2} \, d\theta^T I(\theta) d\theta. \\] 

In other words, the KL-divergence behaves like a second-order approximation of the Fisher-Rao distance. Geodesics on a manifold represent the shortest paths between distributions. Moving on such geodesics corresponds to length-optimal transitions. The KL-divergence can be seen as an approximation of the shortest-path length for points that are relatively close. 

Suppose a scalar field \\( l(\theta) = \int_{x}p(x, \theta) \, dx \\) or \\( \log(L(\theta)) = \int \log(p(x, \theta)) \, dx \\) and a vector field \\( \nabla \log l(\theta) \\) on the manifold. The maximum of \\( \log(l(\theta)) \\) corresponds to the best distribution \\( p(x, \theta) \\). 

Using gradient methods, one can move on the manifold to find the optimal point. Newton's method can also be applied by adjusting the step-size according to \\( \mathbf{I}(\theta) \\), where the Fisher Information Matrix corresponds to the Hessian employed in the standard method. This approach is called natural gradient descent or Fisher scoring. add publications. 

A formalization of information geometry analysis of machine learning is beyond the scope of this work. However, when appropriate, this viewpoint will be consulted to add geometrical intuition. 

### Important Distributions and Relations to EM

Consider the set of mixtures: 

\\[ \mathcal{M}_m = \left\\{ p(x|\theta) = \sum_{i=1}^{d} \theta_i p_i(x) \mid \theta_i > 0, \sum_{i=1}^{d} \theta_i = 1 \right\\}. \\] 

And the set of parametric models from the exponential family: 

\\[ \mathcal{M}_e = \left\\{ p(x|\theta) = \exp\left( \sum_{i=1}^{n} \theta_i t_i(x) - \psi(\theta) \right) \mid \theta = (\theta_1, \dots, \theta_s) \right\\}. \\] 

Sometimes, parametric models different from the exponential family might be used, but this family is ubiquitous, has nice properties, and is well studied. 

Consider furthermore the set of distributions with marginal distribution equal to the empirical distribution, defining the data-manifold: 

\\[ \mathcal{D} = \left\\{ q(y, z, \mu) \mid \sum_{z, \mu} q(y, z, \mu) = \frac{1}{N} \sum_{i=1}^{N} \delta(y - y_i) \right\\}. \\] 

The parameter \\( \nu \\) corresponds to the location on the manifold, and \\( q(y, z, \mu) = q(x) q(z|x, \mu) \\). It has no effect on the marginal distribution. 

Often, for example in Gaussian mixture models, we assume that the data distribution is a mixture of exponential families, i.e., it lies in the intersection of \\( \mathcal{M}_e \\) and \\( \mathcal{M}_m \\). 

Consider the set of points: 

\\( r(x, t) = (1-t)p(x) + tq(x) \\) 

and 

\\( \log r(x, t) = (1-t)\log p(x) + t\log q(x) \\), 

called m-geodesics and e-geodesics, respectively. 

It can be shown that any m-geodesic with \\( p(x) \\) and \\( q(x) \\) from \\( \mathcal{M}_m \\) is contained in \\( \mathcal{M}_m \\). Vice-versa, any e-geodesic with \\( p(x) \\) and \\( q(x) \\) from \\( \mathcal{M}_e \\) is contained in \\( \mathcal{M}_e \\). The m- and e-geodesics introduce a notion of flatness into these spaces, which we will not discuss further. The m-geodesic corresponds to a line connecting expectation parameters, while the e-geodesic corresponds to a line connecting natural parameters. 

Starting from a point in the model-manifold, corresponding to the set of exponential families \\( \mathcal{M}_e \\), moving along the e-geodesic to the point \\( q(x) \\) closest to \\( p(x) \\) can be interpreted as finding the point on the exponential manifold closest to \\( p(x) \\). Moving along the m-geodesic to the point \\( p(x) \\) closest to the model distribution \\( q(x) \\) corresponds to choosing the mixture closest to the exponential distribution. 

In essence: 

\\( \mu_{t+1} = \arg \min_{\mu} KL( q(x, z, \mu_t) || p(y, z, h(\theta_t)) ) \\) 

corresponds to the movement along the e-geodesic, also called e-projection, and 

\\( \theta_{t+1} = \arg \min_{\theta}KL( q(x, z, \mu_{t+1}) || p(y, z, h(\theta)) ) \\) 

corresponds to the movement along the m-geodesic, also called m-projection. E-projection updates the abstract parameter \\( \mu \\), which relates distributions to their location on the manifold. 

The closeness is defined in terms of the Kullback-Leibler divergence, which is anti-symmetric, so that \\( z_{t+1} \neq z_{t} \\) and \\( \theta_{t+1} \neq \theta_{t} \\). The EM-algorithm is expected to converge locally. 

#### Figure 1: MLE as m-projection onto a manifold of Boltzmann distributions

![MLE as m-projection](../images/information-geometry-em/mle.png)

MLE as m-projection onto a manifold of Boltzmann distributions.

#### Q-Metric in EM Algorithm

Consider the Q-metric in the classical EM-algorithm: 

\\[ Q(\theta, \theta_t) = \frac{1}{n} \sum_{i=1}^{n} \int_{z} p(z_{\text{opt}}|y_i; \theta_t) \log p(y_i, z_{\text{opt}}, \theta) \, dz \\] 

where 

\\[ z_{\text{opt}} = \arg \min_{z} KL(q(z) || p(z|y, h_\theta(x))) \, dz . \\] 

This corresponds to the e-projection above. We find the point on the data-manifold that best fits the model without changing the marginal data distribution or model parameterization. 

In the maximization step: 

\\[ \theta_{t+1} = \arg \max_{\theta} Q(\theta, \theta_t) \\] 

is calculated. 

#### Figure 2: EM Algorithm Iteration

![EM Algorithm Iteration](../images/information-geometry-em/em-iter.png)

EM Algorithm Iteration

This section examined methodologies for updating model parameterizations when the data distribution depends on an additional unobserved variable \\( z \\). We showed how the \\( KL(p || q) \\)-divergence between the model distribution \\( p \\) and the data distribution \\( q \\) is a lower bound on the log-likelihood defined by the model \\( p \\), while the \\( KL(q || p) \\)-divergence between the data distribution \\( q \\) and the model distribution \\( p \\) formulates the gap between the lower bound and the log-likelihood. The EM algorithm starts with a model distribution, finds the closest data distribution, and then finds the closest model distribution by minimizing the asymmetric KL-divergence. First, the distance between \\( p(z, x, \theta) \\) and \\( q(z, x) \\) is minimized, yielding an optimal \\( z \\), then the distance between \\( p(z, x, \theta) \\) and \\( q(z, x) \\) is minimized, yielding an optimal \\( \theta \\). 

Often, \\( p(z, x, \theta_t) \\) cannot be evaluated. Some methods deal with this by replacing \\( p(z, x, \theta_t) \\) with a parameterization of a simple family (often independent Gaussians). 

![MLE as m-projection](../images/information-geometry-em/mle.png)
![EM Algorithm Iteration](../images/information-geometry-em/em-iter.png)