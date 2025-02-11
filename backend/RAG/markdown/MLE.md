# MLE MAP and EM

## MLE $\rightarrow$ MAP $\rightarrow$ ELBO $\rightarrow$ EM $\rightarrow$ VAE

This is a description of maximum likelihood estimation, maximum a-posteriori and their connection with each other. These concepts are then extended to distributions with unobserved (latent) variables, leading to expectation maximization which, in turn, is related to gaussian mixture models and variational auto-encoders.

We start with maximum likelihood estimation since it is the easiest. Then the viewpoint is extended to Bayesian maximum a-posteriori, which assumes probabilistic model-parameters. Both methods turn out to be equivalent if one incorporates prior knowledge into maximum likelihood estimation by a regularization term.

Subsequently, distributions with latent variables are examined. The resulting integral and tractability problematic is explained and how it gives rise to the important quantity called evidence lower bound (ELBO). Then, under consideration of this lower bound, the expectation maximization algorithm is explained in its general formulation, on the example of Gaussian Mixture Models, where the optimal posterior can be computed. This is then extended to variational auto-encoders, where intractabilitiy means that the posterior must be parameterized and optimized to fit an assumed prior via gradient descent.

### MLE - Maximum Likelihood Estimation

Consider a model, parameterized by $\theta$, so that,

$$ x = h(\theta) + \epsilon $$ 

with $ \epsilon \sim \mathcal{N}(0,\sigma^2) $ and $\theta $ deterministic. Then $x \sim \mathcal{N}( h(\theta), \sigma^2)$. 

The probability density function of this Gaussian is defined as

$$ p(x, h(\theta) ) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( \frac{-(y - h(\theta) )^2}{2\sigma^2} \right). $$ 

Given data-samples $(x_{i})$ with $i = 1,...,N$, Maximum Likelihood Estimation amounts to finding the model which results in the maximum joint probability of 

$$ \prod_{i=1}^{N} p(x_{i}| h(\theta)). $$ 

For $N= \infty$, by the strong law of large numbers

$$ p(x| h(\theta)) = \prod_{i=1}^{N} p(x_{i}, h(\theta)). $$ 

The problem can be cast into an optimization problem:

$$ \begin{aligned} \theta_{opt}(x) &= \operatorname*{argmax}_{\theta} \prod_{i=1}^{N} p(x_{i}, h(\theta)) \\\ &= \operatorname*{argmax}_{\theta} \sum_{i=1}^{N} \ln(P(x_{i}, h(\theta))) \\\ &= \operatorname*{argmin}_{\theta} - \sum_{i=1}^{N} \ln(P(x_{i}, h(\theta))) \\\ &= \operatorname*{argmin}_{\theta} - \sum_{i=1}^{N} \left[\ln\left(\frac{1}{\sqrt{2\pi \sigma^2}}\right) + \frac{-(x_{i} - h(\theta))^2}{2\sigma^2}\right] \\\ &= \operatorname*{argmin}_{\theta} - \sum_{i=1}^{N} \frac{-(x_{i} - h(\theta))^2}{2\sigma^2} \\\ &= \operatorname*{argmin}_{\theta} \sum_{i=1}^{N} (x_{i} - h(\theta))^2 \ast \text{const.} \end{aligned} $$ 

Often $ \text{const.}=\frac{1}{N} $. The $h_{\theta}$, which minimizes this function, maximizes $\prod_{i=1}^{N} p(x_{i}| h_{\theta} )$. Given a sufficient amount of data-samples, this approximation of the data-density is exact, assuming that a Gaussian parameterization is suitable. The proposed method uses data-samples $(x_{i})$ to extract the most likely parameterization of $x=h(\theta)$.

Note, that:

$$ \sum_{i=1}^{N} (x_i - h(\theta))^2 \ast \frac{1}{N} $$ 

is often referred to as _mean squared error_. Minimizing the mean squared error returns optimal model parameters if the data distribution is Gaussian. It can be proven that the estimator $h(\theta_{opt})$, which maximizes $p(x|h(\theta))$, is an unbiased minimum variance estimator (UMVE) for $x = h(\theta) + \epsilon$.

Summarizing, Maximum Likelihood Estimation amounts to computing

$$ \theta_{opt}(x) = \operatorname*{argmin}_{\theta} \sum_{i=1}^{N} (x_{i} - h(\theta))^2 \ast \text{const.} $$ 

d 

### MAP - Maximum A-Posteriori

Consider a model, parameterized by $\theta$, so that,

$$ x = h(\theta) + \epsilon $$ 

with $ \epsilon \sim \mathcal{N}(0,\sigma^2) $ and $ \theta \sim \mathcal{N}(0,\tau^2) $. Contrary to MLE, we assume a probabilistic model parameterization.

Then, by definition, the density function of this Gaussian can be written as:

$$ p(\theta) = \frac{1}{\sqrt{2\pi\tau^2}} \exp\left(\frac{\theta^2}{2\tau^2}\right). $$ 

Furthermore, given a dataset $\\{x_i\\}$, note that:

$$ p(x, h(\theta)) = p(\theta | \\{x_1,...,x_N\\}) p(\\{x_1,...,x_N\\}) = p(\\{x_1,...,x_N\\}|\theta)p(\theta) $$ 

where $p(\\{x_1,...,x_N\\}) = \text{const.}$ is determined by the data.

Therefore,

$$ \theta_{opt} = \operatorname*{argmax}_{\theta} p(\theta|\\{x_1,...,x_N\\}) $$ 

which is equivalent to:

$$ \theta_{opt} = \operatorname*{argmax}_{\theta} \prod_{i=1}^{N} [p(x_i|\theta)]p(\theta) $$ 

which leads to the following optimization problem:

$$ \theta_{opt} = \operatorname*{argmin}_{\theta} -\sum_{i=1}^{N} [\ln(p(x_i|\theta))] - \ln(p(\theta)). $$ 

This leads to:

$$ \theta_{opt} = \operatorname*{argmin}_{\theta} -\sum_{i=1}^{N} \left[\frac{-(y_{i} - h_{\theta}(x_{i}))^2}{2\sigma^2}\right] - \frac{\theta^2}{2\tau^2} $$ 

or, equivalently:

$$ \theta_{opt} = \operatorname*{argmin}_{\theta} -\frac{1}{N}\sum_{i=1}^{N} [-(y_{i} - h_{\theta}(x_{i}))^2] - \lambda \theta^2, $$ 

with $\lambda = \frac{\sigma^2}{N\tau^2}$.

Note that the first term is equal to the mean squared error, as obtained during MLE. The second term in this objective function can be seen as a regularizer that constrains the model-parameters $\theta$ to be as small as possible. This is in accordance with Occam's razor. The loss obtained through Maximum A-Posteriori coincides with a Maximum-Likelihood approach with regularization.

#### Connections between MLE and MAP

Assume a probability-distribution $p(x,\theta)$ as we have done in the MLE and MAP sections. Then, by the rules of probability:

$$ p(x,\theta) = p(x|\theta)p(\theta). $$ 

Since, in MLE, we consider a fixed parameter $\theta$, the term $p(\theta)$ is irrelevant, and the MLE objective:

$$ \theta_{opt} = \operatorname*{argmax}_{\theta} p(x|\theta) $$ 

is equivalent to:

$$ \theta_{opt} = \operatorname*{argmax}_{\theta} p(x,\theta), $$ 

under the MLE assumption that $p(\theta)$ is uniform, i.e., all $\theta$ are equally likely.

Furthermore,

$$ p(x,\theta) = p(x|\theta)p(\theta) = p(\theta|x)p(x). $$ 

Since $p(x)$ does not depend on $\theta$, the MAP objective:

$$ \theta_{opt} = \operatorname*{argmax}_{\theta} p(\theta|x) = \operatorname*{argmax}_{\theta} p(x|\theta)p(\theta) $$ 

is also equivalent to:

$$ \theta_{opt} = \operatorname*{argmax}_{\theta} p(x,\theta), $$ 

However, this time the parameters $\theta$ are also drawn from a non-uniform distribution $p(\theta)$ which is optimized.

Note that the loss function resulting via MAP corresponds to the loss function resulting via MLE with an additional term. This methodology, which introduces additional penalty terms to the MLE loss, is called model regularization. For example, one might penalize the square of the parameter magnitude so that the algorithm explores low-parameter spaces more. This can help to avoid overfitting by balancing small training errors with complex (and strongly biased) models.

Several methods exist, with common ones being Ridge regularization and Lasso regularization. Maximum likelihood estimation with Ridge regularization is equivalent to Maximum A-Posteriori estimation when the prior $p(\theta)$ is assumed to be Gaussian. Regularization uses prior assumptions to derive terms that penalize deviations. As such, it is not surprising that MLE with regression is equivalent to MAP with an imposed prior.

Penalizing model complexity can help with overfitting, and penalizing distribution differences (as done with the KL-divergence) favors certain parameterizations.

Summarizing, note that MLE and MAP are different, although similar, methodologies for maximizing the likelihood of $p(x,\theta)$ given some observed data $x$. The sections on MLE and MAP considered data availability in pairs $(x_i, y_i)$, but this is not a strict requirement on the methodologies.

### Expectation Maximization

We presented two popular ways of looking at data distributions and finding a corresponding model. Both methods aim towards maximizing $p(x, h(\theta)) = p(x,\theta)$ given the data $\\{x\\}$ and model $h(\theta)$ by minimizing a loss function $L(\theta)$.

The difficulty of the MLE and MAP depends strongly on the chosen model. For simple models, such as linear models $h(\theta) = \theta^T \ast x + b$, a closed-form solution for $\nabla L(\theta)=0$ is available, and the likelihood can be evaluated at the extrema to find the best. In fact, for linear models, the loss function is found to be convex, yielding one unique optimal point. For more complex models, the loss landscape might be non-convex, and closed-form solutions are often not available.

Furthermore, an additional complication is that $p(x,\theta)$ might involve a third, unmeasured variable $z$ so that:

$$ p(x,\theta) = \int_{z} p(x,z,\theta)dz. $$ 

If $p(x,\theta)$ can be evaluated, this indicates that the integral is tractable.

In more complicated cases, $p(x,\theta)$ cannot be evaluated, indicating that the integral is not tractable.

One can also write the log-likelihood as:

$$ \log(p(x,\theta)) = \log \int_{z} \left( \frac{p(x,z,\theta)}{q(z)} q(z) \right)dz. $$ 

By Jensen's inequality, this gives:

$$ \log(p(x,\theta)) \geq \int_{z} q(z) \log \left( \frac{p(x,z,\theta)}{q(z)} \right)dz, $$ 

We call this lower bound the ELBO (Evidence Lower Bound)

$$ \text{ELBO}(p(x,z,\theta), q(z)) = \mathbb{E}_{z \sim q(z)} \left[ \log \left( \frac{p(x,z,\theta)}{q(z)} \right) \right], $$ 

where $\mathbb{E}_{z \sim q(z)}$ denotes the expectation under $q(z)$, and the right-hand side is known as the evidence lower bound (ELBO).

Thus, while the integral itself might not be tractable, the lower bound, defined by the ELBO, is.

Furthermore, note:

$$ \log(p(x,\theta)) - \int q(z) \log \left( \frac{p(x,z,\theta)}{q(z)} \right) dz = D_{KL}(q(z) || p(z|x,\theta)), $$ 

where $D_{KL}$ denotes the Kullback-Leibler divergence. This gives the following relationships:

$$ \log(p(x,\theta)) \geq \text{ELBO}(p(x,z,\theta), q(z)), $$ 

and

$$ \log(p(x,\theta)) - \text{ELBO}(p(x,z,\theta), q(z)) = D_{KL}(q(z) || p(z|x,\theta)). $$ 

In the Expectation-Maximization (EM) algorithm, one starts by minimizing the gap defined by the KL divergence by finding the $q(z)$ closest to $p(z|x,\theta)$. This is equivalent to maximizing the ELBO.

If $p(z|x,\theta_t)$ can be evaluated, then one can directly set:

$$ q(z) = p(z|x,\theta_t). $$ 

This step is called the Expectation-step because it requires finding the expectation of $z$ given $x$ and $\theta$. Then one computes the log-likelohood under this distribution (called Q-function), which quantifies how well this $q(z)$ fits the data.

$$ Q(\theta, \theta_t) = \frac{1}{n} \sum_{i=1}^{n} \int_{z} p(z|x_i; \theta_t) \log p(x_i,z,\theta) dz = \mathbb{E}_{z \sim q(z)} \log p(x,z,\theta) . $$ 

In the next step, one aims to find $\theta_{t+1}$ that maximizes $Q(\theta, \theta_t)$. This step is called the Maximization-step because one maximizes the likelihood of the model given the data samples $x_i$ and the estimated $q(z)$.

Repeated iterations of the Expectation and Maximization step are expected to converge (altough not necessarily to global optima). The EM algorithm can also be used for MAP. The evaluation of $p(z|y,\theta_t)$ is not always possible. In the case of a VAE, the ELBO parameterizes the objective to be maximized, however, since $p(z|y,\theta_t)$ cannot be evaluated analytically, $q(z)$ is parameterized by a prior and the KL-divergence is minimized via gradient descent.

### From EM to Gaussian Mixture Models and VAE

Gaussian Mixture Models (GMMs) are a popular unsupervised learning model which can be optimized with the EM algorithm to find a mixture of \\( K \\) Gaussian's that describe the data optimally. We will see that the problem can be formulated as a distribution \\( p(x) \\) which involves a third, unknown variable so that $$ p(x) = \int p(x,z)dz. $$ This is the same setting as for Expectation Maximization, making the EM-algorithm a popular (but not the only) choice for Gaussian Mixture Models.   
  
Consider the probability that \\( x \\) is drawn from the $k$-th Gaussian as $$ p(x|\mu_k, \Sigma_k) = \frac{1}{\pi^{D/2}|\Sigma_k|^{0.5}} \exp(-\frac{1}{2}(x- \mu_k)\Sigma^{-1}(x- \mu_k)), $$ where \\( x, \mu_k \in \mathbb{R}^D\\) and \\( \Sigma_k\in \mathbb{R}^{DxD} \\). We can write $$ p(x) = \sum_{k=1}^{K} \pi_k p(x | \mu_k, \Sigma_k) = p(x | \mu, \Sigma, \pi) $$ where \\( \pi_k \\) is the probability that \\( x \\) belongs to the \\(k\\)-th Gaussian and \\( \sum_{i=1}^K \pi_i = 1 \\). Equation \ref{eq: GMM x-prob} corresponds to the probability of drawing \\( x \\) from the mixture of Gaussians, with each Gaussian being weighted by the probability which quantifies how likely it is that $x$ actually belongs to the \\(k\\)-th Gaussian generate. This belonging to one of the \\(K\\) Gaussian corresponds to an optimal soft assignment of data-points. We want to maximize $$ \log p(x) = \sum_{i=1}^{N} \log p(x_i | \mu, \Sigma, \pi ) = \sum_{i=1}^{N} \log \sum_{z_i=1}^{K} p(x_i|z_i, \mu, \Sigma)p(z_i|\pi ), $$ where \\( \mu = (\mu_1,...,\mu_K) \\), \\( \Sigma = (\Sigma_1,...,\Sigma_K) \\) and \\( \pi = (\pi,...,\pi_K) \\) Here we have introduced a hidden variable \\( z_i \\) which can take values between \\( 1\\) and \\(K\\) and whose meaning is the association of each \\(x_i\\) with one of the \\(K\\) clusters (i.e. \\(x\\) belongs to the \\(k\\)-th Gaussian). The sum within the log makes this difficult to optimize. It is due to the integration of the probability distribution of \\(z_i\\). Although it is not a-priori known which data-sample belongs to which Gaussian, note that $$ p(z_i = k| x_i) = \frac{ p(x_i|z_i = k)p(z_i = k) }{ \sum_{k'=1}^{K} p(x_i | z_i = k') p(z_i = k') } = \frac{ p(x_i|z_i = k)\pi_k }{ \sum_{k'=1}^{K} p(x_i | z_i = k') \pi_{k'}} , $$ can be evaluated given $\pi$ and equation \ref{eq: GMM gaussian}.   
  
Denote by \\( \theta \\) the parameter sets \\( \pi \\), \\( \mu \\) and \\( \Sigma \\). Let us consider again the likelihood \\( \log p_\theta(x)= \log \int p_\theta(x,z)dz \\) which we would like to maximize but can not directly due to the integral in the \\( \log \\) (in equation \ref{eq:GMM objective 0} this corresponds to the sum).   
  
For intractable problems of this kind, we often construct the \\( \text{ELBO} \\), as in equation \ref{eq:EM ELBO}, $$ \log p_\theta(x)= \log \int p_\theta(x,z)dz \geq \int q(z) \log \frac{p_\theta(x,z)}{q(z)}dz = \text{ELBO}(q(z), p_\theta(x,z)). $$ Equivalently, we also obtain the gap (see equation \ref{eq:EM gap}). $$ \log( p_\theta(x )) = \text{ELBO}( p_\theta(x,z), q(z) ) - D_{KL}(q(z),p_\theta(z|x)) dz $$ The EM-algorithm starts by minimizing the gap from equation \ref{eq:EM gap} by setting \\( q(z) = p_\theta(z|x) \\) at fixed \\( \theta = \theta_t \\). This is equivalent to maximizing the \\( \text{ELBO} \\) with respect to \\( z \\). Since $$ \gamma_k^i = p(z_i = k|x_i,\theta_t)$$ can be evaluated via equation \ref{eq: GMM z-prob} this can be done analytically. This step is called the Expectation-step. It calculates the expected values for \\( z_i \\). Each \\(x_i\\) belongs to the \\(k\\)-th Gaussians with probability \\( \gamma_k^i \\). This is similar to the assignment step in k-nearest neighbors.   
  
Once we know \\( p(z_i = k|x_i,\theta_t) \\) we can compute the log-likelihood (or Q-function from equation \ref{eq:EM Q}) via $$ Q(\theta, \theta_t) = \mathbb{E}_{z \sim p(z | x, \theta_t)} [\log(p(x,z|\theta) )] $$ $$ = \mathbb{E}_{z \sim q(z)} [\log (p(x,z|\theta) )] $$ $$ = \frac{1}{N} \sum_{i=1}^{N} \int_{z} p(z|x_i; \theta_t) \log p(x_i,z|\theta) dz . $$ $$ = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} p(z_i = k|x_i; \theta_t)\log p(x_i,z_i = k|\theta) . $$ $$ = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} \gamma_k^i \log (p(x_i|\mu_k, \Sigma_k)p(z_i = k|\theta)) . $$ $$ = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} \gamma_k^i \log p(x_i|\mu_k, \Sigma_k) + \gamma_k^i \log \pi_k . $$ This can be maximized analytically with $$ \mu_k = \frac{\sum_{i=1}^N \gamma_k^i x_i}{N_k} $$ $$ \Sigma_k = \frac{\sum_{i=1} \gamma_k^i (x_i -\mu_k)(x_i -\mu_k)^T }{N_k} $$ $$ \pi_k = \frac{N_k}{N} $$ where \\( N_k = \sum_{i=1}^N \gamma_k^i \\) and \\( \theta_t = (\mu_t, \Sigma_t) \\).   
  
The Maximization step, maximizes the log-likelihood (see equation \ref{eq:EM Q}) with respect to \\(\theta = (\mu, \Sigma)\\). This is equivalent to maximizing the $\text{ELBO}$ with respect to \\( \theta \\). As such, the Gaussian Mixture Model can be optimized with a the EM-algorithm since \\( p(z|x,\theta) \\) can be evaluated, allowing for an analytical minimization of \\( D_{KL}(q(z),p(z|x,\theta)) \\) by iteration, which, by equation \ref{eq:EM gap}, is equivalent to maximizing the \\( \text{ELBO}\\) with respect to \\( z\\). The unobserved variables \\(z\\) correspond to the assignment of data-points \\(x\\) to one of the \\(K\\)-clusters. It determines from which cluster the data-sample \\(x\\) is drawn. The Gaussian Mixture is in the same dimensional space as \\(x\\).   
  
In a variational auto-encoder setting, no expression for \\(p(z|x,\theta)\\) can be evaluated, so no analytical minimization of \ref{eq:EM gap} is possible by setting \\(q(z) = p(z|x,\theta)\\). Instead, one assumes a prior parametric description \\(p(z) = p(z|x,\theta)\\) and maximizes the \text{ELBO} with numerical gradient methods. Often this prior distribution is the isotropic Gaussian. However, for some data this can be too limiting. For example, the rotational symmetry of an isotropic Gaussian latent space is not suited for disentangled representations. Disentangles representation are understood as latent variables which correspond to single principal directions (not linear combinations thereof) in the data variability.   
  
The rest of the procedure is similar in the sense that the same type of objective function is maximized, however with gradient descent methods instead of explicit iterative expectation maximization. Furthermore, \\(z\\) can have arbitrary dimensionality. 

### Variational Auto-Encoders (VAE)

In a variational auto-encoder setting, no expression for $p(z|x,\theta)$ can be evaluated, so no analytical minimization of the gap equation is possible by setting $q(z) = p(z|x,\theta)$. Instead, one assumes a prior parametric description $q(z,\phi)$ and maximizes the ELBO with numerical gradient methods.

Note that:

$$ \text{ELBO}(p(x,z,\theta), q(z)) = \mathbb{E}_{z \sim q(z)} [ \log(p(x,z,\theta)) ] - \mathbb{E}_{z \sim q(z)} [\log(q(z))]. $$ 

This can be decomposed as:

$$ \text{ELBO}(p(x,z,\theta), q(z)) = \mathbb{E}_{z \sim q(z)} [ \log(p(x|z,\theta)) ] + \mathbb{E}_{z \sim q(z)} [\log(p(z,\theta))] - \mathbb{E}_{z \sim q(z)} [\log(q(z))]. $$ 

Or equivalently:

$$ = \mathbb{E}_{z \sim q(z)} [\log(p(x|z,\theta))] - D_{KL}(q(z) || p(z,\theta)), $$ 

which is the common expression for the objective employed in VAEs. The density $q(z)$ is parameterized by a neural net to $q(z,\phi)$. The posterior $p(z,\theta)$ is parameterized via prior assumptions, usually as an isometric Gaussian. Note, that during the derivation of the ELBO, any distribution that contains the hidden variable $z$ can be used. One might replace $q(z)$ with $q(z|x)$ or $q(z|x,\phi)$ without loss of generality.

While in standard Gaussian Mixture Models the term $\mathbb{E}_{z \sim q(z)} [ \log(p(x|z,\theta)) ]$ can be evaluated directly, for a VAE this is not the case, so instead Monte Carlo approximations are used. This term gives rise to the reconstruction error, which is essentially maximum likelihood estimation of $p(x|z,\theta)$ with respect to $\theta$. A more detailed description can be found in the blog dedicated to VAEs.

