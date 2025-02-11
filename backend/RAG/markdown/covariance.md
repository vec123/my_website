# On the Covariance

As an informative description of a gaussian data distribution, the covariance matrix is an interesting object. Here I try to give some intuitions on its geometry.

As a square matrix, the covariance forms a linear vector space. The special case of symmetric matrices is a linear vector subspace of the linear space of square matrices. A matrix is inherently a linear operator and cannot capture non-linear dependencies by itself.

A linear vector-space is a manifold. In this case, the manifold is flat, meaning it has no curvature and can be globally parameterized. A linear vector-space can be equipped with the Euclidean metric, which defines the distance between two points as the length of the straight-line. For a flat space, this corresponds to the geodesic distance associated with the metric. It coincides with the euclidean distance. A manifold is a topological space that locally resembles Euclidean space. A linear vector-space resembles Euclidean space locally and globally.

The set of symmetric matrices forms a linear vector-space. All open neighborhoods of a manifold are also a manifold. Each positive definite symmetric matrix has another positive definite symmetric matrix in its open neighborhood. Thus, the set of symmetric positive definite matrices forms a manifold. The same does not hold for semi-positive definite matrices.

We say that a covariance matrix induces a regular distribution if it is positive definite and a singular distribution if it is semi-positive definite. If it is semi-positive definite, then there exists a feature-dimension/direction along which no data variability occurs. If one measures data with a singular covariance, the singular dimension carries no information on the data-variability and can be omitted without loss of description accuracy. If one were to examine the likelihood of a model for describing data, the singular dimension has no influence on the likelihood.

This gives us a basis for understanding distributions as being located on manifolds (i.e., we can move smoothly from one distribution, defined by covariance and mean, to another), and a (vector-) directional sense in terms of model parameters/feature dimensions with influence on data-variability and model-likelihood.

We start by considering data distributions and extend this to model-distributions. The concepts are transferable. In one case, each sample corresponds to a data-sample (defined by the feature dimensions/parameters), in the other case, each sample corresponds to a model-sample (defined by the model dimension/parameters). This can be related to the difference between model parametrizations vs. model-free parametrizations of data-distributions.

### Some covariances and their distributions 

The simplest distribution is white noise, which in \\(2D\\) corresponds to a Gaussian with mean at the origin and covariance matrix:

$$ \Sigma = \begin{bmatrix} 1 & 0 \\\ 0 & 1 \end{bmatrix} $$ 

Note that it is regular. White noise can be transformed to another (linear) data-distribution by a linear transformation. Let:

$$ z \sim \mathcal{N}(0, I) $$ 

be sampled from white noise. Then

$$ x = A z + b $$ 

has covariance

$$ \Sigma_x = A I A^T = A A^T $$ 

and mean

$$ \mu_x = b. $$ 

The covariance matrix can be diagonalized. The eigendecomposition of \\(\Sigma\\) gives:

$$ \Sigma = U D U^T $$ 

where \\( U \\) is a matrix of eigenvectors, and \\( D \\) is a diagonal matrix of eigenvalues.

Set \\( A = U D^{1/2} \\), where \\( D^{1/2} \\) is the square root of the diagonal eigenvalue matrix. The transformation then becomes:

$$ x = U D^{1/2} z + b $$ 

giving the desired covariance. In the following, the distributions which can be described in this way are visualized.

### Linear Distributions

Let us visualize some distributions.

Figure 1: Linear Distributions

![White Noise](../images/covariance/data_images/white_noise.png) White Noise ![Transformation to the same mean, different covariance](../images/covariance/data_images/specific_covariance.png) Transformation to the same mean, different covariance

These two distributions gaussian distributions are linear transformations of one another. They are Gaussian and in many cases gaussian distributions are enough. We can obtain the conditional mean and variance by 

$$ \mathbb{E}(x_1|x_2 = k ) = \mu_1 + \Sigma_{x_1 x_2} \Sigma_{x_2}^{-1} (k - \mu_2) $$ $$ \mathbb{Var}(x_1|x_2 = k ) = \Sigma_{x_1} - \Sigma_{x_1 x_2} \Sigma_{x_2}^{-1} \Sigma_{x_2 x_1} $$ 

Note, that if the covariance is zero, the conditional mean is the same as the unconditional mean, meaning \\( x_1 \\) carries no information on \\(x_2 \\) and vice versa. This conditioning can be interpreted as a projection of the distribution onto the line \\( x_2 = k \\). If the correlation is high (in essence \\( \Sigma_{x_1 x_2} \\) is high), then knwoing \\(x_2\\) gives a lot of information on \\(x_1\\). If we consider \\( \mu_1 = \mu_2 = 0 \\), then, high correlation implies \\( x_1 = x_2 \\). The random variable \\(x_1\\) might correspond to a non-linear transformation of another random variable \\(x_3\\), i.e. \\(x_1 = \Phi(x_3) = sin(x_3) \\), or even multiples thereof. This is non-linear transformation is powerful as will be seen in the section on gaussian processes. 

A way of representing the distributions along lines is by the mentioned transformation of white noise so that the variance in the desired direction is very high, potentially infinite. Thus, linear distributions (as I call them) are equivalent to a transformation of white noise.

### Regular and Singular Covariance

Figures below show distributions that can be obtained by a rotation. Both can be obtained via affine transformations of white noise. One direction has a very high variance. This variance would potentially be infinite, but in reality, most real-world data is finite.

An important example is the distribution where the variance normal to the line is zero. The data is two-dimensional, yet only one dimension varies. The covariance matrix does not have full rank and corresponds to a singular distribution. The data can be represented just as well in a single dimension without loss of accuracy since there is perfect correlation between the two random variables. 

Figure 3: Singular Covariance

![Regular covariance with linear mean](../images/covariance/data_images/transformed_uniform_line \(regular distribution\).png) Regular covariance ![Singular covariance](../images/covariance/data_images/transformed_uniform_line \(singular distribution\).png) Singular covariance

If we measure data with a singular covariance, some directions are degenerate (have zero variance). This is an indication that those directions provide no information, and these dimensions can be omitted without losing description accuracy. If the variables correspond to model parameters, then the singular directions have no influence on the likelihood of the model. Two models are indistingushable in their ability to describe the data with respect to changes in the singular direction.   
  
The non-invertability of singular distributions is a problem for some algorithms. It is common practice to add a small constant to the diagonal of the covariance matrix to make it invertible. This term reappears in many algorithms as we will see. 

### Mahalanobis Distance / covariance induces a geometry

Within the linear space that contains our distribution, one can define the Mahalanobis distance as:

$$ d_M(x, \mu) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}. $$ 

In directions where the data has high variance, the Mahalanobis distance will be smaller. Conversely, in directions of low variance, the distance will be larger. This accounts for the fact that outliers in low-variance directions are more significant than in high-variance directions. In light of this, the covariance might be interpreted as a deformation of the linear space, which stretches some directions (those with low variance) and compresses others (those with high variance). While it can introduce diagonal relationships, no curvature or torsion can be created by a constant covariance matric. If \\(\Sigma = I\\) (as for white noise), the Mahalanobis distance coincides with the Euclidean distance, and \\(\Sigma = I\\) might be interpreted as uniformly "stretched" space. The elements of \\(\Sigma^{-1}\\) correct for the stretching and compression of space by accounting for the effect of variance and covariance.

### Distributions with nonlinear mean

The distributions we considered in the figures above have a constant mean. These distributions can be obtained from white noise by linear transformations. All regular distributions can be transformed back to white noise, whereas the two singular distributions cannot.

In some cases, \\(\mu(x,y)\\) is nonlinear, but the covariance is constant.

Figure 4: Sinusoidal Mean and Constant Variance

![Sinusoidal Mean and constant variance](../images/covariance/data_images/sine_const_var.png) Sinusoidal Mean and constant variance ![Circular Mean and constant variance](../images/covariance/data_images/circle_const_Var \(2\).png) Circular Mean and constant variance

### Distributions with varying covariance

Another interesting case is when the data distribution is not described by a constant covariance matrix. Instead, the covariance is a function of the parameters. We can distinguish between cases where this function is linear vs. non-linear. Since this relationship is not a constant, it requires a parameterized covariance matrix of the form:

$$ \Sigma(x,y) = \begin{bmatrix} f_{x,x}(x,y) & f_{x,y}(x,y) \\\ f_{y,x}(x,y) & f_{y,y}(x,y) \end{bmatrix}. $$ 

In most cases, we will consider only continuous functions for the elements of \\(\Sigma(x,y)\\).

Figure 5: Varying Covariance

![Discontinuity in covariance](../images/covariance/data_images/two_variances.png) Discontinuity in covariance ![Smoothly varying covariance \(linear\)](../images/covariance/data_images/continous_varying_variances.png) Smoothly varying covariance (linear) ![Smoothly varying covariance \(nonlinear\)](../images/covariance/data_images/continous_varying_variance_sine.png) Smoothly varying covariance (nonlinear)

Figure 5a displays a discontinuous variation in covariance. While this can still be stitched together by a fiber bundle, it is not a smooth, differentiable manifold. Discontinuous cases can be considered approximately continuous with some loss of accuracy, and we will ignore discontinuous cases in the following. Figures 5b and 5c display distributions with covariance that varies in linear and non-linear manners, respectively. In both cases, the mean follows a linear parametrization.

### Complex Covariance and Mean Structures

More complex cases exist when both the mean \\(\mu(x,y)\\) and covariance \\(\Sigma(x,y)\\) are nonlinear. Examples are displayed in Figures below:

Figure 6: Non-linearly Varying Covariance on a Circle

![Nonlinear distribution on a circle](../images/covariance/data_images/circle_mean_sine_variance.png) Nonlinear distribution on a circle ![Nonlinear distribution on a parabola](../images/covariance/data_images/parabola_mean_sine_variance.png) Nonlinear distribution on a parabola ![Nonlinear covariance distribution with sine-wave mean](../images/covariance/data_images/Gaussian_Process_Regression_data.png) Nonlinear covariance distribution with sine-wave mean

These distributions cannot easily be modeled by a simple covariance matrix and a mean. However, locally (if the transitions are continuous and the neighborhood is small enough), they are well behaved and adhere to the linear description. If the distribution is regular in all points, then smooth transitions from one neighborhood to another can be defined by considering a Riemannian geometry on which the data-distribution lives. Manifold learning exploits this local linearity by determining data-samples that are in linear neighborhoods of each other. Then, the data-distribution of these samples can be assumed to be linear. It is not always trivial to determine which data-samples are close to each other by only looking at the dimensional data representation (e.g. pixel values). However, a common assumption is that most real world data distributions are located on some manifold i.e. they have a mean parameterization and covariance parameterization which might be complex and nonlinear, but can be locally approximated by a linear parameterization. 

### Fisher Information Matrix

An interesting covariance is the Fisher Information Matrix. In machine learning, one aims to maximize a likelihood \\( p(x, \theta) \\) or log-likelihood \\( \log(p(x, \theta)) \\) by finding the optimal parameters \\( \theta \\).

The gradient of the likelihood with respect to the parameters is defined as:

$$ s(x, \theta) = \frac{\partial \log (p(x, \theta))}{\partial \theta} $$ 

and is called the score of the model. It can be shown that, under regularity conditions, the score of the true model \\( \tilde{\theta} \\) is zero. In general, the score is indicative of the model-parameter sensitivity. The notion of a score-based gradient has been exploited successfully in generative modeling, under the name of score matching.

For each \\( \theta \\), it is possible to define the Fisher Information Matrix as the covariance of the score:

$$ \mathbf{I}(\theta) = \mathbb{E}\left[ \left( \frac{\partial \log p(x, \theta)}{\partial \theta} \right) \left( \frac{\partial \log p(x, \theta)}{\partial \theta} \right)^T \right] $$ 

which is also expressed as:

$$ = -\mathbb{E}\left[ \frac{\partial^2 \log p(x|\theta)}{\partial \theta \partial \theta^T} \right] = \int_{x} \frac{\partial}{\partial \theta} \log(p(x, \theta)^2) p(x, \theta) \, dx. $$ 

For a linear system \\( y= w \theta + \epsilon \\), the Fisher Information matrix is the inverse of the covariance matrix of the data, i.e 

$$ \mathbf{I}(\theta)^{-1} = \frac{1}{N}X^TX. $$ 

This is the linear Gram matrix. It corresponds to a linear kernel. Similarily, for other kernels 

$$ \mathbf{I}(\theta)^{-1} = \frac{1}{N} K(X,X'). $$ 

Given the previous examples, this should create some intuition of the underlying geometry defined by this matrix. Note that this covariance matrix varies smoothly, depending on the location. Considering the distance induced by a covariance matrix, it can be shown that \\( \mathbf{I}(\theta) \\) is a Riemannian metric, providing the tangent space of \\( \theta \\) with an inner product and defining distances between distributions. The geodesic distance is the path-integral of distances induced in each point by the local covariance matrix. The idea is similar to the Mahalanobis distance, but, due to the interpretation in terms of model parameter sensitivity (as opposed to data variability), we do not use the inverse. High sensitivity corresponds to larger distances (stretched space) while high variability corresponds to smaller distances (compressed space). 

### Geometries of a neural net

The geometry of neural networks, auch as CNNs and Bolzman machines is focus of recent research. [Geometric Inductive Biases of Deep Networks](https://arxiv.org/abs/2410.12025) introduces quantities which study the relation between input-space covariance and model covariance. They introduce the quantities 

$$ G_t(x) = \mathbb{E}_{\theta \sim T_t} \left[ \nabla_x f_\theta(x) \nabla_x f_\theta(x)^\top \right] $$ 

and 

$$ \Delta t \, F(x) = \mathbb{E}_{\theta \sim T_t} \left[ \nabla^2_{x,\theta} f_\theta(x) \, \dot{\theta} \, \nabla_x f_\theta(x)^\top \right] + \mathbb{E}_{\theta \sim T_t} \left[ \nabla^2_{x,\theta} f_\theta(x) \, \dot{\theta} \, \nabla_x f_\theta(x)^\top \right]^\top $$ 

called average geometry and average geometry evolution respectively. The quantity \\( T_t \\) represents the distribution of the trajectoreis of \\( \theta \\) during training. It is subject to stoachstisticity by different initalizations, mini-batching and randomness in the training procedure. The data-covariance is defined as the linear covariance 

$$ S = \sum_{i=1}^{N}x_ix_i^T = XX^T $$ 

It is shown that \\( \Delta t \, F(x) \rightarrow G_0(x)SG_0(x) \\) as \\( n \rightarrow \infty \\) where \\(n \\) corresponds to model width. In essence, the geometry induced by the covariance is mapped via the average geometry of the model at initalization (similar to a rotation and rescaling of a distribution explained above). For multiplayer perceptrons \\(G_t(x) = I \\). The same does not hold for convolutional networks. If \\(G(x) \\) is structured (i.e. has singularities), then some directions (those with eigenvalue zero) are uninformative and the model is invariant to these. For a convolutional neural net these directions might correspond to translations of the data since a convolutional classifier should be invariant to translation. However, a convolutional object detector should be equivariant, not invariant, not translations. Models fails to generalize, when decision boundary normals ( \\( \mathbb{E}_{\theta} \Delta_x f_\theta(x) \\) ) are aligned with invariant directions. The resuls show that the eigenspace generated by the average geometry of the model determines which features can be learned. Feature-directions which are not in the eigenspace are not learned. As such, a data representation in terms of \\(v \\) and \v_A \in GSG \\) or a data representation in terms of \\(v \\) and \\(v_B \notin GSG \\) makes a difference with respect to the model performance.   
  
Experiments show a correlation of linear data covariance \\(S \\) and transformed covariance \\(GSG\\). These results support the conjecture that the average geometry of the model at initialization \\(G_z(x)\\) converges to \\(GSG\\) for small \\(t \rightarrow 0 \\) . Data-samples \\(x \\) with high correlation to the initial average geometry seem to be more impactful on model performance. These correspond to linear combinations of eigenvectors with large eigenvalues. Feature directions with low correlation to the inital geometry (eigenvectors with small eigenvalues) seem to have less impact on model performance.   
  
This line of research leads towards an understanding of geometrical quantities in neural networks which formalize well-known properties (such as invariance and equivariance) by eigenspace analysis of the average model geometry.   
  
Note, that the eigenvectors and eigenvalues determine the importance of directions for the distribution. Singularity indicates uninformative directions and orthogonal directions are invariant to the distribution.   
  
Note, that these results allow for an interpretation in terms of transfer functions and their power spectral densities. 

### Geodesic Distance in Parameter Space

Consider an infinitesimal displacement \\( \frac{d\theta }{dt} = \lim_{\Delta \rightarrow 0} \theta + \Delta \\) and a path \\( \gamma(t) \\) related to the displacement by:

$$ d \theta = \frac{d \gamma(t)}{dt} \, dt. $$ 

The infinitesimal distance from \\( \theta \\) to \\( d\theta \\) is:

$$ ds^2 = d\theta^T I(\theta) d\theta = \frac{d \gamma(t)}{dt} I(\theta) \frac{d \gamma(t)}{dt} \, dt. $$ 

This is the infinitesimal squared distance in Riemannian geometry with the Fisher information matrix as the metric tensor. When \\( I(\theta) \\) is large, indicating a high likelihood sensitivity of a parameter, the distance in that direction is also large. Moving in directions with high sensitivity (i.e., corresponding to high information content) traverses a larger distance, while moving in directions with low sensitivity does not get us far. 

The distance between \\( \theta_1 \\) and \\( \theta_2 \\) becomes:

$$ d_G(\theta_1, \theta_2) = \int_0^1 \sqrt{\frac{d\gamma(t)}{dt}^T I(\gamma(t)) \frac{d\gamma(t)}{dt}} \, dt = \int_0^1 ds, $$ 

where \\( \gamma(0) = \theta_1 \\) and \\( \gamma(1) = \theta_2 \\).

This geodesic distance allows for the quantification of distribution similarity. Distributions (parameterized by \\( \theta \\)) are close if the information content is similar and distant otherwise. The geodesic distance is symmetric. For infinitesimal step sizes it coincides with the KL-divergence

### The covariance matrix induces a geometry in a cartesian space

These examples serve as visualizations to help build an intuition about how a covariance matrix induces a geometry in linear space. Given a covariance matrix, we might think about a stretched/compressed euclidean space. We have examined how this can be seen as a transformation of Gaussian noise and explored more complex cases, where the data-distribution displays non-linear mean-functions and even non-linear covariance functions. These complex cases, when continuous, can be seen as smooth transitions between linear cases, although asymmetries must be considered. For singular distributions, problems arise due to the existence of degenerate directions. A singular covariance matrix is related to perfect correlation. Additionally, we have seen that the Mahalanobis distance is a measure of distance in the deformed space. As such, learning a covariance can be thought of as learning distances between data-points in a deformed space. These distances are symmetric (since the covariance is symmetric) and can be used to define a Riemannian geometry on a manifold. Algorithms, such as self-attention in transformers, can learn asymmetric distances between data-points. This asymmetry allows for directional distances which can capture more complex relationships between data-points, such as dependence, causality and time. Another an important concept of asymmetric "distances" is called divergence between distributions. We mentioned the KL-divergence as an approximation of the Fisher-Information matrix. 

### PCA

PCA is a popular and well understood algorithm which allows for modelling and dimensionality reduction of linear distributions by considering the eigenspace of the covariance matrix. As such, it is a good introduction into the covariance matrix, its meaning, eigenspace and possible manipulations. The intuitions developed here transfer well to more complex models, such as kernel PCA and Gaussian processes, by considering a nonlinear transformation \\( \Phi(z) \neq Wz \\). If the transformation is linear, equivalence to PCA can be shown. Principal Component Analysis is a popular linear algorithm for systems of the form 

$$ x = f(z) = Wz $$ 

or, in the probabilistic PCA setting 

$$ x = f(z) + \epsilon = Wz + \sigma^2\epsilon, $$ 

where \\( \epsilon \sim \mathcal{N}(0, I)\\) and \\(z \in \mathcal{N}(0,I)\\) so that 

$$ p(x|z) = \mathcal{N}(Wz,WW^T) \text{ and } p(x)=\mathcal{N}(Wz,WW^T) $$ 

in the former case and 

$$ p(x|z) = \mathcal{N}(Wz,WW^T) \text{ and } p(x) = \mathcal{N}(Wz,WW^T + \sigma^2 I) $$ 

in the latter case. Note that \\( \sigma^2 I \\) is an addition of noise in the diagonal direction. It avoids singularity of \\( WW^T + \sigma^2 I\\) and make the matrix invertible by adding noise in the space orthogonal to \\( \text{Im}(W)\\). Consider white Gaussian noise \\( z \sim \mathcal{N}(0,I) \\) and a mean-centered data-distribution \\(p(x) = \mathcal{N}(0,\Sigma)\\) . Let \\(x,z \in \mathcal{R}^{n}\\) . Given the data-set matrix \\(X = (x_1,....,x_N) \in \mathcal{R}^{N \times n}\\) , we can construct the singular value decomposition 

$$ X = U \Delta V^T, $$ 

with \\(U \in \mathcal{R}^{N} \\), \\( \Delta \in \mathcal{R}^{N \times N} \\) and \\(V \in \mathcal{R}^{n} \\) (in the complex case we need to consider the conjugate transpose). We can construct the linear covariance matrices 

$$ XX^T = V\Delta^2V^T = VDV^T \in \mathcal{R}^{n\times n} $$ 

and 

$$ X^TX = U\Delta^2U^T = UDU^T \in \mathcal{R}^{N\times N}. $$ 

In the real case \\( U \\) and \\( V \\) are guaranteed to be real orthogonal matrices so that \\( \text{rank}(X^TX ) = \text{rank}(XX^T) \\). Both covariance matrices have different dimensionality but the same non-zero eigenvalues and equal rank. This equivalent nature of the two covariance matrices and their corresponding distributions has deep consequences in machine learning (especially for Gaussian Processes) and is referred to as the data-space (or image space) and feature-space duality. Note that both matrices span the same subspace, but have different dimensionality. This enables working in an infinite dimensional feature space by considering a \\(N\\) dimensional data space. But don't worry. For now everything remains finite. Consider the question, which linear transformation of \\( z \sim \mathcal{N}(0,I) \\) yields the distribution \\( p(x) = \mathcal{N}(0,XX^T) \\). The answer is 

$$ x = Wz = U D^{1/2}z $$ 

as becomes obvious from the linear transformations of white noise we showed in the beginning. Note that \\( U \\) is the matrix of eigenvectors of the linear covariance matrix \\( XX^T \\) and that \\( D^{1/2} \\) is the square-root of its eigenvalue matrix. Thus the linear transformation relating the data-distribution to white noise is defined by eigen-directions and eigenvalues of the covariance matrix. The eigenvalues are called the principal components, the eigenvectors are called principal directions.   
  
The main application of PCA is in dimensionality reduction. Until now a dimensionality preserving linear map was derived. For dimensionality reduction consider \\( x \in \mathbb{R}^n \\) , \\( z \in \mathbb{R}^d \\) and consequently \\( W \in \mathbb{R}^{n \times d} \\) with \\( d \leq n \\). Consider the distribution \\( \mathcal{N}(0, WW^T) \\) and note that \\( WW^T \\) has rank \\( d \\) but dimension \\( n \\). We have a singular distributions (some eigenvalues are zero). Singularity indicates the existence of directions in which no variability occurs. Given a dataset \\( X \\), we can perform the PCA decomposition and yield equation \ref{eq: pca solution}. By keeping only the \\( d \\) biggest eigenvectors ( \\( D_{-} \in \mathcal{R}^{d \times d} \\) ) and corresponding eigenvalues (\\( U_{-} \in \mathcal{R}^{n \times d}\\) ) , a dimensionality reduction is performed. Note, that then 

$$W = U_{-}D_{-}^{1/2}$$ 

$$ \Sigma = WW^T = U_{-}D_{-}U^T \in \mathbb{R}^{n \times n}$$ 

is a singular matrix with rank \\(d\\). Along the directions corresponding to the omitted eigenvectors no variability occurs. Since singular distributions can be hard to work with (e.g. non-invertible), a common practice is adding noise. In essence, we consider 

$$ x = Wz + \epsilon $$ 

with \\( W \in \mathbb{R}^{n\times d} \\), \\( z \in \mathbb{R}^{d} \\) and \\( \epsilon \sim \mathcal{N}(0, \sigma^2I)\\). From this equation, which corresponds to the probabilistic PCA setting, we obtain 

$$ p(x|z) \sim \mathcal{N}(Wz, WW^T) $$ 

leading to 

$$ p(x) \sim \mathcal{N}(Wz, \Sigma) $$ 

with with \\( \Sigma = WW^T + \sigma^2 I \approx XX^T\\) .Consider the mean centered version \\( p(x) \sim \mathcal{N}(0, \Sigma) \\) and perform MLE, 

$$ \Sigma_{opt} = \operatorname{argmax}_{\Sigma} \prod_{i=1}^{N} \log( \frac{1}{\sqrt{(2\pi)^n |\Sigma|}} \exp \frac{- x_i \Sigma^{-1} x_i^T}{2} ) $$ 

$$ = \operatorname{argmin}_{\Sigma} \sum{i=1}^{N} \log( \frac{1}{\sqrt{(2\pi)^n |\Sigma|}}) + \frac{- x_i \Sigma^{-1} x_i^T}{2} $$ 

$$ = \operatorname{argmin}_{\Sigma} -\frac{N}{2} \log((2\pi)^n) - \frac{N}{2}\log(|\Sigma|) - \frac{1}{2} \sum_{i=1}^N x_i \Sigma^{-1} x_i^T $$ 

$$ = \operatorname{argmin}_{\Sigma}- \frac{N}{2}\log(|\Sigma|) - \frac{1}{2} \sum_{i=1}^N x_i \Sigma^{-1} x_i^T $$ 

$$ = \operatorname{argmin}_{\Sigma} -\frac{N}{2}\log(|\Sigma|) -\frac{1}{2} \sum_{i=1}^N x_i \Sigma^{-1} x_i^T. $$ 

To find the extremum onsider 

$$ \frac{\partial}{\partial \Sigma} -\frac{N}{2}\log(|\Sigma|) = -\frac{N}{2}\Sigma^{-1} $$ and 

$$ \frac{\partial}{\partial \Sigma} - \frac{1}{2} \sum_{i=1}^N x_i\Sigma^{-1} x_i^T = + \frac{1}{2} \Sigma^{-1} \sum_{i=1}^N x_ix_i^T \Sigma^{-1} . $$ so that the extremum is reached if 

$$ \frac{N}{2} \Sigma^{-1} + \frac{1}{2} \Sigma^{-1} \sum_{i=1}^N x_i^Tx_i \Sigma^{-1}= 0. $$ Consider \\( \sum_{i=1}^N x_ix_i^T = XX^T = UDU^T \\) (according to the singular value decomposition) so that 

$$ \frac{N}{2} \Sigma^{-1} + \frac{N}{2} \Sigma^{-1} UDU^T \Sigma^{-1} = 0. $$ 

and thus 

$$ \frac{N}{2} + \frac{N}{2} \Sigma^{-1} UDU^T = 0 \rightarrow UDU^T = \Sigma $$ 

We only want keep the $d$ biggest eigenvalues and eigen-directions so that 

$$ WW^T + \sigma^2 I \approx \Sigma = UDU^T \approx U_{-}D_{-}U_{-}^T + \sigma^2 I. $$ 

$$ Let, \\( W = U_{-}(D_{-} - \sigma^2 I)^{1/2}\\) so that 

$$ WW^T = U_{-}D_{-}U_{-}^T + \sigma^2I \approx \Sigma. $$ \\(D_{-}\\) should contain the \\(d\\) biggest eigenvalues and \\( \sigma = \frac{1}{n-d}\sum_{j=d+1}^{n} \lambda_j\\). Thus, the dimensionality reduction results in 

$$ x = U_{-}(D_{-} - \sigma^2 I)^{1/2}z + \sigma^2 \epsilon. $$ 

If we omit \\( \epsilon \\), this corresponds to a linear projection of the \\( n \\) dimensional data \\( X \\) onto a subspace of \\( d \\) dimension.   
  
We have shown how PCA can be considered a linear transformation of white noise to a Gaussian distribution. Thus given a data-distribution we can find the transformation \\( W \\) which produces our data given white noise as an input. We have also extended this point of view to a dimensionality reduction setting. Any linear transformation of \\( d \\)-dimensional white noise creates a \\( d\\) dimensional subspace in the \\( n \\)-dimensional feature space. Considering noise contributions in its orthogonal directions solves the this issue and defines a regular distribution in the whole space. We have shown how solving for the optimal transformation and noise parameters traces back to the eigen decomposition of the linear covariance matrix defined by \\( X \\). Essentially, one keeps the \\( d \\) biggest directions and eigenvalues (which have the biggest contribution to the covariance matrix) and uses the others to construct the noise parameters. For this we have performed maximum likelihood estimation with a parameterized \\( \Sigma \\) matrix.   
  
We have also hinted at an equivalence between the matrices \\( XX^T \\)and \\( X^TX \\). While living in different spaces, the image of both matrices has equal dimensions. In some sense, both distributions are equivalent.   
  
PCA is a very well understood and popular method. Being completely linear allows for closed form solutions, good interpretability and analysis. Linearity is at the same time its strength and main limitation. As a linear transformation of white-noise we can not expect to create a model for nonlinear distributions. It has been shown [here](https://arxiv.org/abs/2102.06822) that the variational auto-encoder with a diagonally parameterized covariance matrix pursues the principal directions of PCA. Note however, that the matrix obtained by PCA can be multiplied with any invertible transformation (such as a rotation) and still yield the same distribution. This rotation means a representation in a different basis which does not correspond to the eigendirections and thus not to disentangled varability directions. It is possible due to the rotationally invariant nature of the isotropic Gaussian distribution from which \\(z \\) is drawn. Equally, for a VAE, a rotationally invariant prior hinders learning of disentangled representations. Nonetheless, given the results states in that paper, we might consider PCA as the linear alternative of VAEs and VAEs as its nonlinear counterpart. If our data-distribution is indeed linear, both yield the same results (with disentangled latent representations or not). If our data-distribution is non-linear, PCA will fail to capture it while the VAE still might. 

### Gaussian Processes and function space

Gaussian Processes parameterize a covariance matrix via a kernel-covariance. Starting from PCA they become much easier to understand, since they can be viewed as a form of PCA in potentially infinite feature-dimensions, created by applying a nonlinear mapping (kernel) to the feature-space. For a solid introduction i want to recommend the [Gaussian Processes for Dummies]() blog-post. It has some great visualization as to how a high-dimensional covariance matrix leads to a non-linear regression problem. While it does not relate to PCA it also offers a great prespective you should not miss out on. The non-linear function is implicit in the kernel-covariance parameterization and does not need to be explicitly defined. This becomes clearer by relation to the reproducting kernel Hilbert space (RKHS), where the basis functions explicitely define the kernel as their inner product. Since a function is informally thought of as infinite dimensional vector an infinite-dimensional feature-space is also called function space. I also like to think about euclidean space with directions representing functions and non-linear relationships. For example instead of saying that \\(x /) is correlated to \\(y \\) we can say that the function \\(f(x)\\) is correlated to the function \\(g(y)\\). In this euclidean space, the covariance matrix defines a linear distribution and induces a Mahanalobis distance, however, due to the possibility of nonlinear basis directions, the expressivity is augmented. Since a covariance matrix in infinite-dimensional feature-spaces is not computable, gaussian processes exploit the data-space and feature-space duality and perform the covariance parameterization in data-space. The parameters are updated via maximum likelihood estimation. Given an optimal covariance matrix, a gaussian distribution is defined and, by its conditional mean its conditional covariance, inference can be done with uncertainty information. Note, that the model 

$$ x = Wz + \epsilon $$ 

leads to 

$$p(x) = \int p(x|z)p(z) dz$$ 

since \\( z \\)is white noise. If \\( p(z) \\) is Gaussian, then so are \\( p(x|z) \\) and \\( p(x) \\). Remember that \\( XX^T \\) and \\( X^TX \\) define equivalent distributions in different dimensions. Consider now an alternative model. 

$$ x = W\Phi(z) + \epsilon, $$ 

where \\( \Phi(z) \\) is a kernel (some map into lower or higher, potentially infinite dimensions), \\( z \\) deterministic and \\( w \sim \mathcal{N}(0,I) \\) . This leads to 

$$p(x) = \int p(x|w)p(w) dw.$$ 

If \\(p(w)\\) is Gaussian, then so are \\(p(x|w)\\) and \\(p(x)\\). Assume one is interested in 

$$p(x) \sim \mathcal{N}(W\Phi(z), \Sigma), $$ where 

$$\Sigma = WW^T + \sigma^2 I$$ 

and has available a set of \\( N\\) data-samples \\( (x_i,z_i)\\). This direct consideration of \\( p(x) \\) as a Gaussian distribution without concretizing \\( \Phi(x)\\) or \\( W\\) is what is referred to as an integration over the space of all functions (informally: infinite dimensional vectors are functions). However, the covariance matrix still needs to be parametrized, which limits this space of all functions to more concrete, but still fairly broad types. In PCA, the prior was over \\( z\\) and we would just consider \\( \Sigma \approx XX^T \in \mathcal{R}^{n \times n}\\). We can do this because 

$$ \text{Var}(x) = \sum_{i=1}^N \mathbb{E}(Wz_i - \mu)\mathbb{E}(Wz_i - \mu)^T = \sum_{i=1}^N \mathbb{E}(Wz_i)\mathbb{E}(Wz_i )^T = W \sum_{i=1}^N\mathbb{E}(z_i)\mathbb{E}(z_i)^T W^T = W W^T. $$ 

However, for equation \ref{eq:pca to gp} we can not do this since the map \\( \Phi(z) \\) is not known (not even its dimensionality). We have 

$$ \text{Var}(x) = \sum_{i=1}^N\mathbb{E}( W\Phi(z_i))\mathbb{E}(W\Phi(z_i))^T = \sum_{i=1}^N \Phi(z_i)\mathbb{E}(W)\mathbb{E}(W)^T\Phi(z_i)^T $$ 

A way of alleviating the issue of an unknown covariance matrix with unspecified feature dimensions, is by considering the covariance \\( \Sigma = X^TX \in \mathcal{R}^{N \times N} \\) which is limited in dimensionality only by the amount of data-samples, but defines a distribution in the feature-space dimensions, as illustrated on the linear case. One chooses a non-linear parameterization of the kernel covariance matrix \\( \Sigma = K(Z,Z) \\), such as the radial basis function kernel. This imposes a form on the corresponding functions \\( \Phi(z) \\). The maximum likelihood estimation from equation \ref{eq: MLE mutivariate covar} with respect to the kernel parameters can be performed for non-linearly parameterized matrices. For some kernel covariances it can the solved analytically. This is the concept behind Gaussian processes. Given the covariance matrix which maximizes our likelihood 

$$p(x) = \int p(x|w)p(w) dw = \mathcal{N}(0, K),$$ 

we can consider 

$$ \begin{bmatrix} p(x) \\\ p(x^*) \end{bmatrix} = \mathcal{N}(\begin{bmatrix} \mu_X, \mu_x^*\end{bmatrix} ,\begin{bmatrix} K, K(Z,z^*) \\\ K(z^*,Z), k(z^*,z^*)\end{bmatrix}). $$ 

The conditional mean and variance of a multi-variate Gaussian is 

$$ \label{eq:conditional mean} \mathbb{E}[x^* | X ] = \mu_X + K(z^*,Z) K^{-1}(X) $$ and 

$$ \label{eq:conditional variance} \text{Var}(x^* | X) = k(z^*,z^*) - K(z^*,Z) K^{-1} K(Z,z^*). $$ 

These quantities allow for inference with associated uncertainty information. Usually we consider mean centered data, so that \\( \mu_X = 0 \\). Note, that in this framework, we assume an available data-set of pairs \\( x_i,z_i \\) and \\( z_i \\) is not associated to a prior distribution.   
  
In an alternative setting, the data consists only of \\(x_i \\) and the maximum likelihood is optimized also with respect to \\(z_i \\). These types of models are called Gaussian process latent variable models (GPLVM). The dimension of \\( z \\) is a hyperparameter, making this model suitable for dimensionality reduction. Similar to the auto-encoder, the latent-space is unrestricted, so that smoothness and continuity of the latent distribution is not necessarily given. Similar to the variational auto-encoder, a natural extension to bayesian variational gaussian process latent variable models exists. However, within the community, Gaussian Processes are used mainly for regression tasks.   
  
Interesting extensions of gaussian processes are gaussian process latent variable models (GPLVM) and variational GPLVM. GPLVM is a generative model which allows for dimensionality reduction and data generation, similar to an auto-encoder. Variational GPLVM imposes a variational prior onto the latent variables, similar to a variational auto-encoder.   
  
Some approaches connect gaussian processes with auto-encoders by 

### Attention

The attention mechanism, popularized by the transformer model, can be understood as a learned covariance parameterization. The mapping of pairs to a scalar product (to a reproducing hilbert space) corresponds to learning the covariance. This scalar product can also be understood as inducing a geometry in a reproducing hilbert space (infinite dimensional vector space), with zero corresponding to orthogonality and one corresponding to perfect correlation. Note, that the covariance matrix stretches and compresses a linear space allowing for learning a distortion. Attention moves close points closer together (higher covariance) and far points further away (lower covariance). Compare this intuition to the Mahanalobis distance, which is a distance measure in a stretched and compressed space.   
  
To explain the attention mechanism start with a sequence of embeddings, or vectors, \\( \\{e_1, ..., e_n \\} \\), each with dimension \\(d\\). Assume each vector also contains information on its position \\( i \in \\{ 1, ... , n \\} \\). An attention block aims to compute an updated version of the embedding so that each contains information of its context. Each input-embedding attends to all other input-embedding. This enables the encoding of global context in the single embedding. Global context is just a learned aggregation on the whole data. This contrast many other methods, where the aggregation is restricted to neighbourhoods, such as convolutional filters and RNN's. Considering the whole data makes attention powerful but computationally expensive. Because in an attention module all embeddings attend to all others, the context window is global instead of sequential. Thus, the embedding for pronouns can learn to pay attention to the embedding for the sentence subject, while the proposition "to" learns to pay attention to the connected verb "walking" and noun "school". In addition, positional encodings are used, providing information on the embedding location. In some languages the noun is always the first word which can be recognized with positional information. As an example, if the vectors embedding correspond to words, the noun embedding should contain the information of the corresponding adjective. Each embedding vector has a query vector, usually of smaller dimension. It is computed by \\( W_Q e_i = q_i \\). The query vector encodes the context. For example the notion that nouns have preceding adjectives. I like imagining that the query matrix asks "what are you?" to the embedding, and the embedding for orange answers "A juicy fruit" or "a color", depending on its embedding vector (which contains the context that is encoded). Additionally a key is computed \\( W_k e_i = k_i \\) having equal dimension to the query vector. Given an embedding with a specific query vector, the keys which are close to that query should come from embeddings which are conceptually close. In essence, when the \textit{dot product} between \\( q_i \\) and \\( k_j \\) is large, then \\( i \\) and \\( j \\) are conceptually close. For example, the query of the sky embedding could be close to the key of the blue (but not green) embedding. The embedding of \textit{blue} attends to the embedding of sky. The query vector of the orange fruit embedding might be close to the key vector of juice, tree, banana, breakfast. It might be far from petroleum, graphene, lithium. The dot product between query and key vector corresponds to a weight which assigns to each embedding an attention value reflecting how important they are to each other. The \textit{dot product} is passed through a \textit{softmax} layer to create a value range from zero to one. This is the attention value.   
  
Let \\( Q = \\{q_i \\}= \\{ q_1, ... ,q_n \\} \\) and \\( K = \\{k_j \\} = \\{ k_1, ... ,k_n \\} \\) be matrices corresponding to the key and query vectors for the \\( n \\) embeddings. Then 

$$ \text{softmax}( \frac{QK^T}{\sqrt{n}} ) $$ 

corresponds to the attention values computed column by column creating an attention value matrix. Each entry, denoted by \\( a_{ij} \\), reflects the importance of embedding \\( i \\) to embedding \\( j \\). The division with \\( \sqrt{n} \\) happens for numerical reasons. Notice that this matrix has dimension \\( n \times n \\). This matrix is then used to update the \\( d \\) dimensional embedding. This notion of mapping vectors into a inner-product space is similar to kernel methods and reproducing (kernel) hilbert spaces. Usually these methods use parameterized filters which define the inner-product, such as the gaussian kernel. In the attention mechanism, these kernels are learned via the query and value matrix respectively. [Revisiting Kernel Attention with Correlated Gaussian Process Representation ](https://tanmnguyen89.github.io/gp_transformer.pdf) has extended this concept to Gaussian Processes.   
  
Until now, we have explained how to determine contextual closeness via the Query and Key matrix but not how to update embeddings according to this contextual closeness. For this consider the value-vector computed by \\( W_v e_i = v_i \\), where \\(v_i \\) is also \\(d \\)-dimensional. This results in \\(n \\) value-vectors. These value vectors are added to the original embeddings \\(e_i \\) to do a contextual the update. For each query \\(q_i \\) compute 

$$u_i = \sum_{j=0}^{n}a_{ij}*v_i.$$ 

This essentially is an attention weighted sum of all value vectors \\( v_i \\). Then 

$$ e_i^{t+1} = e_i^{t} + u_i $$ 

corresponds to the update of the embedding. This process of updating the embeddings \\( e_i \\) with the key \\( W_k \\), query \\( W_q \\) and value \\( W_v \\) matrices is one head of attention. The three matrices are the tuneable model parameters related to this computation. Often the notation is 

$$ \text{Attention}(Q,K,V) = \text{softmax}( \frac{QK^T}{\sqrt{n}})V $$ 

Using multiple key, query and value matrices is called a \textit{multi attention-head}. The computation of parameters for each attention head can be done in parallel. During the embedding update, all attention heads are used to compute the weighted change of the embedding 

$$ u_i^{k} = \sum_{j=0}^{n}a_{ij}^{k}*v_i^{k}, $$ 

which is then added to the embedding accordingly 

$$ e_i^{t+1} = \sum_{k} e_i^{t} + u_i^{k}.$$ 

This operation is the main ingredient in Transformer architectures. While it has been popularized mainly through applications in the language processing domain, many other areas can be included. [Vision Transformers](https://www.ecosia.org/search?q=Vision+Transformer&addon=chrome&addonversion=6.0.4&method=topbar) use attention in a visual context by splitting the Image into patches, and encoding these into a sequence of embeddings. Once the information is available in this for multi-head attention can be applied as described above. 

### Reproducing Hilbert Space

The Reproducing Hilbert Space is a vector space of functions (infinite dimensional vectors) induced with an inner product. This inner product allows for the notion of orthogonality and correlation. In a reproducing hilbert space it has a reproducing kernel property 

$$ f(z) = \sum_{n=0}^{N} a_n k(x_n,z) = \sum_{n=0}^{N} a_n k_{x_n}(z) $$. 

By the representer theorem, every function that minimizes an empirical risk functional can be represented as a linear combination of the inner products evaluated at the training points. Note the similarity to Kernel-covariances, which can thus be understood as mapping data-sample pairs into this space by defining their inner product. Similariy, the attention mechanism can be understood as learning a kernel function by computing an inner product between query and key. However, for a transformer, the \\( \text{softmax}( \frac{QK^T}{\sqrt{n}} ) \\) can be asymmetric and thus not ammenable to a reproducting kernel hilbert space. Workarounds exist which allow for asymmetric attention but enable symmetric kernel definitions enabling inference with uncertainty information as in gaussian processes.   
  
The formalization of a reproducing hilbert space can be done by considering the space of functions which are linear combinations of \\( N = \infty \\) orthonormal bases (basis functions) 

$$ f(x) = \sum_{n=1}^{N} a_n l_n(x). $$ 

Let us assume that \\( \sum |a_n|^2 dx \leq 1 \\) and \\( -1 \leq a \leq 1\\). Consider 

$$ k(x,x) = \langle f(x)|g(x) \rangle = \langle \sum_{n=1}^{N} a_n l_n(x) | \sum_{i=1}^{m} b_n l_n(x) \rangle = \sum_{n1}^{N} a_n b_n $$ 

which defines an inner product of a Hilbert space. Let us define \\( k(z,x) = \langle f(z)|g(x) \rangle = \sum c_n l_n(z)l_n(x) = \sum_{n=0}^N c_n z^n x^n \\) which converges (let us assume \\(\sum c_n = 1 \\), usually one requires \\(a_n\\) to be square integrable, so that \\( \sum a_nb_n\ = \sum c_n\\) converges to some value \\(c=1\\) ) for \\(zx \leq 1 \\) to 

$$ \sum_{i=1}^{n} c_n z^n x^n = \frac{c}{1 - zx}  $$. 

Here we have chosen the functions \\( l_n(x) = x^n \\) as basis functions. The convergence of \\( k(x,z) = \sum l_n(z)l_n(x) \\) and of \\( \sum c_n \\) are an important requirements on reproducing hilbert spaces. The first constrains the basis functions to be square integrable, the second contrains the coefficients to be square integrable. Note, 

$$ \langle k(z,x) | f(x)\rangle =\langle f(x)| k(z,x) \rangle = \langle f(x)| \frac{c}{1 - l(z)l(x)} \rangle = \langle f(x)| \frac{1}{1 - zx} \rangle = \sum_{n=1}^{N} a_n k(x,z) = \sum_{n=1}^{N} a_n z^n = \sum_{n=1}^{N} a_n l_n(z) = f(z) $$. 

This is the reproducing property of the kernel associated to the hilbert space. Thus a hilbert space with such a kernel is called reproducing hilbert space.   
  
In another example, consider 

$$ l_n(x) = \frac{x^n}{\sqrt{n!}} \exp^{\frac{-x^2}{n}} $$ 

which defines a basis of gaussian functions. As before we obtain 

$$ k(x,x) = \langle f(x)|g(x) \rangle = \langle \sum_{n=1}^{N} a_n l(x) | \sum_{i=1}^{n} b_n l(x) \rangle = \sum_{n=1}^{N} a_n b_n $$ 

and 

$$ k(z,x) = \sum_{n=1}^{N} c_n l_n(z)l_n(x) = \sum_{n=1}^{N} c_n ( \frac{z^n}{\sqrt{n!}} \exp^{\frac{-z^2}{n}} ) (\frac{x^n}{\sqrt{n!}} \exp^{\frac{-x^2}{n}}) $$ $$ = c * \exp^{ \frac{ -(z-x)^2 }{ 2 } } $$ 

which is the gaussian kernel function. So the gaussian kernel function is the inner-product for the reproducing hilbert space of gaussian functions. As before 

$$ \langle f(x) | k(z,x) \rangle = \langle k(z,x) | f(x) \rangle = \sum_{n=0}^{N} a_n k(x_n,z) = \sum_{n=0}^{N} a_n k_{x_n}(z) = \sum_{n=0}^{N} a_n l_n(z) = f(z). $$. 

The typical notation for writing functions as linear combinations of reproducing kernels is 

$$ f(z) = \sum_{n=0}^{N} a_n k(x_n,z) = \sum_{n=0}^{N} a_n k_{x_n}(z) $$. 

This allows for the representation of functions in the reproducing hilbert space. The parameters \\( a_n \\) are learned by gradient methods. Note, that the reproducing kernel is symmetric and positive definite, thus defining a manifold and allowing for an interpretation as a regular (or singular) elliptical distribution in the infinite dimensional (functional) Hilbert space. 

### Reproducting Hilbert Space and Gaussian Process connections

Note the similarity to gaussian processes where inference is done via 

$$ \mathbb{E}[x^* | X ] = k(z^*,Z) K(Z)^{-1}X $$ , (we assume \\(\mu_X =0 \\) for simplicity) and whereas in the reproducing hilbert space 

$$ f(z^*) = \sum_{n=0}^{N} a_n k(z_n,z^*) = \sum_{n=0}^{N} a_n k_{z_n}(z^*) $$. 

If \\( K = k_{i,j} = k_(z_i,z_j) \\) is invertible (i.e. not singular), then \\( a = (a_1,...,a_N) = K^{-1}X \\), where \\( X = (x_1,...,x_N) \\) is the vector of observations. Thus 

$$ f(z^*) = k_{z_n}(z^*)K^{-1}X = k(z^*,Z) K(Z)^{-1}X = \mathbb{E}[x^* | X ] $$. 

This shows that the linear combination of reproducing kernels and the posterior mean of a gaussian process are equivalent. [](https://arxiv.org/abs/1807.02582) provides a good summary and relates the uncertainty information of a gaussian process to the worse case error in a reproducing kernel hilbert space. This perspective enables a unification of fisherian (kernel spaces and ridge regression) and bayesian (posterior inference) concepts. Short, the kernel of a gaussian process corresponds to the reproducing inner product of a kernel hilbert space. The hyptothesis space of the former is slightly larger than the former, probably due to the probabilistic nature of the prior over the model parameters. 

### Kernel Attention

There exist mechanisms which enable using attention to learn a kernel function. Since the kernel of a reproducing hilbert space needs to be symmetric and positive definite, this imposes some restrictions on the attention mechanism. [Calibrating Transformers via Sparse Gaussian Processes](https://arxiv.org/abs/2303.02444) explores a mechanism for learning covariances between key and query pairs, enabling inference in a repoducing hilbert space with uncertainty information. The attention mechanism is constrained so that the resulting kernel is symmetric and positive definite. [Revisiting Kernel Attention with Correlated Gaussian Process Representation](https://tanmnguyen89.github.io/gp_transformer.pdf) proposes a method which avoids these restrictions and constructs a symmetric kernel from two gaussian processes, allowing for asymmetric softmax-functions between query and key. This enables greater flexibility. Intuitively, asymmetry might enable representations where, e.g. the embedding for sky is closer to the embedding to blue than the embedding of blue to the embedding of sky. This asymmetric does not adhere anymore to an intuitive description (at least not to mine) as a covariance matrix. 

### Asymmetric Kernels

The structures we examined up until now are symmetric. This fact is visualized in the figures and represented by a symmetric covariance matrix, which in turn defines symmetric distances. We have seem the KL-divergence as an asymmetric distance measure and highlighted the fact that softmax functions in attention mechanisms are also asymmetric, allowing for a directed distance. This directed distance is useful in many applications, for example in language processing, time dependencies and causality. Directed distances can be represented by asymmetric covariances. Intuitively, consider the Mahanalobis distance, but assume an asymmetric covariance matrix. This would allow for a directed distance. An example of an asymmetric kernels is the exponential decay kernel, defined as 

$$ k(t_i, t_j) = \exp(-\alpha |t_i - t_j|) \cdot \mathbb{1}_{t_i \leq t_j} $$ 

, where \\( \\\mathbb{1}_{t_i \leq t_j} \\) ensures that the kernel only considers past influences. This directional sense is also relevant in causality. [AN OVERVIEW OF CAUSAL INFERENCE USING KERNEL EMBEDDINGS](https://arxiv.org/pdf/2410.22754) provides a good summary on the application of kernels and causality. 

### More thoughts: 

#### Priors in function space

Often the isotropic gaussian prior of variational autoencoders is too restrictive. It could be interesting to use a reproducing hilbert space as the latent space of a variational autoencoder or an autoencoder in general. Some interesting research in this direction is [Autoencoders in Function Space](https://arxiv.org/pdf/2408.01362) and [Autoencoding any Data through Kernel Autoencoders](https://arxiv.org/abs/1805.11028). The first shows a comparable performance of their architecture to convolutional neural nets. [Expressive Priors in Bayesian Neural Networks](https://arxiv.org/pdf/1905.06076). 

#### Kernels with inductive biases

The kernel autoencoder does not have the convolutonal inductive bias (i.e. translation/rotation equivariance). It would be interesting to see if kernels can be restricted to represent equivariant and invariant functions only, thus replicating the inductive bias of convolutional neural nets and allowing for better generalization. For example, using the right kind of basis functions, such as spherical harmonics (which are SO3 equivariant) might enable using equivariant functions for inference with uncertainty information. It is not clear to me, if the corresponding kernel can be computed in reasonable time, but i found a [paper](https://www.sciencedirect.com/science/article/pii/S002190451730076X), which provides the analytical form of the kernel of the space of symplectic harmonics (warning, much math). For many practical application symplectic transformations are desired, although its relation to singular distributions might be relevant. [Sparse Gaussian Processes with Spherical Harmonic Features](https://arxiv.org/pdf/2006.16649) is an interesting application of Gaussian Processes with spherical harmonic features. This can be compared to [So3krates](https://arxiv.org/pdf/2205.14276) and [Accurate global machine learning force fields for molecules with hundreds of atoms ](https://www.science.org/doi/10.1126/sciadv.adf0873). The first uses spherical harmonics to learn equivariant and global features, the second uses a sparse gaussian process approach. [A Euclidean transformer for fast and stable machine learned force fields](https://www.nature.com/articles/s41467-024-50620-6) is the latest publication on learning force fields, using spherical harmonic features in a transformer architecture. No uncertainty information is provided by the architecture (it seems like the sparse gaussian process was omitted in favour of the sperhical harmonics approach ?). Potentially, both can be combined via [Revisiting Kernel Attention with Correlated Gaussian Process Representation](https://tanmnguyen89.github.io/gp_transformer.pdf). Similariy, using the fourier basis might also be interesting in other settings, since they can be understood as a simpler version of harmonics.    
  


#### Kernels for learning function expansions

In quantum chemistry, the wave function is approximated as linear combination of slater-determinants (representing the anti-symmetric orbitals), potentially also allowing for application of kernel methods. A problem can be the high parameter count. Generally, all functions which adher to the formulation 

$$ f(x) = \sum_{n=1}^{N} a_n l(x). $$ 

are amenable to a reproducing hilbert space, kernalized learning and inference with uncertainty information. The kernel function is then 

$$ k(x,y) = \sum_{n=1}^{N} a_n l(x) l(y). $$ $$ 

under conditions of convergence on its fourier series with respect to the orthonormal system \\( l(x) \\). It might not be trivial to obtain and compute the right kernel function. If the real functions has component that are orthogonal to the space of basis-functions, this part can not approximated. 

#### Kernels of basis functions

Not every basis function might be amenable to a reproducing hilbert space. Finding the right kernel might be difficult. The basis functions and its coefficients must be square integrable (which is a rather mild restriction) and the value it converges to must be computable in an efficient way (ideally analytically in closed form). 

![White Noise](../images/covariance/data_images/white_noise.png)
![Transformation to the same mean, different covariance](../images/covariance/data_images/specific_covariance.png)
![Regular covariance with linear mean](../images/covariance/data_images/transformed_uniform_line (regular distribution).png)
![Singular covariance](../images/covariance/data_images/transformed_uniform_line (singular distribution).png)
![Sinusoidal Mean and constant variance](../images/covariance/data_images/sine_const_var.png)
![Circular Mean and constant variance](../images/covariance/data_images/circle_const_Var (2).png)
![Discontinuity in covariance](../images/covariance/data_images/two_variances.png)
![Smoothly varying covariance (linear)](../images/covariance/data_images/continous_varying_variances.png)
![Smoothly varying covariance (nonlinear)](../images/covariance/data_images/continous_varying_variance_sine.png)
![Nonlinear distribution on a circle](../images/covariance/data_images/circle_mean_sine_variance.png)
![Nonlinear distribution on a parabola](../images/covariance/data_images/parabola_mean_sine_variance.png)
![Nonlinear covariance distribution with sine-wave mean](../images/covariance/data_images/Gaussian_Process_Regression_data.png)