## Shape Distances

There are many ways to define distances. We have seen in section geometrical notions how the space of probability distributions is located in a Riemannian manifold, equipped with the Fisher-Information matrix as a metric. Within this manifold, the distance between distributions might be approximated by quadratic form approximations, which are generally non-symmetric but alleviate the computational cost. The Kullback-Leibler divergence can be seen as an information-theoretical distance between distributions. 

However, there are disadvantages to the usage of the KL-divergence as a similarity measure. For example, the anti-symmetric property might not always be desirable. Furthermore, the Kullback-Leibler divergence does not return meaningful values for distributions without overlap (if \\( q(x) = 0 \\) where \\( p(x) \geq 0 \\), it becomes undefined). While it allows for a nice information-theoretical perspective in terms of entropy, it is not well suited as a distance between shapes. To illustrate this, consider add example KL distance between similar shapes. 

Alternatives have been developed and successfully applied to several fields. These alternatives consider a metric space and are not linked to any continuous Riemannian space. However, in special cases, they might induce such a space. 

### Hausdorff Distance

The Hausdorff Distance measures the distance between two subsets of a metric space. It corresponds to the largest smallest distance between the elements of the respective subsets. 

Let \\( (\mathcal{X}, d) \\) be a metric space, and let \\( A, B \subseteq \mathcal{X} \\). The _Hausdorff distance_ \\( d_H(A, B) \\) is defined as 

\\[ d_H(A, B) = \max \left\\{ \sup_{a \in A} d(a, B), \; \sup_{b \in B} d(A, b) \right\\}, \\] 

where \\( d(A, b) = \inf_{a \in A} d(a, b) \\). 

A brute-force algorithm would be: 
    
    
    h = 0
    for each point a_i of A:
        shortest = infinity
        for each point b_j of B:
            d_ij = d(a_i, b_j)
            if d_ij < shortest:
                shortest = d_ij
        if shortest > h:
            h = shortest
            

So for each point in \\( A \\), the smallest distance to \\( B \\) is computed, and in the end, the maximum is picked out of all smallest distances. It is a relatively simple distance, straightforward to compute, and easily interpreted as measuring the worst-case deviation between two sets. Due to this, it is sensitive to outliers; if only one single element has a high minimal distance, the Hausdorff distance corresponds to it, since it focuses on extremes. 

### Wasserstein Distance

The Wasserstein Distance is a widely used metric that quantifies how difficult it is to deform one distribution \\( q(x) \\) into another distribution \\( p(x) \\). It is deeply related to optimal transport and can be interpreted as the minimal cost for shaping one form into another. Furthermore, it has distance properties and can be a good alternative to other dissimilarity measures, such as the KL-divergence. In contrast to the information-theoretical dissimilarity (KL divergence), it measures the difference between distributions in a geometric/spatial sense. 

Consider a space of probability measures \\( P(\Omega) \\) and let \\( p(x) \\) and \\( q(x) \\) define the distributions \\( P \\) and \\( Q \\), so that \\( X \sim P \\) and \\( Y \sim Q \\), with \\( X, Y \in \mathbb{R}^d \\). 

Consider the optimal transport map 

\\( T: \mathbb{R}^d \rightarrow \mathbb{R}^d \\), 

where the distribution of \\( T(X) \\) is called the push-forward (\\( T_{\\#}P \\)) of \\( P \\), defined as 

\\( T_{\\#}P (A) = P(\\{ x: T(x) \in A \\}) = P(T^{-1}(A)) \\). 

Here, \\( T_{\\#}P \\) is a new probability distribution (measure) induced by \\( P \\) through the mapping \\( T \\). \\( T_{\\#}P(A) \\) represents the probability that \\( T(X) \\) is in \\( A \\). The set of points \\( \\{ x: T(x) \in A \\} \\) can also be written as \\( T^{-1}(A) \\). 

The optimal transport distance is given by: 

\\[ \inf_T \int \| x - T(x) \|^p dP(x), \\] 

such that \\( T_{\\#}P = Q \\). If the minimizer \\( T^*(\cdot) \\) exists, it is called the optimal transport map. The map \\( T_t(x) = (1-t)x + tT^*(x) \\) is a geodesic connecting \\( P \\) to \\( Q \\). 

In this formulation, the minimizer might not exist since the measure (mass) cannot be split. To alleviate this, a soft formulation introduces a joint measure \\( dJ(x, y) \\) and defines the Wasserstein distance as 

\\[ W_p(P, Q) = \left( \inf_{J \in \mathcal{J}(P, Q)} \int_{x, y} \|x - y\|^p dJ(x, y) \right)^{1/p}, \\] 

subject to \\( \int_y J(x, y) dy = P \\) and \\( \int_x J(x, y) dx = Q \\). 

The minimizer \\( J^*(\cdot) \\) is called the optimal transport plan. If \\( T: X \rightarrow X \\), i.e., no splitting of mass occurs (input and output measure coincide), it is called the optimal transport map. 

In a discrete setting, with samples \\( x_i \\) and \\( y_j \\) occurring with probabilities \\( p_i \\) and \\( q_j \\), respectively, the Wasserstein distance can be written as: 

\\[ W_p(P, Q) = \min_{J \in \mathcal{J}(P, Q)} \sum_i \sum_j \|x_i - y_j\|^p J(x_i, y_j), \\] 

subject to \\( \sum_j J(x_i, y_j) = p_i \\), \\( \sum_i J(x_i, y_j) = q_j \\), and \\( J(x_i, y_j) \geq 0 \\). 

Interesting cases are \\( p=1 \\) (Earth-Mover distance) and \\( p=2 \\). For \\( p=2 \\), it can be shown that there exists a unique convex function \\( \phi \\) such that \\( J^*(\cdot) = \nabla \phi \\). Furthermore, for \\( p=2 \\), the metric space \\( \\{ P(\Omega), W_p \\} \\) has the structure of a Riemannian manifold, as pointed out by add citation. 

By converting the primal problem to its dual form, it can be shown that 

\\[ W_p(P, Q) = \sup_{\Phi, \Psi} \left( \int_{y} \Phi(y) dQ(y) - \int_{x} \Psi(x) dP(x) \right), \\] 

where for \\( p=1 \\), one can write 

\\[ W_p(P, Q) = \sup_{f} \left( \int_{x} f(x) dQ(x) - \int_{x} f(x) dP(x) \right). \\] 

The Wasserstein distance can also be used to define "typical" distributions from a set of distributions \\( P_1, P_2, \ldots, P_N \\), called the barycenter. This is the distribution \\( P_{\text{center}} \\), such that 

\\[ P_{\text{center}} = \arg \min_{P} \sum_{i=1}^{N} W_p(P, P_i). \\] 

Also interesting is the notion of distances between \\( P_0 \\) and \\( P_1 \\). Consider a map \\( c(t): [0,1] \rightarrow P(\Omega) \\), such that \\( c(0) = P_0 \\) and \\( c(1) = P_1 \\). Then 

\\( c(t) = (1-t)P_0 + tP_1 \\) 

is the geodesic connecting \\( P_0 \\) and \\( P_1 \\). There exists a path \\( c(t) \\), such that its length \\( L(c) = W_p(P_0, P_1) \\) and 

\\( P_t = F_{t \\#}J \\), 

where \\( J \\) is the optimal coupling and \\( F_t(x,y) = (1-t)x + ty \\). This allows for the interpolation between two distributions following the Wasserstein geodesic. 

A Linear Optimal Transportation Framework for Quantifying and Visualizing Variations in Sets of Images presents a method that projects these geodesics (nominally located in a curved, nonlinear space) to a linear embedding via linear optimal transport while preserving distances and angles. Then linear statistical analysis with PCA is used in the embedding space to quantify variations. However, they do not mention extensions to image generation. 

