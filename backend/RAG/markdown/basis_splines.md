# Basis Splines

Here I explain how projection of a smooth function (potentially in any dimension) onto basis splines is performed. For example, one might have a scalar cuntion in a 3D-domain, i.e. $s = f(x,y,z)$, which is stored as a vector of values at discrete coordinates. This continuous function can be represented as a linear combination of weighted basis functions which maps the values vector to a value vector of the control points. Usually, there are less control points than discrete points in the domain, which allows for an efficient dimensionality reduction. This is how I first stumbled upon basis splines, but there are many other possible use cases. This methodology of representing lines, surfaces, 3D-functions or ND-functions in terms of continuous basis functions is called B-spline representation. Its theoretical basis is the sequential application of Bezier functions so that continuity conditions are fulfilled.

## Bezier Functions

![Quadratic Bernstein](../images/basis-splines/quadratic-bernstein.png) ![Cubic Bernstein](../images/basis-splines/cubic_bernstein.png)

Bezier functions can be used to represent lines in space as a linear combination of basis functions. Consider a 1D-function and discrete $d+1$ control points $p_i = x_i$ for $i=0,...,d$.

Consider the parameterization of the Bernstein polynomials, where $u \in [0,1]$ and $d$ corresponds to the degree:

$$ B_{i,d}(u) = \binom{d}{i} (1-u)^{d-i} u^i. $$ 

If $d=3$, then the cubic Bezier functions are defined as:

$$ B_0(u) = (1-u)^3, \quad B_1(u) = 3u(1-u)^2, $$ 

$$ B_2(u) = 3u^2(1-u), \quad B_3(u) = u^3. $$ 

We can also write:

$$ B(u) = [B_0, B_1, B_2, B_3](u) = [u^3, u^2, u, 1] \begin{bmatrix} -1 & 3 & -3 & 1 \\\ 3 & -6 & 3 & 0 \\\ -3 & 3 & 0 & 0 \\\ 1 & 0 & 0 & 0 \end{bmatrix} = u^T M. $$ 

For each control point, there is one Bezier function. The construction of a degree $d$ Bezier function requires $d+1$ control points. The Bezier functions are used as a basis to define a curve as:

$$ \mathbf{f}(u) = \sum_{i=0}^{d} \mathbf{c}_i \, B_{i,d}(u), $$ 

which can be written in matrix form as:

$$ f(u) = Bc = u^T M c, $$ 

where $c = [ c_0, c_1, ..., c_d]^T$, $u = [1, u, u^2, ..., u^d]$ and $M$ is a matrix of scalars. For most practical purposes, $d=3$ is sufficient.

Assume there is a function $r(u)$ that is provided as a vector of discrete values at $n$ points, i.e., $r \in \mathbb{R}^n$. This function can be approximated by:

$$ f(u) = \sum_{i=0}^{d} c_i B_{i,d}(u) = Bc, $$ 

by determining the coefficients $c = [c_0, c_1, ..., c_d]^T$ which minimize the mean squared error:

$$ E(c) = \sum_{j=0}^{n} \left( r[j] - \sum_{i=0}^{d} c_i B_{i,d}(u) \right)^2, $$ 

or in matrix notation:

$$ E(c) = (r - Bc)^2. $$ 

![Bezier Surface](../images/basis-splines/bezier_surface.png)

## Basis Splines

The basis-spline approach approximates functions by the linear combination of section-wise defined basis functions. In essence, each basis function only has a finite range of influence. This provides increased control and eliminates the issue of increasing basis function degree with increasing control points, but it introduces potential continuity issues at the control points.

![Combination of Basis Splines](../images/basis-splines/combinationa.png)

A $k$-degree B-spline curve defined by $n+1$ control points will consist of $n-k+1$ Bezier curves. For example, a cubic B-spline defined by 6 control points $P_0, ..., P_6$ consists of 3 Bezier curves. The support of each basis function is determined by a knot vector $k \in m + d + 1$, where $m$ represents the number of control points and $d$ the degree of the basis functions.

One starts with a base case (degree 0) functions defined as:

$$ B_{i,0}(u) = \begin{cases} 1 & \text{if } u_i \leq u < u_{i+1}, \\\ 0 & \text{otherwise}. \end{cases} $$ 

and recursively constructs higher degree functions via:

$$ B_{i,d}(u) = \frac{u - u_i}{u_{i+d} - u_i} B_{i,d-1}(u) + \frac{u_{i+d+1} - u}{u_{i+d+1} - u_{i+1}} B_{i+1,d-1}(u), $$ 

where $d$ denotes the degree.

The curve in this basis is defined as:

$$ f(u) = \sum_{i=0}^{d} c_i B_{i,d}(u) = Bc. $$ 

Given the vector $s$ of $n$ values, the discretization of the disk domain is chosen. This defines the number of control points $m$. The basis functions and the matrix $B$ are computed. The mean squared error is constructed as:

$$ E(c) = (s - Bc)^2. $$ 

The analytical solution for the optimal parameter values is:

$$ c = (B^T B)^{-1} B^T s. $$ 

![Basis Spline](../images/basis-splines/Basis-spline.png)

## Extension to 3D and Application

The previous sections provide the mathematical background for basis function representations of 1D functions. The extension to multiple dimensions is achieved by defining multiple 1D basis functions in basis directions. For 3D functions, one might define a basis-spline function for each $x$, $y$, $z$ in Cartesian or $r$, $\theta$, $z$ in cylindrical coordinates.

The basis in 3D is defined as a product of the basis-spline functions:

$$ B_i(u) = B_i^r(u) \cdot B_i^\theta(u) \cdot B_i^z(u), $$ 

and the function as a linear combination:

$$ f(u) = \sum_{i=0}^{m-1} c_i B_i(u) = Bc. $$ 

## Sources

For the creation of this summary, I used the YouTube videos from Dr. Jon Shiach from the Manchester Metropolitan University, School of Computing Mathematics and Digital Technology. All of the graphics are directly screenshotted from his two video lectures on basis splines and bezier functions. Freya Holmer provides videos on YouTube regarding basis splines and their continuity properties, which, as mentioned, are the reason for the recursive construction and knot-value vector.

![Quadratic Bernstein](../images/basis-splines/quadratic-bernstein.png)
![Cubic Bernstein](../images/basis-splines/cubic_bernstein.png)
![Bezier Surface](../images/basis-splines/bezier_surface.png)
![Combination of Basis Splines](../images/basis-splines/combinationa.png)
![Basis Spline](../images/basis-splines/Basis-spline.png)