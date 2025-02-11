# Kalman Filter

## Introduction

Let us look at the Kalman filter. We assume, a noisy linear system with covariance matrix \\(Q\\) and that noisy measurements with covariance matrix \\(R\\):

$$ x_t = F x_{t-1} + Bv_{t-1} $$ $$ y_t = H x_t + Dw_t $$ 

where _\\(v_k\\)_ and _\\(w_k\\)_ are independent gaussian noise terms with zero mean and isotropic covariance. Consider _\\( \mathbb{E}(x_0) = \mu_0 \\)_ and _\\( \mathbb{Var}(x_0) = \Sigma_0 \\)._

Let us look at the time evolution of this mean and covariance. 

$$ \mu_{t+1} = F \mu_{t} $$ $$ \Sigma_{t+1} = F \Sigma_{t} F^T + BB^T $$ 

For _y_ , the evolution can be obtained analogously . Note that 

$$ \lim_{t \rightarrow \infty} \Sigma_{t} = \lim_{t \rightarrow \infty} F \Sigma_{t-1} F^T + BB^T $$ $$ = \lim_{t \rightarrow \infty} F^t \Sigma_0 \left(F^t\right)^T + \sum_{k=0}^{t-1} F^{k} BB^T \left(F^k\right)^T $$ $$ = \lim_{t \rightarrow \infty} \sum_{k=0}^{t-1} F^{k} BB^T \left(F^k\right)^T $$ 

since _F_ is assumed stable (i.e. it has eigenvalues stricly smaller zero). In the limit we have

$$ \Sigma_{\infty} = F \Sigma_{\infty} F^T + BB^T $$ 

This equation is stricly positive, convex and a Lyapunov Equation. If \\(F\\) is stable, it has an unique solution _\\( \Sigma_{\infty} \\)_

Consider now \\( B = I\\) and \\( D = I\ ), and assume the noise terms are zero mean with covariance 

$$ \text{VAR} \begin{bmatrix} v \\\ w \end{bmatrix} = \begin{bmatrix} Q & S \\\ S^T & R \end{bmatrix} $$ 

The state and measurement noises are uncorrelated at different times but at time _t_ correlation can exist. We can deconstruct _v_ into a term correlated to _w_ and one uncorrelated to it. We obtain 

$$ \tilde{v}(t) = \mathbb{E}[v(t)|w(t)] = v(t) - S R^{-1} w(t) = v(t) - S R^{-1}y(t) + S R^{-1} C x(t) $$ 

by the conditional mean of a gaussian. Thus

$$ x(t+1) = (F - S R^{-1} C) x(t) + S R^{-1}y(t) + \tilde{v}(t) $$ $$ y(t) = Cx(t) + w(t) $$ 

with

$$ \text{Var}( \begin{bmatrix} \tilde{v} \\\ w \end{bmatrix}) = \begin{bmatrix} \tilde{Q} & 0 \\\ 0 & R \end{bmatrix} $$. 

where _\\( \tilde{Q} = Q - SR^{-1}S^T \\)_. These are the starting equations for the Kalman Filter. We want to obtain a predictor _\\( \hat{x}(t+1|t) = \mathbb{E}(x(t+1)|H_t(y)) \\)_ , filter _\\( \hat{x}(t|t)= \mathbb{E}(x(t+1)|H_{t+1}(y)) \\)_. The predictor error is _\\( e(t+1|t) = x(t+1) - \hat{x}(t+1|t) \\)_ and its covariance is _\\( P(t+1|t) =\mathbb{E}[e(t+1|t)e(t+1|t)^T] \\)_. Analogues can be obtained for the filter. We call _\\( l(t) = y(t) - C \hat{x}(t|t-1) = y(t) - \hat{y}(t) \\)_ the innovation. It can be realted to how much we trust the state-predictor given the measurements. Of course, state and measurements are subject to noise. We have introduced _\\( H_{t}(y) \\)_ which corresponds to an aggregation of the measurement information at time _t_. This allows a representation in terms of a Markov chain, where the next state only depends on the current observation. Consider the time update of the predictor mean and its error covariance 

$$ \hat{x}(t+1|t) = F \hat{x}(t) + S R^{-1}y(t) $$ $$ P(t+1|t) = F P(t|t) F^T + \tilde{Q} $$ 

and the measurement update 

$$ \hat{x}(t+1|t+1) = \hat{x}(t+1|t) + L(t+1)l(t+1) $$ $$ l(t+1) = y(t+1)- C \hat{x}(t+1|t) $$ 

where 

$$ L(t+1) = P(t+1|t)C^T \Delta^{-1} (t+1) $$ $$ \Delta(t) = \text{Var} (l(t)) = CP(t|t-1)C^T +R $$ $$ P(t+1|t+1) = P(t+1|t) - P(t+1|t)C^T \Delta^{-1} (t+1)CP(t+1|t) $$ 

Find a proof here. We can summarize these equations 

$$ \hat{x}(t+1|t) = F \hat{x}(t|t-1) + SR^{-1}y(t) + K(t)(y(t) - C \hat{x}(t|t-1)) $$ $$ K(t) = F L(t) = F P(t|t-1)C^T \Delta^{-1}(t) $$ $$ \Delta(t) = \text{Var}( l(t) ) $$ $$ P(t+1|t) = F[ P(t|t-1) - P(t|t-1)C^T \Delta^{-1}(t)CP(t|t-1)]F^T + Q $$ $$ P(t_0|t_0-1) = P_0 $$ 

This can be interpreted as a system with state-feedback which is fed by the innovation \\(l(t) = y(t+1)- C \hat{x}(t+1|t) \\). Note, that

$$ \hat{x}(t+1|t) = (F - K(t)C)\hat{x}(t|t-1) + (SR^{-1}+K(t))y(t) = \Gamma(t)\hat{x}(t|t-1)+ (SR^{-1}+K(t))y(t) $$ and $$ x(t+1) - \hat{x}(t+1|t) = \Gamma(t)(\hat{x}(t|t-1)) - K(t)w(t) + \tilde{v}(t) $$ 

where _\\( \Gamma(t) = F - K(t)C \\)_ determines the error dynamics.

$$ P(t+1|t) = \Gamma(t)P(t|t-1)\Gamma(t)^T + K(t)RK(t)^T + \tilde{Q} $$ $$ P(t_0|t_0-1) = P_0 $$ 

To simplify we assume no noise correlation, i.e \\(S = 0 \\), leading to the simplified version

$$ \hat{x}(t+1|t+1) = F \hat{x}(t|t) + P(t+1|t)C^T \Delta^{-1}(t+1)(y(t+1) - CF\hat{x}(x|x)) $$ $$ \Delta(t+1)= F L(t+1) = C P(t+1|t)C^T+R $$ $$ P(t+1|t+1) = [I - P(t+1|t) C^T \Delta^{-1} (t+1)C]P(t+1|t) $$ 

Note that 

$$ P(t+1|t) = FP(t|t-1)F^T - FP(t|t-1)C^T \Delta^{-1} (t)CP(t|t-1)F^T + Q $$ 

and

$$ \hat{x}(t+1|t) = (F - K(t)C)\hat{x}(t|t-1) + Ky(t) = \Gamma(t)\hat{x}(t|t-1)+ K(t)y(t) $$ 

are especially important equations. They can be used to determine the asymptiotic behaviour of \\( \lim_{t \rightarrow \infty}P(t+1|t) \\) and \\( \lim_{t \rightarrow \infty}\hat{x}(t+1|t) \\). Recall that we have seen that, in the limit, the covariance of the process is determined by 

$$ \Sigma = F \Sigma F^T + Q $$ 

If \\( F \\) is stable, then \\( x(t) \\) converges to a stationary value and becomes a stationary process. If this is the case, then _P(t)_ also converges. For example if _K = 0_ (no measurement data is avaialable), _\\(\lim_{t \rightarrow \infty} \hat{x}(t+1|t) = 0 = \lim_{t \rightarrow \infty} \mathbb{E}x(t) \\)_ and _\\( \lim_{t \rightarrow \infty}P(t) = \Sigma \\)._ If _\\( \Gamma \\)_ is unstable, then _\\( \hat{x} \\)_ is non stationary. Stability of _\\( F \\)_ is a sufficient but not a necessary criterion for predictor convergence. It is also possible to have predictor convergence and/or bounds with an unstable system. This predictor convergence of a non stationary state evolution is made possible by equally non-stationary measurement information _\\( y(t) \\)_ which feeds the predictor through the innovation. This can be shown on the simple system 

$$ x(t+1) = \alpha x(t) + w(t) $$ $$ y(t) = \gamma x(t) + v(t) $$ 

with _\\( w(t) \sim \mathcal{N}(0, \beta) \\)_ and _\\( v(t) \sim \mathcal{N}(0, 1) \\)_. We obtain 

$$ P(t+1) = \beta^2 + \frac{\alpha^2P(t)}{1 + \gamma^2P(t)}. $$ 

This equation has a stable solution if _\\( \gamma > 0 \\)_. If _\\( \gamma = 0 \\)_ then _P(t)_ diverges. If _\\( \beta = 0 \\)_ , i.e. the state evolution is not noisy, then there exist two solutions. One is stable and one is unstable. The unstable solution is at _P(t) = 0_. _\\( \gamma \neq 0 \\)_ and _\\( \beta \neq 0 \\)_ ensure the existence of a unique solution satisfying the convergence of _P(t)_ to a stabilizing covariance. We have examined a simple system with one dimension only. Translated into a higher dimensional system, the same principles apply. We have 

$$ x(t+1) = A x(t) + Bw(t) $$ $$ y(t) = C x(t) + Dv(t) $$ 

with _\\( w(t) \sim \mathcal{N}(0, I) \\)_ and _\\( v(t) = \sim \mathcal{N}(0, I) \\)_ or _\\( Bw(t) \sim \mathcal{N}(0, BB^T = Q) \\)_ and _\\( Dv(t) \sim \mathcal{N}(0, DD^T = R) \\)_. If _(A,C)_ is observable (no distinct _x(0)_ generate the same outputs), then there exists at elast one solution for the covariance _\\( \lim_{t \rightarrow \infty} P(t) \\)_. There could be more than one solution (as in the case _\\( \beta = 0 \\)_ and _\\( \gamma \neq 0 \\)_ ). If _(A,B)_ is controllable (equivalent to controllable _(A,Q)_), then the solution is unique and stabilizing. These notions can be relaxed to detectability and stabilizability. In essence, if _(A,B)_ is stabilizable and _(A,C)_ is detectable (i.e. all uncontrollable and unobservable modes are asymptotically stable), then there exists a unique stabilizing solution for the covariance _\\( \lim_{t\rightarrow \infty}P(t) \\)_. Find the proof here. 

## Proofs

### Time and Measurement Update

Let us look at the time update of the predictor mean and its error covariance

$$ \hat{x}(t+1|t) = F \hat{x}(t) + S R^{-1}y(t) $$ $$ P(t+1|t) = F P(t|t) F^T + \tilde{Q} $$ 

and the measurement update 

$$ \hat{x}(t+1|t+1) = \hat{x}(t+1|t) + L(t+1)l(t+1) $$ $$ l(t+1) = y(t+1)- C \hat{x}(t+1|t) $$ 

where 

$$ L(t+1) = P(t+1|t)C^T\Delta^{-1}(t+1) $$ $$ \Delta(t) = \text{Var}( l(t) ) = CP(t|t-1)C^T +R $$ $$ P(t+1|t+1) = P(t+1|t) - P(t+1|t)C^T\Delta^{-1}(t+1)CP(t+1|t) $$ 

Consider 

$$ x(t+1) = F x(t) + S R^{-1}y(t) + \tilde v(t) $$ $$ y(t+) = Cx(t) + \tilde w(t) $$ 

and note that _v(t)_ is independent from _y(t)_. Thus _\\( \mathbb{E}(x(t+1)|H_{t+1}(y)) = \hat{x}(t+1|t) = F \hat{x}(t|t) + S R^{-1}y(t) \\)_ and therefore 

$$ e(t+1|t) = x(t+1) -\hat{x}(t+1|t) = Fx(t) + \tilde{v}(t) + S R^{-1}y(t) - F\hat{x}(t|t) - S R^{-1}y(t) $$ $$ = F( x(t) - \hat{x}(t|t) ) + \tilde{v}(t) = F e(t+1|t) + \tilde{v}(t) $$ 

and 

$$ P(t+1|t) = F P(t|t)F^T +\tilde{Q} = \text{Var}(e(t+1|t)) = F \text{Var}(e(t|t))F^T + \text{Var}(\tilde(v)(t)) $$ 

Furthermore, consider the measurement update and the fact that the space spanned by _l(t)_ is orthogonal to the space spanned by _y(t)_ (by definition). Thus _\\( H_{t+1}(y) = H_{t}(y)+ H(l(t+1)) \\)_. Thus 

$$ \mathbb{E}(x(t+1)|H_t(y)) = \mathbb{E}(x(t+1)|H_t(y)) + \mathbb{E}(x(t+1)|H(l(t+1))) = \hat{x}(t+1|t) + \text{cov}(x(t+1,l(t+1)))+\text{Var}(l(t+1))^{-1}l(t+1) $$ $$ = \hat{x}(t+1|t) + P(t+1|t)C^T(CP(t+1|t)C^T + R)^{-1} l(t+1) = \hat{x}(t+1|t+1) = \hat{x}(t+1|t) + L(t+1)l(t+1) $$ 

by the conditional of a gaussian. Updating the covariances is analogous: 

$$ e(t+1)= x(t+1) - \hat{x}(t+1|t+1) = x(t+1) - \hat{x}(t+1|t) - L(t+1)l(t+1) = e(t+1) - L(t+1)l(t+1) $$ 

leading to 

$$ e(t+1|t+1) + L(t+1)l(t+1) = e(t+1|t) $$ 

with variance 

$$ P(t+1|t+1) + L(t+1) \text{var}(l(t+1))^{-1} L(t+1)^T = P(t+1|t) = P(t+1|t+1) + L(t+1) \Delta^{-1} L(t+1)^T $$ 

This concludes the proofs. 

### Convergence and Stability of the predictor

