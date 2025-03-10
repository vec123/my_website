# Variational energy principles

## Introduction

With this post i want so summarize the variational formulation of mechanics and mathematical analysis that arises from it. This formulation is the basis for Hamiltonian mechanics and the Hamilton-Jacobi theory. It is restricted to the fairly broad class of systems subject to forces, derivable from a single scalar function (monogenic forces). Examples of these forces are known as conservative forces, however, by an extension of the state-space even time and velocity dependent forces can be made monogenic. The details and intricacies will become clearer in the following. 

## Newtons equation, virtual work and D'Alemberts principle

Consider the famous equation of motion in newtonian mechanics

$$ F=ma = m\dot{v} = m\ddot{r} $$ 

where \\( F \\) is the force, \\( m \\) is the mass of the particle and \\( a =\dot{v}= \ddot{r}\\) is the acceleration in \\(3D\\) position space. This equation of motion is a second order differential equation, which can be solved by specifying the initial conditions of the system. The solution is the trajectory \\( r(t) \\) of the particle in position space. An important notion that arises is that of an equilibrium point, which is a point in position space where the force (and thus the acceleration) vanishes. If the velocity is also zero, the particle remains at rest. This is a stable, but not necessarily asymptotically stable, equilibrium point. One can differentiate between free particles, i.e. a particle that is not subject to any constraints and can be moved in arbitrary directions, and constrained particles. Assume a particle constrained to a plane (e.g. a ball on a table). Such a constraint can be described by \\( f(r) = f(x,y,z) = 0 \\). This ball is in equilibrium (no acceleration occurs) eventhough the force of gravity is acting on it. This is because a reactive force is acting on the ball, which cancels out the force of gravity. Considering this reactive force, we can write the equation of motion as 

$$ F + R = ma $$ 

where \\( R \\) is the reactive force. Let us introduce the variation of the work-function 

$$ \delta w = F \delta r, $$ 

where \\( \delta r \\) is the virtual displacement of the particle. It corresponds to any arbitrary variation of \\( r \\). The quantity \\(dr\\) is but a special type of variation. Thus, whenever \\( \delta \\) is used, the reader might replace it with \\( d \\) to obtain the special variation. The difference is that \\( \delta \\) represents any arbitrary variation, wheras \\( d \\) corresponds to the acutal displacement in time. Intuitively speaking, the virtual displacement is a small displacement that is not necessarily real. Thus the virtual work \\(\delta w \\) does not necessarily correspond to the real work that is being done. However, \\( dw \\) is a the work performed by a system on a real infinitesimal displacement \\(dr\\). Summarizing, the variation \\( \delta r \\) is any infinitesimal change, while \\(dr\\) is the "true" infinitesimal change occuring in time.   
  
Taking on the perspective of an observer moving together with particle in motion (e.g. a people sitting in a train), the particle appears at rest. From this perspective, the impressed forces cancel with the inertial forces. The work can be written 

$$ \delta w = (F + R - ma) \delta r = 0. $$ 

A common assumption is that the reactive forces \\(R\\) are ideal, in the sense that they act orthogonally to the motion. Then they do not perform any work i.e. \\(R \delta r = 0 \\). As an example, consider a ball on a table. Any impressed force not orthogonal to the table will perform work. The ball is accelerated, creating an equivalent inertial force. However, the reactive force is orthogonal to the table and does not perform work. Furthermore, it is only dependent on the position of the system, not its velocity, enforcing a position dependent (holonomic) constraint. This assumption of ideal reactive forces holds for many systems. More general constraints might be velocity dependent (non-holonomic). These include frictional forces, deformations, contacts with slip-stick conditions and non-equilibrium systems. Although the assumption of purely ideal reactive forces might seem restrictive, it is very common (similar to how most mathematical models are friction-free and offer deep insight nontheless). Under the assumption of ideality we obtain the following equation of equilibrium 

$$ \delta w = (F - ma) \delta r = 0, $$ 

which is known as D'Alemberts principle. This principle makes the statement that a system is in equilibrium if one adds the inertial forces to the impressed forces. It can be interpreted as the change from a static observer to an observer moving with the particle. For this moving observer, the particle appears at rest at all times. The experienced impressed forces and inertial forces cancel out and no work is performed for any virual displacement in accordance (!) with the motion constrains. 

### Generalized coordinates, conservative forces and energy conservation

Let us consider a mapping from the position space \\(r\\) to the generalized coordinates \\(q\\), i.e. \\(r = f(q)\\), so that the Jacobian matrix 

$$ \frac{\partial r}{\partial q} = \begin{pmatrix} \frac{\partial x}{\partial q_1} & ... & \frac{\partial x}{\partial q_n} \\\ \frac{\partial y}{\partial q_1} & ... & \frac{\partial y}{\partial q_n} \\\ \frac{\partial z}{\partial q_1} & ... & \frac{\partial z}{\partial q_n} \end{pmatrix} $$ 

has rank \\(n\\), i.e defines a regular matrix. This implies that there exists, at least locally, a regular map with a regular inverse. Such a map is bijective, smooth and differentiable. It defines a change of coordinates between the position coordinates and the generalized coordinates. Furthermore, the vectors \\( \frac{\partial r}{\partial q_i} \\) form a basis of the tangent space of the position space at the point \\(r\\). Its direction is corresponds to the coordinate line defined by \\(q_i\\). The coordinates \\(q\\) are the generalized coordinates of the system and correspond to its degrees of freedom. We write \\(x_i = f_{x_i}(q_1,...,q_n)\\), \\(y_i = f_{y_i}(q_1,...,q_n)\\) and \\(z_i = f_{z_i}(q_1,...,q_n)\\) for the position of the \\(i-th\\) particle in the generalized coordinates. Or, more generally \\(r_i = f_i(q_1,...,q_n)\\).   
  
The variation of the position vector \\(r\\) can be expressed as 

$$ \delta r = \sum_i \frac{\partial r}{\partial q_i} \delta q_i $$ 

D'Alemberts principle, expressed in generalized coordinates, becomes 

$$ \delta w = (F - ma)\delta r = \sum_i \frac{\partial r}{\partial q_i} (F - ma) \frac{\partial r}{\partial q} \delta q_i = 0. $$ 

The number of generalized coordinates is equal to the number of degrees of freedom of the system. For example, if a system can only move on a \\(2d-\\)surface, only two generalized coordinates are needed to locally express all possible values of \\(r\\). Let us denote 

$$ \delta w_F = \sum_j F_j \delta r_j = \sum_j \sum_i F \frac{\partial r_j}{\partial q_i} \delta q_i = \sum_i Q_i \delta q_i = \delta w_Q , $$ 

as the virtual work of the impressed forces described in terms of virtual displacements \\(\delta r\\) and \\(\delta q\\) respectively. From the relation above we obtain the generalized forces \\(Q_i = \sum_j F_j \frac{\partial r_j}{\partial q_i} \\).   
  
Important quantities are the kinetic energy \\(T\\) and potential energy \\(V\\) 

$$ T = \frac{1}{2} m(r,t) \dot{r}^2 + f_(r,t) \dot{r} + \frac{1}{2} g(r,t) $$ $$ = \frac{1}{2} m(r,t) \frac{d}{dt} r \frac{d}{dt} r + \sum f(r,t) \frac{d}{dt} r + \frac{1}{2} g(r,t) $$ $$ = \frac{1}{2} m(r,t) \frac{d}{dt} \sum_k \frac{\partial r}{\partial q_k} \sum_l \frac{d}{dt} \frac{\partial r}{\partial q_l}q_k q_l + \sum_l f(r,t) \frac{d}{dt} \frac{\partial r_i}{\partial q_l} q_l + \frac{1}{2} g(r,t) $$ $$ = \frac{1}{2} \sum_k\sum_l m(r,t) \frac{\partial r}{\partial q_k} \frac{\partial r}{\partial q_l} \dot{q_l} \dot{q_k} + \sum_l (r,t) \frac{\partial }{\partial q_l} \dot{q_l} + \frac{1}{2} g(r,t) $$ $$ =\frac{1}{2} \sum_k\sum_l a_{k,l}(q,t) \dot{q_k}\dot{q_l} + \sum_l b_l(q,t) \dot{q_l} + \frac{1}{2} c(q,t) = T_2 + T_1 + T_0 $$ 

and 

$$ V = \sum_i V_i = \sum_i V_i(q_1,...,q_n, \dot{q}_1,...,\dot{q}_n) $$ 

Often we consider only position dependent potential energies \\(V = V(q_1,...,q_n) \\) and only the \\(T_2\\) term of the kinetic energy. Systems where only these two energy components are considered give rise to special conservation laws. In position space we can write \\(T_2 = \frac{1}{2} \sum_i m_i v_i^2\\), where \\(v_i\\) is the velocity of the \\(i\\)-th particle. It is possible to describe the kinetic energy \\(T_2 \\) space as 

$$ T_2 = \frac{1}{2} \frac{d\bar{s}}{dt}^2 $$ 

where \\(\bar{s}^2 = 2Tdt^2 = \sum_i m_i (dx_i + dy_i + dz_i) = \sum_{l,k} a_{l,k} dq_l dq_k\\). This formulation relates the kinetic energy to Riemanian line elements with the mass matrix \\(a_{l,k}\\) as metric tensor. Note that the kinetic energy is a quadratic form in the velocities and that the force field is conservative (only depends on positions). Furthermore, both terms are independent from time. For more general systems, the conservative and time-independence does not hold. For these systems the line element \\( d\bar{s} \\) is defined in a more general manner than the Riemannian line element.   
  
Often the forces are derivativable from a scalar function \\( U = -V \\). Frequently this scalar function \\(V\\) is position (but not velocity) dependent. Then it is called a conservative potential energy. Conservative potential energies give rise to conservative forces via \\(F = -\nabla_r V \\) and \\(Q = -\nabla_q V \\) by derivation of the scalar function. Forces which are derivable from a single scalar function are called monogenic. Furthermore, if the scalar function is only position dependent, they are called conservative. Under the assumption of conservative forces we can write the work as 

$$ dw = Fdr - madr = -dV - madr = -dV - \frac{1}{2} \frac{d}{dt} mv^2 dt = -d(V + T) = 0, $$ 

which is the well known theorem of energy conservation for conservative systems. 

### The Lagrangian in improper and proper form

Consider \\(\delta r = \frac{\partial r}{\partial q} \delta q \\), then 

$$ \delta w = ( F\frac{\partial r}{\partial q} - ma\frac{\partial r}{\partial q} )\delta q = 0, $$ 

leads to 

$$ F\frac{\partial r}{\partial q} = ma\frac{\partial r}{\partial q} = Q_h $$ 

in equilibrium.   
Note the following two relations, 

$$ ma\delta r = m \frac{d}{dt} (v\delta r) = m \frac{d}{dt} (v \delta r ) - m v \frac{d}{dt} (\delta r) $$ 

and 

$$ \frac{\partial r}{\partial q} = \frac{\partial vdt}{\partial \dot{q}dt} = \frac{\partial v}{\partial \dot{q}} $$ $$ \frac{d}{dt}\frac{\partial r}{\partial q} = \frac{d}{dt} \frac{\partial vdt}{\partial \dot{q}dt} = \frac{\partial v}{\partial q} $$ 

which leads to 

$$ Q = ma \frac{\partial r}{\partial q} = m \frac{d}{dt} (v \frac{\partial r}{\partial q} ) - m v \frac{d}{dt} (\frac{\partial r}{\partial q} ) $$ $$ = m \frac{d}{dt} (v \frac{\partial v}{\partial \dot{q}} ) - m v \frac{d}{dt}\frac{\partial v}{\partial q} $$ $$ = \frac{d}{dt} \frac{\partial}{\partial \dot{q}} \frac{1}{2} m v^2 - \frac{\partial}{\partial q} \frac{1}{2} m v^2 $$ $$ = \frac{d}{dt} \frac{\partial T}{\partial \dot{q}} - \frac{\partial T}{\partial q}. $$ 

These are the Lagrange equations of motion in improper form. If \\(Q \\) is conservative, (i.e. \\(Q = -\nabla_q V = \frac{\partial V}{\partial q }\\) with \\(V\\) only position dependent), then 

$$ Q = \frac{\partial V}{\partial q } = \frac{d}{dt} \frac{\partial T}{\partial \dot{q}} - \frac{\partial T}{\partial q} = \frac{\partial V}{\partial q } + \frac{d}{dt} \frac{\partial V}{\partial \dot{q}} $$ 

Since \\(\frac{\partial V}{\partial \dot{q}} = 0\\) for velocity independent scalar functions fields, we obtain the Lagrangian 

$$ 0 = \frac{d}{dt} \frac{\partial T - V}{\partial \dot{q}} - \frac{\partial T - V}{\partial q} $$ $$ = \frac{d}{dt} \frac{\partial L }{\partial \dot{q}} - \frac{\partial L }{\partial q}. $$ 

If \\(Q = \frac{\partial V}{\partial q } + Q'\\) (i.e. it can not be described as a conservative force), this becomes 

$$ Q' = \frac{d}{dt} \frac{\partial L }{\partial \dot{q}} - \frac{\partial L }{\partial q} \frac{1}{2}, $$ 

where \\(Q'\\) is the non-conservative part of the generalized force.   
  
These equations are the Lagrange equations of motion in proper form. They have been derived from the generalized force and D'Alemberts principle. The generalized force is related to the kinetic energy of the system, through the improper form of the Lagrangian equations. If this force can be expressed as the gradient of a position-dependent scalar function, the system is conservative and we obtain the proper Lagrangian equations of motion. If the force can not be expressed fully as part of a potential energy, then the non-conservative force term is considered seperately. From d'Alemberts principle we can also obtain apparent forces in accelerated reference systems. This will provide some insights into conservative and non-conservative forces. 

## Apparent Forces via D'Alemberts Principle

For this section we consider two reference systems which give rise to distinct, but equivalent coordinates \\( r \\) and \\( r' \\). Let \\( r \\) correspond to a fixed system with origin at zero and \\( r' \\) to a moving system whose origin is denoted by \\( c \\) . We might equally well use generalized coordinates \\(q\\) and \\(q'\\) for the two systems. Consider the relations 

$$ r = r' + c , $$ $$ \ddot{r} = \ddot{r'} + \ddot{c} $$ 

leading to the Newtonian forces 

$$ F' = ma' = m \ddot{r'} = m \ddot{r} - m \ddot{c} . $$ 

Here \\( m \ddot{c} \\) corresponds to the inertial forces caused by the movement of the system and \\( \ddot{r'} \\) corresponds to the movement itself. Correspondingly, \\( m \ddot{r} \\) are the inertial forces measured from the fixed system. The term \\( m \ddot{c} \\) is called the einstein force. The Einstein force is an apparent force in the moving system, i.e. its forces are not 'impressed' from its point of view. Imagine being on a carussel and rolling a ball. The ball will deviate from a straight line, even though no force is acting on it. Consider now that a rotating reference system´whose origin overlaps with the fixed origin. For a vector \\(B\\) viewed as fixed from the rotating system we have the relations 

$$ r = r' , $$ $$ dB = (\Omega \times B) dt , $$ 

where \\(\Omega\\) is the angular velocity vector of the rotating system. Thus 

$$ \frac{dB}{dt} = \Omega \times B , $$ 

Denote by \\( \frac{dB'}{dt} \\) the change of \\(B\\) viewed from the moving system. Then 

$$ \frac{dB}{dt} = \Omega \times B + \frac{dB'}{dt}. $$ 

So we obtain 

$$ v = \Omega \times r + v' = \Omega \times r' + v' . $$ 

Now carefully consider the acceleration 

$$ \frac{dv}{dt} = \Omega \times v + \frac{d ( \Omega \times r' + v') }{dt} $$ $$ = \Omega \times v + \dot{\Omega} \times r' + \Omega \times \frac{dr'}{dt} + \frac{dv'}{dt} $$ $$ = \Omega \times (\Omega \times r') + \Omega \times v' + \dot{\Omega} \times r' + \Omega \times v'+ \frac{dv'}{dt} $$ $$ = \Omega \times (\Omega \times r') + 2 \Omega \times v' + \dot{\Omega} \times r' + \frac{dv'}{dt}. $$ 

By multiplication with the mass \\( m \\) we obtain the inertial forces 

$$ I = I' + 2m \Omega \times v' + m\Omega \times ( \Omega \times r') + m \dot{\Omega} \times r'. $$ 

We write \\( B = 2m \Omega \times v' = 2mv' \times \Omega \\) and \\( C = m\Omega \times ( \Omega \times r') = m \omega^2 r_{\perp}' \\) and \\( K = \dot{\Omega} \times r' \\). The centrifugal force \\(C\\) is derivable from the position dependent scalar function \\( \Phi_C = \frac{1}{2} m \omega^2 r_{\perp}^2 \\) and is thus a conservative force. The term \\( B \\) corresponds to the corriolis force. It is not derivable from a position dependent scalar function and is thus non-conservative. However, it is derivable from a scalar function \\( \Phi_B = m v' \times \Omega r' \\) which is linear in the velocities as \\( B = \frac{d}{dt} \frac{ \partial \Phi_B}{\partial \dot{r} } - \frac{\partial \Phi_B}{\partial r} \\). Thus it is nonetheless monogenic. The term \\( K \\) is the euler force, it arises only for accelerated rotations. The term \\( I' \\) is the inertial force in the moving system. The total effective force is 

$$ F_{total} = F + I' + C + B + K $$ 

where \\(F\\) is the impressed force onto the particle. 

## From D'Alemberts principle to the Lagrangian by variational considerations

D'Alemberts principle is a powerful statement which determines the equations of motion for a system. However, it is not in the form of a definite integral. The advantage of a formulation as stationary value of a definite (meaning a bounded) integral lies in the analytical solution which can be obtained. 

### The stationary value of a definite integral 

The mathematical form of a definite integral is 

$$ I = \int F(y,y',x)dx $$ 

where \\( F(y,y',x) \\) is any function of the three variables \\( y,y',x \\) and \\(y = f(x)\\). By the calculus of variation, one can solve analytically for the stationary value of this integral. It is also possible if \\( F(y,y',y'',x) \\) contains \\( 4 \\) variables including the second derivative. However, we will not consider this case. We want to find \\(y = f(x)\\) which makes the integral an extremum (or at least stationary) with boundary conditions \\( f(a) = \alpha \\) and \\( f(b) = \beta \\). For problems of this kind variational calculus is a powerful tool.   
  
Consider \\( z_j = \frac{\Delta y}{\Delta x}|_{x=x_j} =\frac{ y_{j+1} - y_{j} }{ x_{j+1} - x_{j} } \\). In this discrete formulation the integral becomes a sum 

$$ I = \sum_j F(y_j, z_j, x_j) \Delta x_j = \sum_j F(y_{j+1}, z_j, x_j) \Delta x_j $$. 

The last equality follows from the fact that in the limit \\( y_{j+1}=y_{j} \\). Partial differentiation w.r.t \\(y_{k+1}\\) (Note that \\(y_{k+1}\\) appears with coefficient \Delta \Delta x_{k} and \Delta x_{k+1}\ )leads to 

$$ \frac{\partial F}{\partial y} (x_{k+1} - x_{k}) + \frac{\partial F}{\partial y'}|_{x=x_{k}} -\frac{\partial F}{\partial y'}|_{x=x_{k+1}} $$ $$ = \frac{\partial I}{\partial y} $$ 

which through multiplication by \\( \frac{d}{dx} = \lim_{\Delta x \rightarrow 0} \frac{1}{\Delta x} \\) leads to 

$$ \frac{d}{dx} \frac{\partial I}{\partial y} = \frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} $$. 

Thus the stationary value of a definite integral is found when 

$$ \frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} = 0 $$. 

This is a necessary and sufficient condition. In this explanation we have considered \\(x\\) as the independent variable. In the Lagrangian formulation of mechanics, often time is the independent variable. However, extensions where time is considered as a position variable are possible. In these cases another independent variable \\(\tau\\) is required and all positional variables (including time) are considered functions of \\(\tau\\).   
  
Consider 

$$ \delta I = \int \delta F(y,y',x) dx = \epsilon \int \frac{\partial F}{\partial y}\Phi(x) + \frac{\partial F}{\partial y'}\Phi'(x) dx $$ 

where \\( \delta F(y,y',x) = F(y + \epsilon \Phi,y'+ \epsilon \Phi', x) - \delta F(y,y',x) = \epsilon (\frac{\partial F}{\partial y}\Phi + \frac{\partial F}{\partial y'}\Phi' ) \\) and \\(\Phi = \Phi(x)\\) is any arbitrary function of \\(x\\). Thus 

$$ \frac{\delta I}{\epsilon} = \int \frac{\partial F}{\partial y}\Phi + \frac{\partial F}{\partial y'}\Phi' dx $$ 

and through integration by parts 

$$ \int \frac{\partial F}{\partial y'}\Phi' dx = [ \frac{\partial F}{\partial y'}\Phi]_{t_1}^{t_2} - \int \frac{d}{dx} \frac{\partial F}{\partial y'}\Phi dx $$ 

we obtain 

\\[ \frac{\delta I}{\epsilon} = \int ( \frac{\partial F}{\partial y} - \frac{d}{dx} \frac{\partial F}{\partial y'} ) \Phi dx + [ \frac{\partial F}{\partial y'}\Phi]_{t_1}^{t_2} \\] 

This equation gives the variation of the integral for arbitrary variations of form \\( \delta f(x) = \epsilon \Phi(x) \\). The boundary term depends only on the end-points. Often, one considers the variation to be zero at the end-points. Vanishing of the definitite integral for all first order variations is a necessary and sufficient condition for a stationary value, indicating an extremum or saddle-point. For an extremum classification, the second variation must be examined. The stationary value of a functional leads to the lagrangian equations of motions through consideration of the action functional. The action was first introduced by Pierre Louis Maupertuis in 1740 and subsequently studied by Euler, Lagrange and Bernoulli. It allows a perspective unifying classical mechanics and optics through the principle of least action.   
  


### The action functional

D'Alemberts principle can be transformed into such a definite integral (the action) by integration with respect to time. Consider the time integrated work function with a conservative force 

$$ \delta \int w dt = \int ( F\delta r- m\frac{d}{dt}v\delta r ) dt $$ 

For the term corresponding to the impressed forces we have 

$$ \int F \delta r dt = \int -\nabla_r V \delta r = \int \delta V dt $$ 

For the term corresponding to the inertial forces we have (through integration by parts) 

$$ \int \frac{d}{dt}(mv) \delta r dt = -[mv \delta r]_{t_1}^{t_2} + \int m v\frac{d}{dt} \delta r dt $$ $$ = -[mv r]_{t_1}^{t_2} + \int m v \delta \frac{d}{dt} r dt $$ $$ = -[mv \delta r]_{t_1}^{t_2} + \int m v \delta v dt $$ $$ = -[mv \delta r]_{t_1}^{t_2} + \frac{1}{2} \int m \delta v^2 dt $$ $$ = -[mv \delta r]_{t_1}^{t_2} + \frac{1}{2} \delta \int m v^2 dt $$ $$ =-[m \frac{d}{dt} r \delta r] + \delta \frac{1}{2} \sum_k\sum_l a_{k,l}(q,t) \dot{q_k}\dot{q_l} $$ $$ =-[ \sum_l m \frac{\partial r}{\partial q_l} \frac{d}{dt} q_l \delta r] + \delta \frac{1}{2} \sum_k\sum_l a_{k,l}(q,t) \dot{q_k}\dot{q_l} $$ $$ =-[ \sum_l m \dot{q_l} \frac{\partial r}{\partial q_l} \sum_k \frac{\partial r}{\partial q_k} \delta q_k ] + \delta \frac{1}{2} \sum_k\sum_l a_{k,l}(q,t) \dot{q_k}\dot{q_l} $$ $$ = -[ \sum_l \sum_k m \frac{\partial r}{\partial q_l}\frac{\partial r}{\partial q_k} \dot{q_l} \delta q_k ] + \delta T $$ $$ = -[ \sum_k \sum_l a_{k,l} \dot{q_l} \delta q_k ] + \delta T $$ $$ = -[ \sum_k \frac{\partial T} {\partial \dot{q_k}} \delta q_k] + \delta T $$ $$ = -[ \sum_k \frac{\partial L} {\partial \dot{q_k}} \delta q_k] + \delta T $$ $$ = -[ \frac{\partial L} {\partial \dot{q}} \delta q] + \delta T $$ 

Thus we obtain 

$$ \int \delta w dt = - \delta \int V dt + [mv \delta r]_{t_1}^{t_2} + \delta \int T dt $$ $$ = \delta \int T-V dt - [mv \delta r]_{t_1}^{t_2} $$ $$ = \delta \int L dt - [\frac{ \partial L }{ \partial \dot{q} } \delta q]_{t_1}^{t_2} $$ 

Summarizing, we have 

$$ \delta \int w dt = \delta \int L(r,r') dt - [mv \delta r]_{t_1}^{t_2} = \delta \int L dt \+ [\frac{\partial L}{\partial \dot{q}} \delta q ]_{t_1}^{t_2} . $$ 

If we vary between definite limits the boundary term vanishes because at the limits the variation is zero. Then we can reformulate the principle of d'Alembert as the stationary value of a definite integral. A particle in motion is in equilibrium if 

$$ \int \delta w dt = \delta \int L dt = \delta A = 0 $$ 

where \\( A \\) is called the action functional. This is Hamiltions principle of least action. Here we have assumed that the work done by the forces is derivable from a single scalar function which is only positon dependent. The impressed forces must be monogenic and even conservative. Note, that also non-conservative and time-dependant forces can be monogenic. For polygenic forces the transformation of d'Alemberts principle into a definite integral is not possible. By the calculus of variation for the stationary value of definite integrals we obtain the necessary and sufficient condition for stationarity 

$$ \frac{\partial L}{\partial q_i} - \frac{d}{dt} \frac{\partial L}{\partial q_i'} = 0 $$. 

These are the Lagrangian equations of motion, obtained from time integration of d'Alemberts principle by imposing stationarity on the action functional. A strong side of these equations is that they do not contain the accelerations, but only the velocities. The equations of motion are of first order in time, not second order and completely determined by a scalar function \\(L(q,q')\\). In the more general case where variations at the limits are allowed, the boundary term must be considered and we have 

$$ \delta \int L dt = [\frac{\partial L}{\partial \dot{q}} \delta q ]_{t_1}^{t_2} .$$ 

### Lagrangian multipliers - Motion constrains as potential energies 

Often we consider systems which are constrained in their motion. This constraint is enforced by the reactive forces \\(R\\), which are assumed to be ideal w.r.t allowed variations (i.e. orthogonal to all possible motions). Ideal reactive forces are conservative and can be derived from a scalar function \\(U\\), which is only position dependent. They can only represent holonomic constraints, i.e. constraints which are only dependent on the position of the system. To show how the reactive forces can be derived from a scalar function, consider the constraint \\(f(q) = 0\\), representing a surface or line on which the motion occurs. Due to the constraint, \\(\delta q \\) can not be choosen freely. Consider 

$$ \delta f(q) = \frac{\partial f(q_1)}{\partial q_1} \delta q_1 + ... + \frac{\partial f(q_n)}{\partial q_n} \delta q_n = 0, $$ 

which holds always due to \\( f(q) = 0 \\). We write the new action as 

$$ \delta A = \delta \int L dt + \int \lambda \delta f(q) dt , $$ 

which is permissible since \\( \delta f(q) = 0 \\). By the variational calculus, the stationary value of \\(A\\) is found when 

$$ \frac{\partial L}{\partial q_i} - \frac{d}{dt} \frac{\partial L}{\partial q_i'} + \lambda \frac{\partial f}{\partial q_i} = 0 $$ 

for all \\(i\\). We have an additional parameter \\(\lambda\\) which corresponds to the motion constraint. For each of the \\(m\\) constraints, we have one \\(\lambda\\) value. The \\(\lambda\\) are choosen so that the coefficients of \\(m\\) position coordinates \\(q_i\\) vanish. Then the variations \\(\delta q \\) can be chosen freely and the equations of motion fulfill the constraints. The physical interpretation of \\( \lambda \frac{\partial f}{\partial q_i} \\) is that of the reactive force enforcing the constraint. Note, that this force is indeed orthogonal to the motion surface \\(f(q) = 0 \\). The terms \\( \lambda_l f_l(q) \\) correspond to potential energies which give rise to the conservative and ideal reactive forces that enforce the kinematical constraints.   
  
This shows how positional constrains can be incorporated into the action integral as potential energies. 

### Conservation of energy

If we consider the special variation \\(\delta q = \epsilon \dot{q} \\), allowing for variations at the limits, we obtain the relation 

$$ \delta \int w dt = \int \epsilon \dot{L}(q_1,...,q_n; \dot{q_1},...,\dot{q_n}) dt - \epsilon [ \sum_i \frac{\partial L}{\partial \dot{q_i}} \dot{q_i}]_{t_1}^{t_2} = 0 $$ 

which leads to 

$$ [ L(q_1,...,q_n; \dot{q_!},...,\dot{q_n}) ]_{t_1}^{t_2} - [\frac{\partial L}{\partial \dot{q}} \dot{q}]_{t_1}^{t_2} = 0. $$ 

Since \\(t_1\\) and \\(t_2\\) are arbitrary and independent, we obtain the condition 

$$ L - \sum_i p_i \dot{q}_i = const. $$, 

where \\( p_i = \frac{\partial L}{\partial \dot{q}_i} \\).   
  
If \\( T = \frac{1}{2} \sum_{i,k} a_{i,k} \dot{q}_i \dot{q}_k \\) and \\(V\\) is indepenend from \\(\dot{q}_i\\), then \\(p_i = \frac{ \partial T}{ \partial \dot{q}_i}\\) and \\( \sum_i p_i \dot{q}_i = 2T \\). The equation above becomes 

$$ 2T - ( T - V) = T + V = 0 $$ 

which is the law of the conservation of energy. Note that we required \\(L\\) to be independent from time, \\(T\\) to be of quadratic form in the velocities and \\(V\\) to be independent from the velocities \\(\dot{q}_i\\). In general relativity the kinetic energy depends on the velocities in more complicated ways. When we encounter gyroscopic terms the potential energy also depends on the velocities. However the conservation law 

$$ L - \sum_i p_i \dot{q}_i = const. = E $$ 

holds even for these systems, as long as they are time-independent. The interpretation of this constant of motion is the energy of the system. 

### Elimination of variables in the Lagrangian

We have introduced the quantities \\(p_i = \frac{\partial L}{\partial \dot{q}_i}\\) (called momentum) so that \\(\dot{p_i} = \frac{\partial L}{\partial q_i}\\). If \\(q_i\\) is an ingorable variable, then \\(\dot{p}_i = 0 \\) and \\(p_i = const.\\). The momentum associated with the ignorable variable is preserved. Consider the lagrangian, where only the variation of non-ignorable variables is zero at the limits 

$$ \delta \int L(q,\dot{q}) dt = [ \sum_i \frac{ \partial L }{ \partial \dot{q}_i } \delta q_i ]_{t_1}^{t_2} = [ p_n \delta q_n]_{t_1}^{t_2} = p_n \delta \int \dot{q_n} dt = \delta \int p_n \ ot{q_n} dt $$. 

This leads to 

$$ \delta \int (L - p_n q_n') dt = 0. $$ 

We call 

$$ \tilde{L} = L - p_n q_n' = L - c_n q_n' $$ 

the modified lagrangian. The modified action integral is \\(A = \int \tilde{L} dt \\). If there is a kinetic coupling between ignorable and non-ignorable variables, we obtain additional terms which can be interpreted as gyroscopic energy terms (linear in velocities). 

### Time as ignorable variable - Energy as negative momentum associated to time 

Up until now we have considered the Lagrangian to be independent from time. Here we will consider a Lagrangian which has time as an ignorable variable. Up until now we considered the time \\(t\\) as the independent variable. Now we will consider \\(q_i\\), \\(q_i'\\), \\(t\\) as functions of some independent parameter \\( \tau \\). In essence \\(t\\) becomes one of the position variables. The notation \\(q_i' = \frac{dq_i}{d \tau}\\) denotes the derivative with respect to \\( \tau \\). Then we have 

$$ A = \int_{t} L(q_1,...,q_n; \dot{q}_1, ..., \dot{q}_n) dt = \int_{\tau} L(q_1,...,q_n; \frac{q_1'}{t'},...,\frac{q_n'}{t'}) t' d\tau . $$ 

It follows 

$$ p_t = \frac{\partial L t'}{\partial t'} = L - \sum_{i}^{n} \frac{\partial L}{\partial \dot{q_i}} \frac{q_i'}{t'^2}t' = L - \sum_{i}^{n} p_i \dot{q_i} . $$ 

If \\(t\\) is ignorable (i.e. the system is conservative), then \\(p_t = const. = -E\\). The modified Lagrangian is 

$$ \tilde{L} = Lt' - p_t t' = (L - p_t)t' = \sum_{i}^{n} p_i \dot{q_i} t' . $$ 

leading to the action integral 

$$ A = \int_{\tau} \sum_{i}^{n} p_i \dot{q_i} t' d\tau = 2 \int_{\tau} T t' d\tau $$ 

The kinetic energy can also be written in configuration space as 

$$ T = \frac{1}{2} \frac{d\bar{s}}{dt}^2 $$ 

with \\( \bar{s}^2 = 2T dt^2 = \sum_i m_i (dx_i^2 + dy_i^2 + dz_i^2) \\), which, by introduction of \\(\tau\\) as independent variable leads to 

$$ T = \frac{1}{2} \left( \frac{d\bar{s}}{d\tau} \right)^2 \frac{1}{t'^2}. $$ 

From this we obtain 

$$ t' = \frac{1}{\sqrt{2(E-V)}} \frac{d\bar{s}}{d\tau} = \frac{1}{\sqrt{2T}} \frac{d\bar{s}}{d\tau} . $$ 

leading to the action integral 

$$ A = \int_{\tau} \sqrt{2(E-V)} \frac{d\bar{s}}{d\tau} d\tau = \int_{\tau} \sqrt{2(E-V)} d\bar{s} . $$ 

Note that the free parameter is \\( \tau \\) (and not \\( \bar{s} \\) ). Some parameter must be chosen as free parameter. For example it might correspond to \\( q_n \\), giving all other \\( q_i \\) as functions of \\( q_n \\). Minimization of this integral to find the path of the particle is Jacobi's principle. It relates the stationary action functional to the motion along some geodesic of a given Riemannian space. Here we considered a time-dependent work function. The time does not appear in the equations, but can be solved for by the given relation \\( t' = f(\tau)\\). If we restrict ourselves to a single particle, then \\( d\bar{s} \\) is the (curvi-linear) line element of ordinary three-dimensional space. Jacobi's principle then represents a notable analogy to Fermats principle of least time in optics which minimizes 

$$ I = \int_{\tau} n d\bar{s}. $$ 

Here \\(n\\) is the refractive index of the medium which, much like \\( \sqrt{2(E-V)} \\), can change at different spatial positions. This unifying principle between optics and mechanics was first introduced by Pierre Louis Maupertuis in 1740 and further developed by Hamilton in his optico-mechanical theory. Remarkably, both principles solve for paths in space by minimization of an integral quantity. In mechanics this integral quantity is the action, in optics it is the travel time. 

### Small Vibrations about a state of equilibrium - Linearization about a stationary point

Consider the kinetic energy 

$$ K = \frac{1}{2} \sum_{i,k} a_{i,k} \dot{q}_i \dot{q}_k = \frac{1}{2} \dot{q}^T A \dot{q} \ $$ 

and a Taylor-expansion of the potential energy about a state of equilibrium 

$$ V = V(q_0) + \sum_{i} \frac{\partial V}{\partial q_i}(q_i - q_{0,i}) + \sum_{i} \frac{1}{2} \frac{\partial^2 V}{\partial q_i^2} (q_i - q_{0,i})^2 + O(|3|) $$. 

Since, for monogenic forces derivable from \\(V\\), at the state of equilibrium \\( \frac{\partial V}{\partial q_i} = 0\\) (by the vanishing of the virtual work in directions tangent to the constraint and ideality of constraints) for all \\(i\\) and \\(V(q_0) = const. \\) , we obtain 

$$ V = \frac{1}{2} \sum_{i,k} \frac{\partial^2 V}{\partial q_i \partial q_k} q_iq_k = \frac{1}{2} \sum_{i,k}b_{i,k}q_i q_k = \frac{1}{2} q^T B q $$. 

We obtain the linearized Lagrangian 

$$ L = \frac{1}{2} \dot{q}^T A \dot{q} - \frac{1}{2} q^T B q $$ 

with equations of motion (i.e. minimum of the action integral) 

$$ A \ddot{q} + B q = 0 $$. 

These equations are linear homogenous and the superposition principle holds. Since \\(A\\) is positive definite, it is possible to diagonalize both matrices simultaneously. In fact \\(A\\) can be diagonalized to the identity. The eigenvalues are the roots of 

$$ det(B - \lambda A) = 0 $$ 

and the eigenvectors are so that \\( u^{(i)} A u^{(j)} = \delta_{i,j} \\) and \\( (U^T B U)_{i,j} = \lambda_i \delta_{i,j}\\). By a change of coordinates \\( \bar{q} = Uq \\) we obtain the diagonalized equations of motion 

$$ \bar{A} \ddot{\bar{q}} + \bar{B} \bar{q} = 0 $$ 

, where \\(\bar{A} = I \\) and \\(\bar{B} = U^T B U = \text{diag}(\lambda_1,...,\lambda_n)\\). We know that for a differential equation of form 

$$ \ddot{x} = - \lambda x $$ 

the solution is 

$$ x(t) = \cos(\omega t + \Phi) $$ 

with \\(\omega = \sqrt{\lambda}\\) since \\( \frac{d^2}{dt^2} \alpha\cos(\omega t + \Phi ) = - \alpha \omega^2 \cos(\omega t + \Phi) \\). Thus, the general solution for the linearized equations of motion is 

$$ \bar{q}_i(t) = \sum_{i} \alpha_i \cos(\omega_i t + \Phi_i) u_i $$, 

where \\( \bar{q}_i(t) \\) caputres the time-dependent motion of the system in the direction of the eigenvector \\(u_i\\) with eigenvalue \\(\lambda_i = \sqrt(\omega_i)\\). We can obtain the general solution for the original coordinates \\(q_i(t)\\) by the inverse transformation \\(q = U^T \bar{q}\\). 

## Towards the Hamiltionian: Canonical transformations and Phase Space

By applying a Legendre transform to the Lagrangian we can obtain the Hamiltonian. A Legendre transform assumes as starting point any function \\( F(q_1,....,q_n) \\) and introduces the new variable \\( v_i = \frac{\partial F}{\partial u_i}\\). The new function is then \\( G = \sum_i v_i u_i - F = G(v_1,...,v_n) \\). Let us examine the variation of the new function 

$$ \delta G = \sum_i \frac{\partial G}{\partial v_i} \delta v_i $$ $$ = \sum_i (u_i \delta v_i + v_i \delta u_i) - \delta F $$ $$ = [u_i \delta v_i + (v_i - \frac{\partial F}{\partial u_i}) \delta u_i] $$ $$ = u_i \delta v_i $$ 

where the last equality follows from the definition of the new variable \\(v_i\\). We obtain 

$$ u_i = \frac{\partial G}{\partial v_i} \text{ and } v_i = \frac{\partial F}{\partial u_i} $$ $$ G = \sum_i v_i u_i - F \text{ and } F = \sum_i u_i v_i - G $$ $$ G(v_1,...,v_n) \text{ and } F(u_1,...,u_n) $$ 

This transformation is symmetrical. Not all variables need to participate in the transformation. For variables which do not participate, we have by consideration of the variation 

$$ \frac{\partial F}{\partial w_i} = -\frac{\partial G}{\partial w_i}. $$ 

Application to the Lagrangian leads to 

$$ p_i = \frac{\partial L}{\partial \dot{q}_i} \text{ and } \dot{q}_i = \frac{\partial H}{\partial p_i} $$ $$ H = \sum_i p_i \dot{q}_i - L \text{ and } L = \sum_i p_i \dot{q}_i - H $$ $$ H(q_1,...,q_n, p_1,...,p_n;t) \text{ and } L(q_1,...,q_n, \dot{q}_1,...,\dot{q}_n;t) $$ 

with 

$$ p\frac{\partial H}{\partial q_i} = -\frac{\partial L}{\partial q_i} = -\dot{p_i}. $$ 

Note that \\(H\\) is of first order in all its variables, which allows for the interpretation of \\(p\\) and \\(q\\) as position variables. We can write the equations of motion in the Hamiltonian form 

$$ \dot{q}_i = \frac{\partial H}{\partial p_i} $$ $$ \dot{p}_i = - \frac{\partial H}{\partial q_i}, $$ 

with the equations arising from the properties of the legendre transform. Since \\(q_i\\) and \\(p_i\\) are positional variables, the Hamiltonian equations of motion only contain time on the left-hand side. This is opposed to the Lagrangian equations of motion which also contain velocities \\(\dot{q}_i\\). Similar to how the dependency on velocities might be seen as an improvement to the dependency on accelerations in a Newtonian formalism. the dependency on positions in a Hamiltonian formalism can be considered an improvement to the dependency on velocities in a Lagrangian formalism. Remember that the coefficient of \\(\delta p_i \\) is zero due to the legendre transform. Thus we may write 

$$ \delta A = \int \delta L dt = \int \delta (\sum_i p_i \dot{q}_i - H) dt $$ 

which is minimized by 

$$ \frac{d}{dt} \frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = \frac{dp_i}{dt} - \frac{\partial H}{\partial q_i} = 0 $$ $$ \frac{d}{dt} \frac{\partial L}{\partial \dot{p}_i} - \frac{\partial L}{\partial p_i} = 0 - \dot{q}_i + \frac{\partial H}{\partial p_i} = 0 $$ 

which gives rise to the equation of motion in Hamiltonian and Lagrangian form. The transformation into \\(2n\\) positional space is powerful since the equations of motion define a velocity field in phase space. The behaviour of the motion in this space is similar to that of an incompressible fluid. In incompressible fluids, certain integral invaraits are preserved, for example the circulation and the volume. It is a property of the phase space that trajectories do not cross. They can be considered rays.   
  


### Ignorable variables

The elimination of ignorable variables is particularily simple in the Hamiltonian form. By inspection of the adapted Lagrangian and the definition of the Legendre transform, it becomes apparent that ignorable variables and their momenta are simply dropped from the Hamiltonian. 

## Extended phase space - time as mechanical variable

Given the notion of the phase space we can include time as a position variable in the phase space and consider an independent parameter \\( \tau \\) for path parameterization. We have done something similar above when considering the time-dependent Lagrangian. Some free parameter \\( \tau \\) was chosen to be the independent variable. All other variables, including time, were considered as functions of \\( \tau \\). We call the phase space including time the extended phase space. Time is simply regarded as a positional variable \\( t = q_{n+1} \\) with an associated momentum \\( p_t = p_{n+1} \\). We can write the extended action integral 

$$ A= \int L(q_1,...q_n,q_{n+1}, q_1',...,q_{n}') q_{n+1}' d\tau = \int L_1 d\tau $$ 

where \\( q_i' = \frac{q_i}{d \tau} \\) so that \\( A = \int L dt = \int L_1 d\tau\\). We call the Lagrangian \\(L_1\\) the extended Lagrangian. Often \\( \tau \\) is chosen so that \\( q_{n+1}' = \frac{dq_{n+1}}{d \tau} = \frac{dt}{d \tau}\\).   
Let \\( q_i' \rightarrow \lambda q_i'\\) for all \\(i = 1,...,n+1\\). For this transformation it can be shown that \\(L_1 \rightarrow \lambda L_1\\) and thus the action integral is homogenous of first order in the variables \\(q_i\\). By Eulers theorem on homogenous functions we have 

$$ \sum_{i=1}^{n+1} \frac{\partial L_1}{\partial q_i'} q_i' = L_1 $$ 

so that (by consideration of the legendre transform) we have 

$$ \sum_{i=1}^{n+1} \frac{\partial L_1}{\partial q_i'} q_i' - L_1 = H_1 = 0 $$ 

which allows for a representation of the extended action integral as 

$$ A = \int \sum_{i=1}^{n+1} \frac{\partial L_1}{\partial q_i'} q_i' d\tau = \int \sum_{i=1}^{n+1}p_i q_i' d\tau. $$ 

Here the extended Hamiltonian \\(H_1\\) vanishes. It is however replaced by an auxiliary condition which captures the relation between the \\(q_i\\) and \\(p_i\\). We denote this relationship by 

$$ K(q_1,...,q_{n+1}; p_1,....,p_{n+1}) = 0. $$ 

For example, we can single out the variable \\(p_{n+1}\\) and obtain 

$$ p_{n+1} q_{n+1}' + \sum_{i=1}^{n} \frac{\partial L_1}{\partial q_i'} q_i' - L_1 = H_1 = 0 $$ 

so that 

$$ p_{n+1} q_{n+1}' = - \sum_{i=1}^{n} \frac{\partial L_1}{\partial q_i'} q_i' + L_1 = - \sum_{i=1}^{n} p_i \dot{q_i} + L_1 = - H(q_1,...,q_{n+1}, p_1,...,p_n) $$ 

(by considering the legendre transform). If we choose \\(\tau\\) so that \\(q_{n+1}' = \frac{dq_{n+1}}{d\tau} = 1\\) we obtain \\( p_{n+1} = - H(q_1,...,q_{n+1}, p_1,...,p_n) \\) which inserted into the extended action integral gives 

$$ A = \int \sum_{i=1}^{n} p_i \dot{q_i} - H(q_1,...,q_{n+1}, p_1,...,p_n) dq_{n+1}. $$ 

However, any other functional relationship between the \\(q_i\\) and \\(p_i\\) is possible. Thus we use the more general form \\(K = 0\\). The integral \\(A\\) is to be made stationary under this constraint. Since the constraint depends only on the position variables (i.e. it is a holonomic constraint), it can be incorporated by the \\(\lambda\\) method, giving rise to additional potentials with corresponding conservative forces. By adding the time to the phase space, even time-dependant systems can be considered conservative systems. We can incorporate the constraint into the integral by the \\(\lambda\\) method and obtain 

$$ \bar{A} = \int \sum_{i=1}^{n+1}p_i q_i' - \lambda K d\tau. $$ 

By proper choice of \\(\tau\\) we can set \\( \lambda = 1\\) and obtain the form 

$$ \bar{A} = \int \sum_{i=1}^{n+1}p_i q_i' - K d\tau. $$ From the stationary value of a definite integral we obtain the equations of motion 

$$ q_k' = \frac{\partial K}{\partial p_k} $$ $$ p_k' = - \frac{\partial K}{\partial q_k}. $$ If \\(K = p_{n+1} + H = p_t + H\\), then the equations of motion are 

$$ q_k' = \frac{\partial H}{\partial p_k} $$ $$ p_k' = - \frac{\partial H}{\partial q_k} $$ $$ q_{n+1}' = 1 $$ $$ p_{n+1}' = - \frac{\partial H}{\partial q_{n+1}} $$ These are the extended Hamiltonian equations of motion. Time is regarded as a positional variable, so that even time dependent systems can be treated with holonomic constraints and conservative forces arising from the motion constraint defined by \\(K\\). This constraint is only position dependant and thus interpretable as a potential energy. The derivation of the extended equations of motion is enabled by the introduction of a new independent parameter \\(\tau\\) which determines the phase-space evolution. This independent parameter can be chosen so that \\(q_{n+1}' = \frac{dq_{n+1}}{d\tau} = 1 \\). 

## Canonical Transformations

The equations derived up until here are not necessarily easy to solve. Transformations can help to simplify the equations. A transformation is called canonical if it preserves the action integral. This is required, so that the equations of motion are also preserved. Consider the extended action integral 

$$ A = \int \sum_{i=1}^{n+1} \frac{\partial L_1}{\partial q_i'} q_i' d\tau = \int \sum_{i=1}^{n+1}p_i q_i' d\tau. $$ 

which is to be minimized under the constraint \\(K = 0\\). The action is preserved if a point-transformation satisfies \\( \sum__{i=1}^{n+1} p_i \partial q_i = \sum__{i=1}^{n+1} P_i \partial Q_i \\). We can express the constraint as 

$$ p_{n+1} + H = P_{n+1} + H' = K = 0 $$. 

For the extended phase space we have 

$$ \dot{q}_i = \frac{\partial H}{\partial p_i} + p_t \partial t $$ $$ \dot{p}_i = - \frac{\partial H}{\partial q_i} + p_\bar{t} \partial \bar{t} $$ 

and 

$$ H' = H + p_t - p_\bar{t} $$ 

. The invariance principle \\( \sum_i p_i \partial q_i = \sum_i P_i \partial Q_i \\), requires at least one additional functional relation \\( f(q_1,...,1_n, Q_1,...,Q_n) = 0 \\), else all the \\(p_i\\) and \\(P_i\\) would vanish due to the independent arbitrary variations. We can have up to \\(n\\) such constraints, leading to more or less restricted forms. By the Lagrangian \\(\lambda\\) method we can incorporate the constraints 

$$ \sum_i p_i \partial q_i - \sum_i P_i \partial Q_i = \lambda_1 \partial f_1 + ... + \lambda_n \partial f_n $$ 

allowing for arbitrary variations of the \\(q_i\\) and \\(Q_i\\). We obtain 

$$ p_i = \lambda_1 \frac{\partial f_1}{\partial q_i} + ... + \lambda_n \frac{\partial f_n}{\partial q_i} $$ $$ P_i = -\lambda_1 \frac{\partial f_1}{\partial Q_i} - ... - \lambda_n \frac{\partial f_n}{\partial Q_i} $$. These equations characterize the transformation. If we add time to the position variables, we obtain one further condition 

$$ t = \bar{t} $$ and 

$$ p_t - p_\bar{t} = \lambda_1 \frac{\partial f_1}{\partial t} + ... + \lambda_n \frac{\partial f_n}{\partial t} $$ so that, in extended phase space 

$$ H' = H + \lambda_1 \frac{\partial f_1}{\partial t} + ... + \lambda_n \frac{\partial f_n}{\partial t} $$ \sum p_i \partial q_i = \sum P_i \partial Q_i + \partial S where \\(S = \lambda_1 f_1 + ... + \lambda_n f_n \\) is a differential of a certain function \\(S\\). The transformed action integral (we consider time independence and actual displacements) is then 

$$ A = \int_{t} \sum_i p_i dq_i - H' dt = \int \sum_i P_i dQ_i - Hdt + \int_{t} dS $$ , the addition of the last term is constant and does not change the stationary value of the action integral. We have the relation 

$$ \sum_i p_i \delta q_i - \sum_i P_i \delta Q_i = \delta S = \sum ( \frac{\delta S}{\delta q_i} \delta q_i + \frac{\delta S}{\delta Q_i} \delta Q_i) $$. If we inculde \\(t\\) in the phase space, we obtain 

$$ \sum_i p_i \delta q_i + p_t \delta t - \sum_i P_i \delta Q_i - p_\bar{t} \delta \bar{t} = \delta S $$ so that 

$$ p_t - p_\bar{t} = \frac{ \partial S }{ \partial t } $$ and 

$$ H' = H + \frac{ \partial S }{ \partial t } $$. Other ways of ensuring a canonical transformation is by properties of the phase-space. As mentioned before, the phase-space behaves like an incompressible fluid. It has integral invariants such as the circulation. This can be related to the Poisson brackets. A canoncial transformation leaves the Poisson brackets invariant. While the generating function \\(S\\) gives the transformations only in implicit form, the Poisson brackets can be used to test if a transformation is canonical. 

### Infinitesimal canoncical transformations

## Hamilton Jacobi equations

