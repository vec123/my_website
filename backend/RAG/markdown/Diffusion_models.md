## Diffusion Networks

Diffusion Networks derive their working from the parametrization of a Markov-chain which removes noise from a random image. This parametrization is learned by applying a diffusion process on the training data. This process iteratively adds a small amount of Gaussian noise over T steps, producing a sequence of noisy samples \\(x_1, x_2, \dots ,x_T\\). The step sizes are controlled by a variable schedule \\((\beta_t \in (0,1))_{t=1}^T\\).

The forward diffusion process is defined by
    
    
            q(x_t | x_{t-1}) 
            = \mathcal{N}(x_t; \sqrt{(1 - \beta_t)}x_{t-1}, \beta_t \cdot I)
            

and
    
    
            q(x_{1:T} | x_0) 
            = \prod_{t=1}^{T} q(x_{t} | x_{t-1})
            

The data sample \\(x_0\\) is gradually superimposed with noise. This creates a chain of images, from the original at time 0, to the maximally noisy image at time T. An output at time \\(t\\) can be sampled considering that
    
    
            x = \mu + \sigma \epsilon \sim \mathcal{N}(\mu,\sigma)
            

if
    
    
            \epsilon \sim \mathcal{N}(0, 1).
            

Therefore,
    
    
            x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon \sim \mathcal{N}(x_t; \sqrt{(1 - \beta_t)}x_{t-1}) = q(x_t | x_{t-1}).
            

Let
    
    
            \alpha_t = 1 - \beta_t \text{,}
            
    
    
            \epsilon_{t} \sim \mathcal{N}(0,I) \forall t
            

and
    
    
            \bar{\epsilon}_{t} \sim \mathcal{N}(0,(\sigma_1^2 + \sigma_2^2)I)
            

be from a new distribution which merges two Gaussians.

Then
    
    
            x_t 
            = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1} 
            = \sqrt{\alpha_t \alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_t \alpha_{t-1}} \bar{\epsilon}_{t-2} = \dots
            

enables sampling at every time-step \\(t\\) by knowing \\(\alpha_t\\) and \\(\bar{\epsilon}_{t}\\) so that
    
    
            q(x_t|x_0) = \mathcal{N}(\bar{\alpha}x_0, (1-\bar{\alpha}) I)
            

where \\(\bar{\alpha}_T = \prod_{t=1}^{t=T}\alpha_t\\) and
    
    
            x_t = \bar{\alpha}x_0 + (1-\bar{\alpha}) \epsilon_t.
            

This enables sampling noisy data given \\(x_0\\). If this noise amplifying process can be reversed, one can recreate the noise-free sample \\(x_0\\) from Gaussian Noise \\(x_T \sim \mathcal{N}(0,I)\\).

The forward diffusion process defined by \\( q(x_{t} | x_{t-1}) \\) has a corresponding reverse diffusion \\(q(x_{t-1}| x_{t})\\) which we would like to approximate with a learned model \\(p_\theta(x_{t-1} | x_{t})\\), so that
    
    
    p(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_{t},t), \Sigma_\theta(x_{t},t) )
    

and
    
    
    p(x_{0:T}) 
    = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_{t}).
    

Importantly, the goal is that
    
    
    p_\theta(x_0) = \int_{x_{1}} p(x_0|x_{1}) p(x_{1}) dx_1 =\int_{x_{2}} p(x_0|x_{1}) p(x_1|x_2) p(x_{2}) dx_2 = p_{Data},
    

which corresponds to
    
    
    p_\theta(x_0) = \int_{x_1:T} p(x_0,x_{1:T}) dx_{1:T} = p_{Data}
    

This corresponds to a maximization of the log-likelihood \\(\log( p_\theta(x_0) )\\), which has been examined in the section on maximum likelihood estimation. However, note that \\(p_\theta(x_0)\\) involves additional unmeasured variables \\(z=x_{1:T}\\) so that we write
    
    
    p_\theta(x_0) = \int_{z} p_\theta(x_0) p_\theta(x_0,z) dz = \int_{z} p(x_0|z) p(z).
    

This quantity has been examined in the section on Expectation Maximization. In an equivalent manner, one can formulate the bound,
    
    
    - \log(p_\theta(x_0)) = - \log( \int_{z}p_\theta(x_0,z) \frac{q(z|x_0)}{q(z|x_0)} ) dz &\leq  - \int_{z} \log( \frac{p_\theta(x_0,z)}{q(z|x_0)} ) q(z|x_0)dz
    
    
    
    = - D_{\text{KL}}( p_\theta(z,x) || q(z|x_0) ) \\
    = - \log(p_\theta(x_0)) +  D_{\text{KL}}( q(z|x_0)|| p_\theta(z|x_0) )
    
    
    
    = -\log(p_\theta(x_0)) + \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)}[\log(\frac{q(x_{1:T}|x_0) }{ p_\theta(x_{0:T}) + \log(p(x_0)) })] 
    
    
    
    = \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)}[\log(\frac{q(x_{1:T}|x_0) }{ p_\theta(x_{0:T}) })]
    \end{aligned}
    

This quantity defines a bound on the objective. By minimizing it, the maximum likelihood is implicitly maximized. Let us decompose the loss into the relevant time-steps.
    
    
    \begin{aligned}
    &= \mathbb{E}_{x_{1:T} \sim q(x_{1:T}|x_0)}[\log(\frac{q(x_{1:T}|x_0) }{ p_\theta(x_{0:T}) })]
    \\
    &=\mathbb{E}_{q}[ \frac{ \prod_{t=1}^{T}q(x_t|x_{t-1}) }{ p_\theta(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}|x_{t})} )]
    
    
    
        &=\mathbb{E}_{q}[ -\log(p_\theta(x_T)) + \sum_{t=1}^{T} \log( \frac{q(x_t | x_{t-1}) }{ p_\theta(x_{t-1}|x_t) } )]
        \\
        &=\mathbb{E}_{q}[ -\log(p_\theta(x_T)) + \sum_{t=2}^{T}( \frac{q(x_{t-1} | x_t, x_0) }{ p_\theta(x_{t-1}|x_t) }) \frac{q(x_{t} | x_0 )}{ q_\theta(x_{t-1}|x_0) }   )  ) + \log(\frac{q(x_1|x_0)}{ p_\theta(x_0,x_1)} )]
        \\
        &=\mathbb{E}_{q}[ -\log(p_\theta(x_T)) + \sum_{t=2}^{T}( \log( \frac{q(x_{t-1} | x_t, x_0) }{ p_\theta(x_{t-1}|x_t) }) ) + \sum_{t=2}^{T}( \log( \frac{q(x_{t} | x_0 )}{ q_\theta(x_{t-1}|x_0) }   )  ) + \log(\frac{q(x_1|x_0)}{ p_\theta(x_0,x_1)}  )]
        \\
        &=\mathbb{E}_{q}[ -\log(p_\theta(x_T)) + \sum_{t=2}^{T}( \log( \frac{q(x_{t-1} | x_t, x_0) }{ p_\theta(x_{t-1}|x_t) }) ) + \log( \frac{q(x_{T} | x_0 )}{ q_\theta(x_{1}|x_0) }    ) + \log(\frac{q(x_1|x_0)}{p_\theta(x_0,x_1)} )]
        \\
        &=\mathbb{E}_{q}[ \log( \frac{q(x_{T} | x_0 )}{p_\theta(x_T)} + \sum_{t=2}^{T} \log( \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)} -  \log(p_\theta(x_0|x_1) )]
        \\
        &=\mathbb{E}_{q}[ D_{KL}(q(x_T|x_0) || p_\theta(x_T)) +  \sum_{t=2}^{T} D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t)) - \log(p_\theta(x_0|x_1)) ]\\
        &= L_T + L_{t-1} + L_0.
        

\\(L_T\\) can be ignored during training since \\(q(x_T|x_0)\\) has no learnable parameters and \\(x_T\\) is pure Gaussian noise. The paper Denoising Diffusion Probabilistic Models models \\(L_0\\) separately as a discrete decoder. \\(L_t\\), on the other hand, is parameterized for the training Loss.

Conditioning \\(q(x_{t-1}|x_{t})\\) on \\(x_0\\) and applying Bayes' rule enables a parameterization of its mean as
    
    
        \mu_t(x_t,x_0) = \frac{ \sqrt{\alpha_t } (1-\bar{\alpha}_{t-1}) }{1- \bar{\alpha}_{t} } x_t + \frac{\sqrt{ \bar{\alpha}_{t-1} }\beta_t } { 1-\bar{\alpha}_{t} }x_0.
        

By expressing \\(x_0\\), as shown before, as
    
    
        x_0 = \frac{1} { \bar{ \alpha_t }} (x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_t)
        

leading to
    
    
        \mu_t(x_t,x_0) = \frac{1}{\bar{ \alpha_t }} (x_t - \frac{1 - \alpha_t} {\sqrt{1 - \bar{\alpha}_t} } \epsilon_t)
        

This allows for a parameterization of \\(q(x_{t-1}|x_{t}, x_0) = \mathcal{N}(\mu(x_t, x_0), \Tilde{\beta}I)\\). For now, let us ignore that we did not obtain \\(\Tilde{\beta}\\) and assume it were given.

By considering the mean of the denoising process \\(q(x_{t-1}|x_{t}, x_0)\\)
    
    
        \mu_t(x_t,t) = \frac{1}{\bar{ \alpha_t }} (x_t - \frac{1 - \alpha_t} {\sqrt{1 - \bar{\alpha}_t} } \epsilon_t).
        

and that, at training time, \\(x_t\\) is available, we can write
    
    
        \mu_\theta(x_t,t) = \frac{1}{\bar{ \alpha_t }} (x_t - \frac{1 - \alpha_t} {\sqrt{1 - \bar{\alpha}_t} } \epsilon_\theta)
        

and parameterize \\(\epsilon_\theta\\).

Correspondingly the model shall be
    
    
        p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t)  ).
        

Instead of calculating the divergence between \\( D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))\\), simple diffusion models consider a loss that attempts to fit the mean of parameterization \\(p_\theta(x_{t-1}|x_t)\\) to the real mean of \\(q(x_{t-1}|x_t)\\) as
    
    
    L_t = \mathbb{E}_{x_0,t}\left[\frac{1}{2||\Sigma_\theta(x_t,t)||_2^2 }||\mu_t(x_t,t) - \mu_\theta(x_t,t) ||^2\right] 
    = \mathbb{E}_{x_0,t}\left[\frac{(1-\alpha_t)^2}{2 \alpha_t (1 - \bar{\alpha_t})||\Sigma_\theta(x_t,t)||_2^2} ||\epsilon_t - \epsilon_\theta( \sqrt{\bar{\alpha}}x_0 + \sqrt{1-\bar{\alpha}}\epsilon_t,t ||^2\right].
    

Here, \\(\Sigma_\theta(x_t,t) = \sigma_t^2 I\\), where \\(\sigma = \beta_t\\) or \\(\sigma = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \cdot \beta_t\\). Denoising Diffusion paper shows that learning a diagonal variance leads to unstable training. Other approaches, Improved Denoising Diffusion Probabilistic Models, propose learning as an interpolation of the two diagonal matrices.

It has been shown that the algorithm performs better with a simplified objective
    
    
    L_t = \mathbb{E}_{t \sim [1,T],x_0,t}[ ||\epsilon_t - \epsilon_\theta(x_t,t) ||^2 ]  
    = \mathbb{E}_{t \sim [1,T],x_0,t}[  ||\epsilon_t - \epsilon_\theta( \sqrt{\bar{\alpha}}x_0 + \sqrt{1-\bar{\alpha}}\epsilon_t,t )||^2   ]
    

where \\(\epsilon_t\\) is sampled during the noisifying process and then used in training. \\(\epsilon_\theta\\) is the output of a neural network that receives as input the current time \\(t\\) and the noisy sample \\(x_t\\) and learns the (de-)noisifying relationship, which is the inverse of
    
    
    x_t = \bar{\alpha}x_0 + (1-\bar{\alpha}) \epsilon_t.
    

Note, that although the real model should be
    
    
    p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t)  ),
    

only the mean is considered in the simple loss, and the variance is not learned. The Diffusion model replaces the KL-divergence between the real denoising process \\(q(x_{t-1}|x_t,x_0)\\) and parameterization \\(p_\theta(x_{t-1}|x_t)\\) with a mean squared error loss. It can be shown that for two Gaussians with equal variance, minimizing the mean squared error loss is equivalent to minimizing the KL-divergence.

The variance of the (de-)noisifying schedule is fixed by the schedule. Furthermore, it is very small compared to the values of \\(x\\). Extensions that attempt to learn the variance exist, but they are difficult to optimize. In practice, this does not seem to be necessary.

The following figure illustrates the diffusion process:

The Diffusion Process ![The Diffusion Process](../images/diffusion/Diffusion/DDPM.png)

![The Diffusion Process](../images/diffusion/Diffusion/DDPM.png)