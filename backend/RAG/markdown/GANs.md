## Generative Adversarial Networks

Generative Adversarial Networks (GANs) make use of two competing (adversarial) Networks, which aim to minimize/maximize an objective function. These two networks are denoted as the Generator-model and the Discriminator-model respectively: 

\\[ \text{Generator:} \quad G(z) \quad \text{Discriminator:} \quad D(x) \\] 

The original framework can be oriented in mathematical _Game theory_ , which is the study of strategic interaction between rational agents. The generative adversarial game consists of the Generator-Agent trying to minimize the adversarial objective while the Discriminator-Agent tries to maximize it. 

Let \\(x \sim p_{G}(z)\\) be a sample created by the Generator from random noise \\(z\\), and \\(x \sim p_{\text{Data}}\\) be a sample from the data distribution. Consider a penalty for the false discrimination of original samples and a penalty for false discrimination of generated samples. The goal of the discriminator \\(y = D(x)\\) is to produce values of \\(y\\) close to zero whenever \\(x \sim p_{\text{Data}}\\) and close to one when \\(x \sim p_G\\). Its output is restricted to \\([0, 1]\\). 

The Generator aims to produce \\(x \sim p_G\\), so that the Discriminator outputs zero even though \\(x \sim p_G\\). We can therefore distinguish between two components. The Generator Loss: 

\\[ \mathcal{L}_G = \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] \\] 

This should be as small (negative) as possible, since \\(D(G(z)) = 0\\) means that the Generator produced a sample that could not be discriminated from the real data. The other important component is the Discriminator Loss: 

\\[ \mathcal{L}_D = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] \\] 

This should be as big as possible since high values for the first term mean incorrect discrimination, whereas \\(1 - D(G(z)) \approx 0\\) means the Discriminator was fooled, causing highly negative values. 

The game-theoretic training algorithm can be described as: 

\\[ \text{Repeat until convergence:} \\]   
**1\. Update the discriminator:** \\(D \leftarrow \arg \max_D \mathcal{L}_D\\)   
**2\. Update the generator:** \\(G \leftarrow \arg \min_G \mathcal{L}_G\\) 

This is equivalent to: 

\\[ \min_G \max_D \mathcal{L}(D, G) = \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log (1 - D(G(z)))] \\] 

This min-max formulation makes GANs relatively complicated. For example, the notion of an equilibrium can depend on who is updated first. 

GANs are oriented in the field of implicit generative models. They do not explicitly model the data-likelihood function, unlike Variational Autoencoders. The learned distribution is, however, trained implicitly through discriminative training which penalizes outputs that do not fit the original distribution. 

When the data contains multiple classes, supervised or semi-supervised learning can be used. In this case, the Discriminator is a multi-class classifier and trained with the class labels, with the ability to output class-prediction and realness. 

### Conditional GANs

Ways of guiding the generative process are _Conditional GANs_. Here, the Generator, as well as the Discriminator, receive labeled data. As before, the Generator tries to create a sample that fools the Discriminator, starting from a random latent vector and a label vector. The Discriminator then evaluates the _realness_ of the sample, receiving the generated sample with its respective label. During training, the Discriminator is updated on the labeled dataset, enabling it to distinguish which label corresponds to which samples. With this architecture, it is possible to learn a Generator that produces outputs corresponding to the input labels. 

### Challenges with GANs

GANs suffer from several problems which make them hard to train and interpret. Among these is a phenomenon called _mode collapse_ , where they fail to generalize properly. A possible cause is when the Generator learns much quicker than the Discriminator. It might learn a very specific output \\(x \sim p_G\\) that consistently fools the Discriminator but does not span the data space well. 

Additionally, due to the nature of the adversarial formulation, the training is sensitive to the learning rates of both networks. While equilibria for the objective function exist, it is possible for both agents to get stuck in local minima, where a poor Generator is not able to fool the Discriminator. The Discriminator might learn too fast or too slow, and either case can result in poor generative sample quality. A balance is required between the learning of the Generator and the Discriminator. 

### Geometric Perspective

From a geometric perspective, GANs also aim to minimize the divergence between the space of generated distributions and real distributions. The type of divergence is not explicit as in other algorithms (like EM and VAE), but rather implicit. Consider the optimal Discriminator, which can be derived by point-wise differentiation of the loss with respect to \\(D(x)\\), resulting in: 

\\[ D_{\text{opt}}(x) = \frac{p_{\text{data}}}{p_{\text{data}} + p_G} \\] 

Then, we have: 

\\[ \min_G \mathcal{L}(D_{\text{opt}}, G) = \mathbb{E}_{x \sim p_{\text{data}}} \left[\log \frac{p_{\text{data}}}{p_{\text{data}} + p_G}\right] + \mathbb{E}_{z \sim p_z} \left[\log \frac{p_G}{p_{\text{data}} + p_G}\right] \\] 

\\[ = \mathbb{E}_{x \sim p_{\text{data}}} \left[\log \frac{2 p_{\text{data}}}{p_{\text{data}} + p_G}\right] + \mathbb{E}_{z \sim p_z} \left[\log \frac{2 p_G}{p_{\text{data}} + p_G}\right] \\] 

\\[ = \mathbb{E}_{x \sim p_{\text{data}}} \left[\log \frac{p_{\text{data}}}{p_{\text{data}} + p_G} - \log 2\right] + \mathbb{E}_{z \sim p_z} \left[\log \frac{p_G}{p_{\text{data}} + p_G} - \log 2\right] \\] 

\\[ = \mathbb{E}_{x \sim p_{\text{data}}} \left[\log \frac{p_{\text{data}}}{p_{\text{data}} + p_G}\right] + \mathbb{E}_{z \sim p_z} \left[\log \frac{p_G}{p_{\text{data}} + p_G}\right] - \log 4 \\] 

\\[ = KL(2 p_{\text{data}} \| p_{\text{data}} + p_G) + KL(2 p_G \| p_{\text{data}} + p_G) - \log 4 \\] 

\\[ = 2 \cdot JS(p_G \| p_{\text{data}}) - \log 4 \\] 

Here, the factor of 2 comes from considering the mixture distribution \\(0.5(p_{\text{data}} + p_G)\\). It is required for the mixture to be a well-defined distribution. 

### Extensions to Other Divergences

The GAN framework can be extended to minimize other divergences, such as the Wasserstein or KL divergence. Let us examine the extension to other forms of divergences. 

A common class of divergences are the so-called f-divergences, which include the KL and Jensen-Shannon divergences. They are defined as: 

\\[ D_f(p(x) \| q(x)) = \int q(x) f\left(\frac{p(x)}{q(x)}\right) dx \\] 

If \\(f(x) = x \log(x)\\), then it corresponds to the KL-divergence. If \\(f(x) = -(x+1) \log\left(\frac{x+1}{2}\right) + x \log(x)\\), it corresponds to the Jensen-Shannon divergence. The f-GAN objective is: 

\\[ \min_G \max_D \mathcal{L}(D, G) = \mathbb{E}_{x \sim p_{\text{data}}} [f'(D(x))] - \mathbb{E}_{z \sim p_z} [f^*(D(x))] \\] 

Here, \\(f'(\cdot)\\) is the derivative, and \\(f^*(\cdot)\\) is the convex conjugate known as the Fenchel conjugate \\(f^*(t) = \sup_{u \in \text{dom} f} \\{ ut - f(u) \\}\\). The generator is implicit in the second term via \\(x \sim p_z\\). 

### Wasserstein GANs

Another successful objective is the Wasserstein or Earth-Mover distance. Contrary to the divergences examined until now, this quantity fulfills all properties necessary for distances (e.g., the triangle inequality and symmetry). It is based on optimal transport and quantifies how easy it is to transform one distribution into another, as described in the section on Wasserstein distances. Consider the dual formulation of the Wasserstein distance for \\(p=1\\): 

\\( W_1(P, Q) = \sup_{||f||_L \leq 1} \left( \mathbb{E}_{x \sim P} (f(x)) - \mathbb{E}_{x \sim Q}(f(x)) \right) \\). 

Note that this formulation is only valid for \\( W_1(P, Q) \\), whereas for \\( W_2(P, Q) \\), there is a gradient field for the optimal transport map. However, Wasserstein GANs (WGANs) use the \\( W_1(P, Q) \\) distance, which still has properties that the KL-divergence or Jensen-Shannon divergence does not offer. 

An extension to \\(K\\)-Lipschitz functions is done by replacing \\(||f||_L \leq 1\\) with \\(||f||_L \leq K\\), leading to \\(K \cdot W_1(P, Q)\\). Thus, given a parameterized family \\(\\{ f_w \\}\\), solving: 

\\( \arg \max_w \left( \mathbb{E}_{x \sim P} (f_w(x)) - \mathbb{E}_{z \sim p(z)} (f_w(g_\theta(z))) \right) \\), 

where \\(g_\theta(z)\\) is a feed-forward neural network defining the distribution \\(Q\\), and \\(p(z)\\) is a prior over \\(z\\), this forms the WGAN objective. Here, \\(g_\theta(z)\\) corresponds to the Generator, while \\(f_w\\) is known as the Critic. The \\( W_1(P, Q) \\) objective is continuous everywhere and differentiable, even in cases where the Jensen-Shannon divergence might not be. 

The WGAN algorithm uses weight clipping to ensure that the Lipschitz constraint is maintained during training, though alternative methods like gradient penalty can also be employed to enforce the constraint in a more flexible manner. These methods aim to make the training of GANs more stable and effective. 

### Practical Considerations for GAN Training

Training GANs involves challenges such as balancing the Generator and Discriminator's learning rates, preventing mode collapse, and ensuring that the training dynamics converge to a stable equilibrium. Techniques like batch normalization, different learning rate schedules, and architectural choices for the networks can help address these issues. Additionally, using improved loss functions like Wasserstein loss and implementing techniques like gradient penalty can significantly enhance training stability and performance. 

### Conclusion

GANs are a powerful class of generative models that can generate high-quality synthetic data by training two adversarial networks in a game-theoretic setting. While training GANs presents numerous challenges, various modifications to the original architecture and training objectives, such as f-GANs and Wasserstein GANs, have expanded their applicability and made the training process more stable. Despite these advancements, further research is still needed to fully overcome the limitations inherent in adversarial training and to better understand the theoretical foundations behind these models. 

![An Auto-Encoder](../images/GAN/Generative-Adversarial-Network-Architecture.png) An illustration of the Generative Adversarial Network Architecture

![An Auto-Encoder](../images/GAN/Generative-Adversarial-Network-Architecture.png)