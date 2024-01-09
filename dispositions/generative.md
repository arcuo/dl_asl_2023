## Generative Networks

### Unsupervised learning

Supervised learning:

- Input: $x$
- Output: $y$
- Goal: learn function to map $x$ to $y$
- Examples: classification, regression

Unsupervised learning:

- Input: $x$
- Output: ?
- Goal: Learn some underlying structure of the data
- Examples: clustering, dimensionality reduction, density estimation

#### Use cases

**Training with few labels**
You have a lot of data, which does not have labels.

You can pre-train a network on the data, and then use the pre-trained network for a supervised task.

**Look for anomalies in data**

### Generative models

From a distribution, train a model to generate new samples from the same distribution.

**Discriminative (classifier) versus Generative**

Generative models capture the joint probability distribution $p(x,y)$. They include the distribution of the data and tells you how likely it is to see a particular label.

Models that predict the next word in a sentence from a language are a generative model.

Generative models keep more information. Where a distriminative model can detect a boat, from just the sail, the generative model has to keep more information about the whole boat to generate it.

## Autoencoders

### Traditional Autoencoder (AE)

Attempts to copy its input to its output.

$$
h = f(x)\\
\hat{x} = g(x) = g(f(x)) = x\\

f(x) = \sigma(Wx + b)\\
g(x) = \sigma(W^Tx + b')\\
$$

Traditionally used for dimensionality reduction. Learn features of the data.

**Loss function L2-norm**

$$
J(W) = \frac{1}{2}\sum^n_{i=1}||x^{(i)} - \hat{x}^{(i)}||^2_2
$$

**Loss function Cross entropy**

For probability

$$
J(W) = -\sum^n_{i=1} x \log(\hat{x})
$$

### Undercomplete Autoencoder

THe hidenn layer has smaller dimension than the input layer. This compresses the data. Good for detecting features in the training data, but not as good for other inputs.

We can use this to detect anomalies in the data. As we cat check if $\hat{x}$ deviates from $x$.

### Overcomplete Autoencoder

The hidden layer has larger dimension than the input layer.

Useful for training a linear classifier.

### Stacked AE

Stack multiple layers that first increase and then reduce in dimensionality, before mirroring back the original dimensionality.

Perform much better than a single layer AE.

The middle 2D layer is called the **bottleneck** or the **latent space**.

Impose a bottleneck to force a constrained representation of the data.

Sensitive enough to inputs so that it can be used for reconstruction, but insensitive enough to not just memorize the data.

Regularizer prevents memorization $L(x, \hat{x}) + regularizer$

### Sparse AE

Penalize activations within the network.

Instead of regularizing the weights, we regularize the activations.

Add a penalty term **L1-regularisation** that penalizes the number of activations $a$ for a layer $h$ based on a tuning parameter $\lambda$.

$$
L(x, \hat{x}) = \lambda \sum_i | a_i^{(h)}|
$$

and **KL-divergence** that minimizes the average activation over a layer with a sparsity parameter $\rho$.

$$
\rho = \frac{1}{m} \sum_i [a_i^{(h)}(x)]\\
L(x, \hat{x}) = \sum_j KL(\rho || \hat{\rho}_j)\\
$$

### Denoising AE

Add noise to the input but keep the output the same.

### Convolutional autoencoders

### Variational autoencoders

### Generative Adversarial Networks (GANs)

Create a supervised task by training two networks against each other.

A generator network $G$ generates samples from the data distribution. Random noise sampling.

A discriminator network $D$ tries to distinguish between real and fake samples.

Zero-sum game (What one opponent wins, the other loses).

**Discriminator**

Sample from some gaussian distribution $G(z)$. We update the noise function to a prior noise $p_g(z)$. The loss function:

$$
\begin{align*}
D(x) &= 1\ (real) \\
D(G(z)) &= 0\ (fake) \\

L_D(x, z) &= \max_{W_D} \sum_i^m \log (D(x^{(i)})) + \log (1 - D(G(z^{(i)})))
\end{align*}
$$

**Generator**

The loss function:

$$
D(G(z)) = 1\ (fake)\\
L_G(x) = \min_{W_G} \sum_i^m \log (1 - D(G(z^{(i)})))
$$

### Training minibatch

We randomly sample from prior noise $\{...z\} = p_g(z)$ and from the data distribution $\{...x\} = p_{data}(x)$. Update the discriminator using $L_d$ with stochastic gradient descent for $k$ steps.

Sample new prior noise. Update the generator using $L_g$ with stochastic gradient descent.

**Minmax game**

$$
\int p_{data}(x)\log(D(x)) dx + \int p_g(z)\log(1 - D(G(z))) dz
$$

Global maximum when $p_{generator} = p_{data}$

#### Issues

- Non-convergence: The model oscillates
- Mode collapse: The model only produces limited variety of samples
- Vanishing gradients: The discriminator becomes too good and the generator cannot learn because $D(G(z))$ is close to 0 or 1.


#### Convolutional GAN

Convolution is good for generating images. GAN is good for generating samples. Convolutional GAN is good for generating images.

Using deconvolution we don't have to create random sampling that knows nothing of the data.

[Go back](main.md)
