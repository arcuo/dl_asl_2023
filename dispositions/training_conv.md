## Training Convolutions Neural Networks

### Backpropagation

Computing the gradients of the loss function with respect to the weights and biases.

**Quadratic cost function**: $J(W,b) = \frac{1}{2n} \sum_x \| y(x) - a^L(x) \|^2$

#### Computing the error

Let $L$ denote the number of layers in the network.

$$
\begin{align*}
    \delta^{(L)} &= (a^{(L)} - y^{(k)}) \odot \sigma'(z^{(L)})\\
    \delta^{(l)} &= ((W^{(l)})^T \delta^{(l+1)}) \odot \sigma'(z^{(l)})\\
    \sigma'(z^{(l)}) &= \sigma(z^{(l)}) \odot (1 - \sigma(z^{(l)})) = a^{(l)} \odot (1 - a^{(l)}) & \text{is the derivate of the sigmoid function}\\
\end{align*}
$$

### Activation functions

### Data preprocessing

### Weight initialization

### Batch normalization

### Stochastic Gradient Descent (SGD)

### SGD extensions: Momentum, AdaGrad, RMSProp, and Adam

### Learning rate decay and cycling

### Regularization: Early stopping, weight decay, dropout, data augmentation

### Hyperparameter search

### Transfer learning

[Go back](main.md)
