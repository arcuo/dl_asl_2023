## Fundamentals of machine learning

### Linear regression

A model that takes a vector of inputs $x \in \mathbb{R}^m$ and outputs a scalar $y \in \mathbb{R}$.

Draw a line that best fits the data.

For linear regression, choose a parameter $w_1$ such that $y = w_1 x$

The linear model

$$
h_w(x) = \sum_{j = 1}^m w_jx_j = w^Tx
$$

We can also use polynomial regression where $h_w(x) = \sum_{j = 1}^m w_jx_j^j$.

### Loss functions

We want to choose a linear model such that $h_w(x^{(i)}) \approx y^{(i)}$ is as close to the training output $y^{(i)}$ for each training example $i$ as possible.

An example of a loss function is L2-norm loss $J(w)$:

$$
J(w) = \frac{1}{2} \sum_{i = 1}^n (h_w(x^{(i)}) - y^{(i)})^2 = \frac{1}{2} \sum_{i = 1}^n (w^Tx^{(i)} - y^{(i)})^2
$$

### Gradient descent

Imagine the loss function as a landscape. The goal is to find the lowest point in the landscape.

Gradient descent follows the slope of the landscape to find the lowest point.

The algorithm is iterative and

$$
w^{k+1} = w^k - \alpha \nabla J(w^k)
$$

Here $\alpha$ is the learning rate and how far we step in the direction of the gradient.

The gradient is the partial derivative of the loss function with respect to the parameters $x_i$.

$$
\frac{\partial}{\partial x_i}f(x)\\

\nabla f(x) = \begin{bmatrix}
\frac{\partial}{\partial x_1}f(x)\\
\frac{\partial}{\partial x_2}f(x)\\
\vdots\\
\frac{\partial}{\partial x_n}f(x)
\end{bmatrix}
$$

It measures the rate of change in point $x$ with respect to $x_i$.

The derivative of the loss function is in point $j$

$$
\frac{\partial J(w)}{\partial w_j} =  \frac{1}{2} \sum_{i = 1}^n x_j^{(i)} (w^Tx^{(i)} - y^{(i)})^2
$$

By the chain rule: for $q = f(p)$ and $p = g(x)$, then $\frac{\partial q}{\partial x} = \frac{\partial q}{\partial p} \frac{\partial p}{\partial x}$.

$$
p = (wx-y)\\
\frac{\partial p}{\partial w} = x\\
q = \frac{1}{2} p^2\\
\frac{\partial q}{\partial p} = p\\
\frac{\partial q}{\partial x} = \frac{\partial q}{\partial p} \frac{\partial p}{\partial x} = xp = x(wx-y)
$$

Gradient descent is not fast and only guaranteed to find a local minimum. You can use different strategies to improve the performance.

 - Momentum - Accelerates gradient descent in the right direction and dampens oscillations when we are close to the minimum.

### Hyperparameters

Variables we set before training

#### Learning rate

How big a step you take in each gradient descent iteration. If the steps are too big you might overshoot the minimum or jump back and forth across the minimum. If the steps are too small it can take a long time to converge. Furthermore, you can end up i local minima instead of the global minimum.

#### Polynomial degree

For a polynomial regression, the degree of the polynomial is a hyperparameter.

We can search for the best hyperparameters by trying different values and see which one gives the best result. I.e. calculate the mean-squared error (MSE) for each hyperparameter and choose the one with the lowest error.

### Logistic regression

Classification problems as opposed to regression problems. We want to select whether an input belongs to one of two classes.

For this we can use logistic regression.

Pick a differentiable function that outputs the probability that an input belongs to a class.

$$
\begin{align*}
h_w(x) = P(y = 1 | x) &= \frac{1}{1 + e^{-w^Tx}} \equiv \sigma(w^Tx)\\
P(y = 0 | x) &= 1 - P(y = 1 | x) = 1 - \sigma(w^Tx)
\end{align*}
$$

$\sigma$ is the sigmoid function and squashes the output to the interval $[0, 1]$.

#### Loss function

When $y^{(i)} = 1$ we want $h_w(x^{(i)}) \approx 1$ and $h_w(x^{(i)}) \approx 0$ otherwise, to minimize the loss. 

$$
J(w) = - \sum_{i = 1}^n y^{(i)} \log(h_w(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_w(x^{(i)}))
$$

For gradient descent the derivative is almost the same as for linear regression except for the sigmoid function.

### Weight decay (regularization)

Overfitting is when the model is to complex and is able to fit to the random noise of our sample data.

If for example our model is a polynomial, weight decay is removing some of the unimportant terms in the polynomial.

We extend our loss function with an extra term $\lambda R(w)$ where $R$ is either the L1 or L2-norm of $w$.

$$
J_{reg} = J(w) + \lambda R(w)
$$

### Softmax regression

For classification problems with more than two classes we can use softmax regression.

Outputs a vector of probabilities for each class.

### K-Nearest Neighbours (KNN) classifier

Linear models have a hard time predicting non-linear data. I.e if you have decision boundary that is a straight line, its not very often that it will do very well at separating complex data.

Prediction: For a new input $x$ find the $k$ closest training examples and let them vote on the class of $x$.

This is very computationally expensive.

In high dimensional data, images can be very far apart even if they are similar. Therefore, we can use convolutional neural networks to encode/extract features from the image and reduce the dimensionality of the data.

This increases both performance and speed of prediction.

### Supervised vs. unsupervised learning

### K-means clustering

Grouping a set of vectors such that the vectors in the same group are similar to each other than to other groups.

_k-Means clustering_:

1. Create random centroids for each group
2. Assign each vector to the closest centroid using L1 or L2-norm
3. Update the centroids to be the mean of the vectors in the group
4. Repeat from 2. until convergence

[Go back](main.md)
