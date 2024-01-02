## Fundamentals of machine learning

### Linear regression
A model that takes a vector of inputs $x \in \mathbb{R}^m$ and outputs a scalar $y \in \mathbb{R}$.

Draw a line that best fits the data.

For linear regression, choose a parameter $w_1$ such that $y = w_1 x$

### Loss functions

The linear model

$$
h_w(x) = \sum_{j = 1}^m w_jx_j = w^Tx
$$

We want to choose a linear model such that $h_w(x^{(i)}) \approx y^{(i)}$ is as close to the training output $y^{(i)}$ for each training example $i$ as possible.

An example of a loss function is L2-norm loss $J(w)$:

$$
J(w) = \frac{1}{2} \sum_{i = 1}^n (h_w(x^{(i)}) - y^{(i)})^2 = \frac{1}{2} \sum_{i = 1}^n (w^Tx^{(i)} - y^{(i)})^2
$$

### Gradient descent

Imagine the 
### Hyperparameters
### Logistic regression
### Weight decay (regularization)


### Softmax regression
### K-Nearest Neighbours (KNN) classifier
### Supervised vs. unsupervised learning
### K-means clustering

[Go back](main.md)
