## Visualising CNNs

We canuse visualizations to debug and understand the inner workings of a CNN. This can be hard, because CNNs are not invertible. We can't just reverse the process to get the input from the output. Instead we can visually inspect the output of each layer.

We can derive how we might pre-process our data to make it easier for the network to learn, or how to adjust the architecture of the network to make it easier to train.

### Visualizing layer activations and filters

Each layer in the model detects different features of the input. The first layer might detect edges, the next layer detects shapes like "dog" or "leaf". The last layer may create complicated connections like "dog in front of house" or "people in a car".

Visualizing the filters should show smooth gradients of color, and not random noise. If the filters are random noise, then this might indicate that the network is not converged or is overfitting.

### Inspecting fully connected layers

#### K-nearest neighbors check

Run nearest neighbors on the output of the fully connected layer. Check if the images look similar to the input image. Are they the same class and how to the look similar?

This information can be further investigated by looking into the confusion matrix. Which classes are confused with each other?

### Visualization using dimensionality reduction (PCA or t-SNE)

Reduce the dimensionality of the feature vectors to two dimensions.

#### Principal Component Analysis (PCA)

Uses the correlation between dimensions to create the minimum number of dimensions that explain the most variance in the data.

It uses eigenvalues and eigenvectors to find the dimensions that explain the most variance in the data.

Draw a plot with a lot of dots and and arrows pointing to the widest areas of the dots.

#### t-SNE

t-Distributed Stochastic Neighbouring Entities.

This technique is probabilitistic. Gives a rough idea of the topology of the learned representation.

You can take an input image, take the last output before the dense layer and run t-SNE on it. This will give you a coordinate. You can then plot inputs in a grid.

Inputs that the CNN considers similar will be close together in the grid. Inputs that the CNN considers different will be far apart in the grid.

### Maximally activating patches

Visualise how some inputs maximally activate some neurons in the network.

We can then visualise what in the input, the neuron is activated by.

#### Receptive field

The region in input space that the CNN is looking at, defined by the center and size.

### Deconvolution

Project hidden feature maps back to the input space.

Deconvnet added to each convolutional layer. It takes the reconstructed feature map from the last layer and the feature map from the current layer and reconstructs the feature map from the previous layer.

**Max-unpooling**: Max pooling is not invertible. We add switches that record the location of max pooling cells. We then use these switches to place the values back in the correct location.

**Rectification**: We use the rectified linear unit (ReLU) to keep values positive

**Filter reconstruction**: We flip filters vertically and horizontally. Corresponds to backpropagation.

Produces gray images that indicate the activation of the filter.

### Saliency maps / class activation maps

Show which parts of the image are important for the classification. E.g. is it the dog or the green grass that is important for the classification for the class "dog"?

Grad-CAM: Take the output of the last convolutional layer given an input image and weigh each output by the gradient of predicted class.

### Reconstruction-based visualization techniques

Inverting CNNs

- $x$ is the input image
- $y$ is the generated image.
- let $a^{[l](G)}$ be the activation of layer $l$ when $y$ is the input.
- let $a^{[l](C)}$ be the activation of layer $l$ when $x$ is the input.

We want to minimize the difference between $a^{[l](G)} \approx a^{[l](C)}$.

$$
J^{[l]}_C(x,y) = \left\lVert a^{[l](G)} - a^{[l](C)} \right\rVert^2_\mathcal{F}\\
||G||_{\mathcal{F}} = \sqrt{\sum_{i,j} g_{i,j}^2}
$$

this can be regulized by $R_a(y) = \lambda_a||y||^a_a$

1. Initialize $y$ with random noise
2. Feedforward pass the image
3. Compute the loss function
4. Compute the gradients of the loss and backpropagate to input space
5. Update the input $y$ using gradient descent
6. Repeat steps 2-5 until convergence

#### Reconstruction-based filter visualization

Visualize the convolution filters.

Start with random noise and take the output of a given layer with that input.

Find a global average of each filter in the output layer to form a loss function. Update our input image using gradient ascent, to maximize the loss function.

#### Class-based reconstruction

Optimize random input to maximize the probability of a given class.
This doesn't work well by itself, but add the restriction that pixel colors must correlate to create something that looks like the class.

This can help us see if the network has learned the correct features of a class.

E.g. the image of a "dumbbell" might include an arm because all the training data images has arms in them.

### Texture synthesis

### Neural Style Transfer

[Go back](main.md)
