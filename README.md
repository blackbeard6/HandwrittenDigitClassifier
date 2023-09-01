# HandwrittenDigitClassifier
A neural network built from scratch using fundamental linear algebra to acheive approximately 82% accuracy on the MNIST dataset.

## How it works:
This neural network propogates an input $X$ from the first layer to the last layer, and then to a loss funciton. The layer-by-layer structure of Linear -> Non-Linear -> Linear -> ... Linear -> Loss, where the output from one layer is the input to the next layer. The final linear layer outputs an $256x10$ matrix $Y_{\text{pred}}$, and we consider the maximum value in each row to be the class prediction for that particular input (converting to a one-hot encoding vector). At the end of our forward step, we calculate the current loss. In our backward step, we work backwards from the loss (the output from the current layer is the input to the previous layer) to calculate change in the loss function with respect to changes in the inputs to the layer. By working backwards, we employ the chain rule to simplify this process of calculating otherwise complex derivatives. During backward propogation, we also calculate the gradient parameters for gradient descent. Note that our forward pass really serves to calculate the intermediate values needed for backward propogation, but we also measure the accuracy of our predictions during each epoch to measure our model's performance. With each epoch, we expect to see our loss converge to 0.

<p align="center">
  <img width="406" alt="Screenshot 2023-09-01 at 4 17 39 PM" src="https://github.com/blackbeard6/HandwrittenDigitClassifier/assets/103332396/871054d8-d910-482f-8b08-c414a72fc6b4">
</p">

## Training Process (Gradient Descent)

We slice the data horizontally into many batches to avoid memory issues from working with a tall matrix. In each batch we calculate the gradients by conducting a forward pass followed by a backward pass. These gradients are stored and used to update the values of our parameters in the next iteration (by subtracting the gradients multiplied by the learning rate from the current weights). By repeating this process many times, we seek to tune the our parameters such that we achieve the ability to predict class labels with high accuracy.

## Linear Layer
The linear layer simply conducts matrix-matrix multiplication with the input matrix and a matrix of weights.

### Forward
* Input: $X$ (# of samples, # of features)
* Output: $Y = XW$ (# of samples, # of features)
* Stored Data: Input ($X$) for backward propogation

### Backward
* Input: $\frac{d\text{loss}}{dY}$ (# of samples, # of features)
* Output: $\frac{d\text{loss}}{dX}$ (# of samples, # of features)
* Stored: Weight gradient matrix $\frac{d\text{loss}}{dW}$ (# of samples, # of features)
  
## Non-Linear Layer (ReLU activation)

### Forward
* Input: $X$ (# of samples, # of features)
* Output: $Y$ where $ReLU(x) = max(0, x)$ (# of samples, # of features)
* Stored Data: Input ($X$) for backward propogation

### Backward:
* Input: $\frac{d\text{loss}}{dY}$ (# of samples, # of features)
* Output: $\frac{d\text{loss}}{dX}$ (# of samples, # of features)

The gradient of $ReLU(x)$ is 1 if $x > 0$ and 0 otherwise. Thus, we apply the chain rule to calculate each element in the output matrix as follows:

$$
\frac{d\text{loss}}{dX}(i,j) = \begin{cases}
\frac{d\text{loss}}{dY}(i,j), & \text{if } A(i,j) > 0 \\
0, & \text{if } A(i,j) \leq 0
\end{cases}
$$

## MSE Loss Function

### Foward (Loss)
* Input: $Y_{pred}$ (# of samples, # of features), $Y_{truth}$ (# of samples, # of features)
* Output: $loss$ (scalar)
* Stored data: input $Y_{pred} - Y_{truth}$

Where $loss = \frac{1}{\text{number of samples}}||Y_{pred} - Y_{truth}||^2$

### Backward (Loss)
* Input: None
* Output: $\frac{d\text{loss}}{dY}$ (# of samples, # of features)

Where $\frac{d\text{loss}}{dY}= \frac{2}{\text{number of samples}}||Y_{pred} - Y_{truth}||$.

## Road Map
- [ ] Implement bias in the neural network model for classification.
- [ ] Inject artificial noise into gradient descent.
- [ ] Design alternative nonlinear activation functions.
- [ ] Experiment with alternative architectures and compare performance.
