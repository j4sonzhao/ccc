import numpy as np

class Layer(object):
    '''
    Abstract class representing a neural network layer
    '''
    def forward(self, X, train=True):
        '''
        Calculates a forward pass through the layer.

        Args:
            X (numpy.ndarray): Input to the layer with dimensions (batch_size, input_size)

        Returns:
            (numpy.ndarray): Output of the layer with dimensions (batch_size, output_size)
        '''
        raise NotImplementedError('This is an abstract class')

    def backward(self, dY):
        '''
        Calculates a backward pass through the layer.

        Args:
            dY (numpy.ndarray): The gradient of the output with dimensions (batch_size, output_size)

        Returns:
            dX, var_grad_list
            dX (numpy.ndarray): Gradient of the input (batch_size, output_size)
            var_grad_list (list): List of tuples in the form (variable_pointer, variable_grad)
                where variable_pointer and variable_grad are the pointer to an internal
                variable of the layer and the corresponding gradient of the variable
        '''
        raise NotImplementedError('This is an abstract class')

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        '''
        Represent a linear transformation Y = X*W + b
            X is an numpy.ndarray with shape (batch_size, input_dim)
            W is a trainable matrix with dimensions (input_dim, output_dim)
            b is a bias with dimensions (1, output_dim)
            Y is an numpy.ndarray with shape (batch_size, output_dim)

        W is initialized with Xavier-He initialization
        b is initialized to zero
        '''
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0/input_dim)
        self.b = np.zeros((1, output_dim))

        self.cache_in = None

    def forward(self, X, train=True):
        out = np.matmul(X, self.W) + self.b
        if train:
            self.cache_in = X
        return out

    def backward(self, dY):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        db = np.sum(dY, axis=0, keepdims=True)
        dW = np.matmul(self.cache_in.T, dY)
        dX = np.matmul(dY, self.W.T)
        return dX, [(self.W, dW), (self.b, db)]

class ReLU(Layer):
    def __init__(self):
        '''
        Represents a rectified linear unit (ReLU)
            ReLU(x) = max(x, 0)
        '''
        self.cache_in = None

    def forward(self, X, train=True):
        if train:
            self.cache_in = X
        return np.maximum(X, 0)

    def backward(self, dY):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        return dY * (self.cache_in >= 0), []

class BatchNorm(Layer):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.beta = np.zeros((1, input_dim))
        self.cache_in = None
        self.running_mean = np.zeros((1, input_dim))

    def forward(self, X, train=True):
        if train:
            mu = np.mean(X, axis=0, keepdims=True)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mu
            out = X - mu + self.beta
            return out
        else:
            out = X - self.running_mean + self.beta
            return out

    def backward(self, dY):
        batch_size = dY.shape[0]
        dbeta = np.sum(dY, axis=0, keepdims=True)
        dxhat = dY
        dmu = -1 * np.sum(dY, axis=0, keepdims=True)
        dX =  dmu / batch_size + dxhat
        return dX, [(self.beta, dbeta)]
		
class DilatedConvFlatten(Layer): 
    '''
    a brute force implementation of the dilated convolution forward pass, with additional reshaping
    for ease of implementation, we assume that in_channels=1, as we only need to implement this for the first layer
    convolution filters are initialized in a gaussian manner, similar to the linear layer
    '''
    def __init__(self, kernel, dfactor, out_channels):
        self.kernel = kernel
        self.dfactor = dfactor
        self.out_channels = out_channels
        self.in_shape = None

        self.w = np.random.randn(kernel[0], kernel[1], out_channels) * np.sqrt(2.0 / (kernel[0] * kernel[1]))
    
    def forward(self, X, train=True):
        if self.in_shape == None: self.in_shape = X.shape[1:3]
        batch_size = X.shape[0]
        
        out_shape = (self.in_shape[0] - self.dfactor * (self.kernel[0] - 1), self.in_shape[1] - self.dfactor * (self.kernel[1] - 1))
        out = np.zeros((batch_size, out_shape[0], out_shape[1], self.out_channels))
        for c in range(self.out_channels):
            for i in range(out_shape[0]):
                for j in range(out_shape[1]):
                    for a in range(self.kernel[0]):
                        for b in range(self.kernel[1]):
                            out[:,i,j,c] += self.w[a][b][c] * X[:,i+self.dfactor*a,j+self.dfactor*b]

        return np.reshape(out, (batch_size, -1))

    def backward(self, dY):
        return np.zeros(self.in_shape), []

class Loss(object):
    '''
    Abstract class representing a loss function
    '''
    def get_loss(self):
        raise NotImplementedError('This is an abstract class')

class SoftmaxCrossEntropyLoss(Loss):
    '''
    Represents the categorical softmax cross entropy loss
    '''

    def get_loss(self, scores, labels):
        '''
        Calculates the average categorical softmax cross entropy loss.

        Args:
            scores (numpy.ndarray): Unnormalized logit class scores. Shape (batch_size, num_classes)
            labels (numpy.ndarray): True labels represented as ints (eg. 2 represents the third class). Shape (batch_size)

        Returns:
            loss, grad
            loss (float): The average cross entropy between labels and the softmax normalization of scores
            grad (numpy.ndarray): Gradient for scores with respect to the loss. Shape (batch_size, num_classes)
        '''
        scores_norm = scores - np.max(scores, axis=1, keepdims=True)
        scores_norm = np.exp(scores_norm)
        scores_norm = scores_norm / np.sum(scores_norm, axis=1, keepdims=True)

        true_class_scores = scores_norm[np.arange(len(labels)), labels]
        loss = np.mean(-np.log(true_class_scores))

        one_hot = np.zeros(scores.shape)
        one_hot[np.arange(len(labels)), labels] = 1.0
        grad = (scores_norm - one_hot) / len(labels)

        return loss, grad

