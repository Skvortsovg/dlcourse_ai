import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W**2)
    grad = 2*reg_strength*W
    
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (N, batch_size) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    if not isinstance(target_index, np.ndarray):
        target_index = np.array([[target_index]])
    target_index.shape = (-1,)
   
    f_num = predictions.shape[-1]
    batch_size = target_index.shape[0]
    
    # Make correct shape
    pred_copy = predictions.copy().reshape(batch_size, f_num)
    pred_copy -= np.amax(pred_copy, axis=1).reshape(batch_size, 1)
    
    # Compute softmax
    e = np.exp(pred_copy)
    probs = e / np.sum(e, axis=1).reshape(batch_size, 1)
    
    # Compute loss
    s = np.arange(batch_size) # make array of [0, 1, 2, ..., (batch_size-1)]
    target_probs = probs[s, target_index] # using multiple array indexing
    
    loss = -np.average(np.log(target_probs))
    
    
    # Compute gradient
    dprediction = np.copy(probs)
    dprediction[s, target_index] -= 1
    dprediction /= batch_size # normalize gradient
    
    dprediction.shape = predictions.shape
    
    
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X # Will be used in the back pass
        return np.maximum(0, X)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_input = np.ones(self.X.shape)
        d_input[np.where(self.X < 0)] = 0
        d_result = d_out * d_input
        return d_result

    def params(self):
        # Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        result = X @ (self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        d_result = d_out @ (self.W.value).T
        self.W.grad = self.X.T @ d_out
        self.B.grad = np.sum(d_out, axis=0, keepdims=True)
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
