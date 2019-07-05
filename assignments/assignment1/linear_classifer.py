import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin
from metrics import multiclass_accuracy

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    pred = predictions - np.max(predictions, axis=-1, keepdims=True)
    return np.exp(pred) / np.sum(np.exp(pred), axis=-1, keepdims=True)

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (N, batch_size) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    return -np.log(probs[target_index])

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
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    batch_size = X.shape[0]
    f_num = W.shape[0]
    class_num = W.shape[1]
    
    predictions = np.dot(X, W)
    
    loss, grad = softmax_with_cross_entropy(predictions, target_index)
    
    dW = np.sum(grad.reshape(batch_size, 1, class_num)*X.reshape(batch_size, f_num, 1), axis=0)
  
    return loss, dW


class LinearSoftmaxClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size=100, learning_rate=1e-7, reg=1e-5, epochs=1):
        self.W = None
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs

    def fit(self, X, y):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(self.epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(self.batch_size, num_train, self.batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            for batch_index in batches_indices:
                
                predictions = X[batch_index] @ self.W
                loss, dprediction = softmax_with_cross_entropy(predictions, y[batch_index])
                dW = X[batch_index].T @ dprediction
                reg_loss = self.reg * np.sum(np.square(self.W))
                grad = 2 * np.array(self.W) * self.reg
                loss += reg_loss
                dW += grad
                self.W -= self.learning_rate*dW
            loss_history.append(loss)
#             print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        predictions = X @ self.W
        y_pred = np.argmax(predictions, axis = 1)
        return y_pred
    
    def score(self, X, y):
         return multiclass_accuracy(self.predict(X), y)

                
                                                          

            

                
