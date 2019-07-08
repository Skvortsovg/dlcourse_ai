import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        
        self.reg = reg
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.act1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.act2 = ReLULayer()

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Clear gradients
        params = self.params()
        for p in params:
            params[p].grad = 0
        
        X = self.fc1.forward(X)
        X = self.act1.forward(X)
        
        X = self.fc2.forward(X)
        
        loss, d_pred = softmax_with_cross_entropy(X, y)
        
        # X = self.act2.forward(X)
        
        # d_act2 = self.act2.backward(d_pred)
        d_fc2 = self.fc2.backward(d_pred)
        
        d_act1 = self.act1.backward(d_fc2)
        d_fc1 = self.fc1.backward(d_act1)
        
        for p in params:
            regular_loss, regular_grad = l2_regularization(params[p].value, self.reg)
            loss += regular_loss
            params[p].grad += regular_grad
        return loss
        
        

    def predict(self, X):
        """
        Produces classifier predictions on the set
    
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        pred = np.zeros(X.shape[0], np.int)
        
        X = self.fc1.forward(X)
        X = self.act1.forward(X)
        X = self.fc2.forward(X)
        X = self.act2.forward(X)
        
        pred = np.argmax(X, axis=1)

        return pred

    def params(self):
        result = {}

        for param in self.fc1.params():
            result[param + '_fc1'] = self.fc1.params()[param]
            
        for param in self.fc2.params():
            result[param + '_fc2'] = self.fc2.params()[param]

        return result
