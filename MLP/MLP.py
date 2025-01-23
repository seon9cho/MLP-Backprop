import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, max_iter=1000, shuffle=True):
        """ Initialize class with chosen hyperparameters.
        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        Example:
            mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.num_layers = len(hidden_layer_widths) + 1
        self.lr = lr
        self.momentum = momentum
        self.max_iter = max_iter
        self.shuffle = shuffle
        
        self.z = [0] * self.num_layers
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.train_loss = []
        self.val_loss = []
        self.val_score = []
        self.num_iter = 0

    def fit(self, X, y, initial_weights=None, deterministic=False, val_size=0.25):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]
        self.initial_weights = self.initialize_weights() if not initial_weights else initial_weights
        self.weights = self.initial_weights.copy()
        if deterministic:
            for i in range(deterministic):
                self._train(X, y)
        else:
            X, X_val, y, y_val = train_test_split(X, y, test_size=val_size)
            break_cond = "Maximum number of iterations reached."
            for i in range(self.max_iter):
                self._train(X, y)
                self._validate(X_val, y_val)
                stop = self._stopping_criterion()
                if stop:
                    break_cond = "Model converged after {} iterations.".format(i)
                    break
            self.num_iter = i
            print(break_cond)
        
        return self
    
    def _train(self, X, y):
        initial_weights = self.weights.copy()
        temp_loss = []
        if self.shuffle:
            _X, _y = self._shuffle_data(X, y)
        else:
            _X, _y = X, y
        for x,t in zip(_X, _y):
            t_hat = self._forward(x)
            loss = ((1/2) * (t - t_hat)**2).mean()
            temp_loss.append(loss)
            self._backward(t_hat, t)
        self.train_loss.append(sum(temp_loss) / len(temp_loss))
    
    def _validate(self, X, y):
        temp_loss = []
        for x, t in zip(X, y):
            t_hat = self._forward(x)
            loss = ((1/2) * (t - t_hat)**2).mean()
            temp_loss.append(loss)
        self.val_loss.append(sum(temp_loss) / len(temp_loss))
        score = self.score(X, y)
        self.val_score.append(score)
        
    def _forward(self, x):
        self.a = []
        for i in range(self.num_layers):
            x = np.append(x, 1)
            self.a.append(x)
            x = self.weights[i] @ x
            x = self.sigmoid(x)
        return x
    
    def _backward(self, t_hat, t):
        delta = (t - t_hat) * t_hat * (1 - t_hat)
        dw = np.outer(delta, self.a[-1])
        self.z[-1] = self.lr*dw + self.momentum*self.z[-1]
        for i in range(1, self.num_layers):
            derivative = self.a[-i][:-1] * (1 - self.a[-i][:-1])
            delta = (self.weights[-i][:,:-1].T @ delta) * derivative
            dw = np.outer(delta, self.a[-i-1])
            self.z[-i-1] = self.lr*dw + self.momentum*self.z[-i-1]
        for i in range(self.num_layers):
            self.weights[i] += self.z[i]
            
    def _stopping_criterion(self):
        bssf = min(self.val_loss)
        if min(self.val_loss[-10:]) == bssf:
            return False
        else:
            return True
    
    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pred = []
        for x in X:
            y = self._forward(x).argmax()
            pred.append(y)
        return np.array(pred)
        

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """
        weights = []
        input_layer = np.random.normal(size=(self.hidden_layer_widths[0], self.input_dim+1))
        weights.append(input_layer)
        for i in range(self.num_layers - 2):
            w1, w2 = self.hidden_layer_widths[i:i+2]
            layer = np.random.normal(size=(w2, w1+1))
            weights.append(layer)
        output_layer = np.random.normal(size=(self.output_dim, self.hidden_layer_widths[-1]+1))
        weights.append(output_layer)
        return weights

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        pred = self.predict(X)
        y_a = y.argmax(axis=1)
        return sum(pred==y_a) / len(pred)
    
    def average_loss(self, X, y):
        temp_loss = []
        for x, t in zip(X, y):
            t_hat = self._forward(x)
            loss = ((1/2) * (t - t_hat)**2).mean()
            temp_loss.append(loss)
        return sum(temp_loss) / len(temp_loss)
    
    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        Xy = np.hstack([X, y])
        np.random.shuffle(Xy)
        return Xy[:, :-self.output_dim], Xy[:, -self.output_dim:]

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights