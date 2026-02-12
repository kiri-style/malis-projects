import numpy as np

class Perceptron:
    '''
    perceptron algorithm class
    __init__() initialize the model
    train() trains the model
    predict() predict the class for a new sample
    '''

    def __init__(self, alpha):
        '''
        INPUT :
        - alpha : is a float number bigger than 0 
        '''

        if alpha <= 0:
            raise Exception("Sorry, no numbers below or equal to zero. Start again!")
            
        # empty initialization of X and y
        self.X = []
        self.y = []
        # learning rate parameter (initialization input)
        self.alpha = alpha

        self.w = None  # weights initialization
        self.b = 0     # bias initialization

        
    def train(self,X,y, epochs = 1000):
        '''
        INPUT :
        - X : is a 2D NxD numpy array containing the input features
        - y : is a 1D Nx1 numpy array containing the labels for the corresponding row of X
        '''     
        # weights, size of X 
        self.w = np.zeros(X.shape[1])

        for _ in range(epochs):
            for i in range(X.shape[0]):
                # perceptron formula
                prediction = np.dot(X[i], self.w) + self.b
                # apply sign function to make binary prediction
                predicted_label = 1 if prediction >= 0 else -1
                
                # if prediction wrong: update weights and bias
                if predicted_label != y[i]:
                    self.w += self.alpha * y[i] * X[i]
                    self.b += self.alpha * y[i]
   
       
    def predict(self,X_new):
        '''
        INPUT :
        - X_new : is a MxD numpy array containing the features of new samples whose label has to be predicted
        A
        OUTPUT :
        - y_hat : is a Mx1 numpy array containing the predicted labels for the X_new samples
        ''' 
        # compute predictions for each new sample
        predictions = np.dot(X_new, self.w) + self.b #perceptron formula
        y_hat = np.sign(predictions)  # apply sign function to get -1 or 1
        
        return y_hat

    
