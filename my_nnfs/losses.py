import numpy as np
from .activations import Activation_Softmax

class Loss : 
    def calculate(self,output, y ):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss

    def regularization_loss(self, layer):
        regularization_loss = 0

        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss


class Loss_CategoricalCrossEntropy(Loss):
    
    def forward(self,y_pred , y_true):
        no_of_samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred , 1e-7 , 1 - 1e-7)

        #if categorical labels
        if len(y_true.shape) == 1: 
            correct_confidences = y_pred_clipped[range(no_of_samples) , y_true]
        #if one hot coded labels
        if len(y_true.shape) == 2 : 
            correct_confidences = np.sum(y_pred_clipped * y_true , axis = 1)
        
        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods
    
    def backward(self,dvalues, y_true):
        
        samples = len(dvalues)

        labels = len(samples[0])

        if y_true.shape == 1:
            y_true = np.eye(labels)[y_true]
            
        self.dinputs = -y_true / dvalues

        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy(Loss):


    def __init__(self):
        self.activation = Activation_Softmax()                
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self,inputs , y_true):
        
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output , y_true)


    def backward(self,dvalues , y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true =np.argmax(y_true , axis = 1)


        self.dinputs = dvalues.copy()
        self.dinputs[range(samples) , y_true] -=1

        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # Clip to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(sample_losses)

    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        clipped_values = np.clip(dvalues , 1e-7 , 1-1e-7)
        self.dinputs = -(y_true / clipped_values - (1-y_true) / (1-clipped_values)) / outputs
        self.dinputs = self.dinputs / samples


