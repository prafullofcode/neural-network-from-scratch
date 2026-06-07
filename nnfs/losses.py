import numpy as np
from activations import Activation_Softmax

class Loss : 
    def calculate(self,output, y ):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss


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

class Activation_Softmax_Loss_CategoricalCrossentropy():


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