import numpy as np

class Loss : 
    def calculate(self,output, y ):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss


def Loss_CategoricalCrossEntropy(Loss):
    
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

                


