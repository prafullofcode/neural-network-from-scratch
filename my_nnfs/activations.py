import numpy as np

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis = 1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values , axis=1 , keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):

            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

class Activation_Sigmoid: 
    def forward(self,inputs):
        self.inputs = inputs
        clipped_inputs = np.clip(inputs, -500, 500)
        self.output = 1 / (1 + np.exp(-clipped_inputs))

    def backward(self,dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output) 

class Activation_Linear:
    def forward(self,inputs):
        self.inputs = inputs 
        self.output = inputs 

    def backward(self,dvalues):
        self.dinputs = dvalues.copy()

