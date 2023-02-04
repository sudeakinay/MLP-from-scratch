#############################  MY FIRST NEURAL NETWORK  #############################

import numpy as np                      # allows scienftific computing
import pandas as pd                     # allows easy manipulation of data structures   
from pandas import DataFrame as df     
import matplotlib.pyplot as plt
from csv import reader
import math

class NeuralNetwork:
    # classes are the main building blocks of object-oriented programming
    # the NeuralNetwork class generates random start values for the weights and bias variables

    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    # we'll use nonlinear functions are called activation functions: relu activation function
    
    def _LReLU(self, x):
        # a = 0.01 olsun
        if x < 0:
            x = 0.01 * x 
        elif x >= 0:
            x = x  
        return x

    def _LReLU_deriv(self, x):
        if x < 0:
            x = 0.01 
        elif x >= 0:
            x = 1 
        return x
    
    def _tanh (self, x):
        result = math.tanh((2 / (1 + np.exp(-2*x))) - 1)
        return result
        
    def _tanh_deriv (self, x):
        result_deriv = 1 - (self._tanh(x)**2)
        return result_deriv

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    # this is a perfect function for our problem because we have just 2 outputs: 0 and 1
    # and this function follows the bernoulli distribution

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._LReLU(layer_1)
        prediction = layer_2
        return prediction

################# ADJUSTING THE PARAMETERS WITH BACKPROPAGATION #####################

    def _computer_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._LReLU(layer_1)
        prediction = layer_2

        # to adjust the weights we'll use the gradient descent and backpropagation algorithms
        # first, we'll predict the error
        # the function used to measure the error is called the cost function or loss function
        # we used the mean squared error (mse) as our cost function

        # we compute the MSE in 2 steps:
        # 1.compute the difference between the prediction and the target
        # 2.multiply the result by itself

        # since the MSE is the squared difference between the prediction and the correct result
        # with this metric we’ll always end up with a positive value
     
        derror_dprediction =  (prediction - target)

        # we'll take the derivative of layer_1 with respect to the bias
        # the bias variable is an independent variable 
        # so the result after applying the power rule is 1:

        dprediction_dlayer1 = self._LReLU_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        # the derivative of the dot product is the derivative of the first vector multiplied 
        # by the second vector, plus the derivative of the second vector multiplied by the first vector

        # we want to take the derivative of the error function with respect to the bias, derror_dbias
        # then we'll keep going backward, taking the partial derivatives until we find the 'bias' variable:

        derror_dbias = ( derror_dprediction * dprediction_dlayer1 * dlayer1_dbias )
        derror_dweights = ( derror_dprediction * dprediction_dlayer1 * dlayer1_dweights )
        
        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):

        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - ( derror_dweights * self.learning_rate)

############### in short: ##################
# we pick a random instance from the dataset 
# compute the gradients 
# and update the weights and the bias 
# we also compute the cumulative error every 100 iterations and save those results in an array. 
# we'll plot this array to visualize how the error changes during the training process

################################# TRAINING THE NETWORK WITH MORE DATA ################################

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations) :
            # pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index] 
            target = targets[random_data_index]

            # compute the gradients and update the weights #
            derror_dbias, derror_dweights = self._computer_gradients(
                input_vector, target
            )

            # update the bias and the weights which we defined in the previous code block
            self._update_parameters(derror_dbias, derror_dweights)

            # # measure the cumulative error for all the instances
            if current_iteration % 100 == 0: 
                # check the current iteration index is a multiple of 100
                # we do this to observe how the error changes every 100 iterations.
                cumulative_error = 0
                # loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]
 
                    prediction = self.predict(data_point)       # computes the prediction result
                    error = np.square(prediction - target)      # computes the error for every instance

                    # we accumulate the sum of errors using the cumulative_error variable
                    # this is for plot a point with the error for ALL the data instances
                    cumulative_error = cumulative_error + error
                
                # we append the error to cumulative_errors: the array that stores the errors
                cumulative_errors.append(cumulative_error)      # we'll use this array to plot the graph

################################## STOP CRITERIA ##################################
            # if (current_iteration >= 10000): this is not a good solution to make a stop criteria
            #     break

            for i in cumulative_errors:
                if i <= 10^(-5):
                    break  

        return cumulative_errors, self.weights, self.bias

############## IRIS DATASETI EKLENIR #######################


df = pd.read_csv("dataset.csv")                      # add data
input_vectors = df.drop(["Species"], axis = 1)
targets = df.Species

df = pd.read_csv("dataset.csv")                      # add data
input_vectors = df.drop(["Species"], axis = 1)
targets = df.Species

'''
data seti okuma işlemi asagidaki gibi de yapilabilirdi
    input_vectors = []
    targets = []
    with open('basic_nn/dataset.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
        if header != None:
            for row in csv_reader:
                input_vectors.append(row)

    targets = np.delete(input_vectors, [0, 1, 2, 3], 1)
    input_vectors = np.delete(input_vectors, 4, 1)

    input_vectors = np.array(input_vectors, dtype = float)
    targets = np.array(targets, dtype = float)
'''

###################### EGITIM BASLANGICI ################################

learning_rate = 0.001
neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 100000000)

test_data = [6.4, 3.1, 5.5, 1.8]
test_data_prediction = neural_network.predict(test_data)

'''
prediction işlemi asagidaki gibi de yapilabilirdi
    test_data = [6.4, 3.1, 5.5, 1.8]
    training_error, final_weights, final_bias = neural_network.train(input_vectors, targets, 100000000)
    print('weights: %f, bias: %f', final_weights, final_bias)
    test_data_prediction = np.dot(test_data, final_weights) + final_bias
'''

if(test_data_prediction <= 0.69):
    print('Predict: Iris-setosa', test_data_prediction)
elif(test_data_prediction > 0.69 and test_data_prediction < 1.5 ):
    print('Predict: Iris-versicolor', test_data_prediction)
else:
    print('Predict: Iris-virginica', test_data_prediction)

################ hata grafiği ################
plt.plot(training_error)
plt.xlabel('iterations')
plt.ylabel('error for all the training instances')
plt.show()


dataset = pd.read_csv('dataset.csv')
colors = ['blue', 'red']
targets = [0, 1]

for i in range(2):
    x = dataset[dataset['Species'] == targets[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label =targets[i])
plt.xlabel('SepalWidthCm')
plt.xlabel('PetalWidthCm')
plt.legend()
plt.show