#############################  MY NEURAL NETWORK MODEL (TEST) #############################

import numpy as np                      # allows scienftific computing                  
import pandas as pd                     # allows easy manipulation of data structures          
from train import NeuralNetwork         # my very first Neural Network model
from pandas import DataFrame as df           
import matplotlib.pyplot as plt

learning_rate = 0.01
iterations = 100000
loaded_model = NeuralNetwork(learning_rate, iterations)

df = pd.read_csv("basic_nn/train_data.csv")          # add train data
train_data = df.drop(["Species"], axis = 1)
train_results = df.Species
train_data = np.array(train_data)
train_results = np.array(train_results)

nf = pd.read_csv("basic_nn/test_data.csv")           # add test data
test_data = nf.drop(["Species"], axis = 1)
test_results = nf.Species
test_data = np.array(test_data)
test_results =np.array(test_results)

# test_errors, final_weights, final_bias = loaded_model.train(test_data, test_results, iterations)
# print('final weights = ', final_weights)
# print('final ias = ', final_bias)

# sonuçta bulunan ağırlık değerleri aşağıdaki gibidir:

final_weights =[-0.21989351, -0.26251323,  0.46154674,  0.53166462]
final_bias = 0.510874753594822

############################# test data controlling ############################

test_data_pred = loaded_model.test_function(test_data, final_weights, final_bias)
if (test_results == test_data_pred).any():
    print("The test results are Correct!")
else:
    print("The test results are False!")

######################### new data prediction ###############################

new_data = list(map(float, input("\n Enter the numbers: ").strip().split()))[:4]

if len(new_data) < 4:
    print("you entered less than 4 feature. please give only 4 feature data!")    
elif len(new_data) > 4:
    print("you entered more than 4 feature. please give only 4 feature data!")
new_data_result = loaded_model.new_data_prediction(new_data, final_weights, final_bias)
print(new_data_result)

################################ error graph ####################################

# test_errors, final_weights, final_bias = loaded_model.train(test_data, test_results, iterations)
# plt.plot(test_errors)
# plt.xlabel('iterations')
# plt.ylabel('error for all the training instances')
# plt.show()
