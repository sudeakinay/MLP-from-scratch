#############################  MY NEURAL NETWORK MODEL (TRAIN) #############################
import numpy as np                      # allows scienftific computing                                   
import joblib                           # allows load models

class NeuralNetwork:
    def __init__(self, learning_rate, iterations):
        self.weights = np.array([np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()

        self.learning_rate = learning_rate
        self.iterations = iterations
        if np.any(learning_rate <= 0):
            raise ValueError("learning_rate must be greater than zero")
        if  np.any(iterations <= 0):
            raise ValueError("number of iteration must be greater than zero")
       
    def _LReLU(self, x):
        x = np.array(x)
        if (x < 0).any():
            x = 0.01 * x 
        elif (x >= 0).any():
            x = x  
        return x
 
    def _LReLU_deriv(self, x):
        x = np.array(x)
        if (x < 0).any():
            x = 0.01 
        elif (x >= 0).any():
            x = 1 
        return x

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._LReLU(layer_1)
        prediction = layer_2
        return prediction

    def _computer_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._LReLU(layer_1)
        prediction = layer_2

        derror_dprediction =  (prediction - target)
        dprediction_dlayer1 = self._LReLU_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = ( derror_dprediction * dprediction_dlayer1 * dlayer1_dbias )
        derror_dweights = ( derror_dprediction * dprediction_dlayer1 * dlayer1_dweights)
        
        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - ( derror_dweights * self.learning_rate)
        return self.weights, self.bias

    def train(self, input_vectors, targets, iterations):
        total_errors = []
        for current_iteration in range(iterations) :
            random_data_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index] 
            target = targets[random_data_index]
            derror_dbias, derror_dweights = self._computer_gradients(input_vector, target)
            self._update_parameters(derror_dbias, derror_dweights)

            if current_iteration % 100 == 0: 
                total_error = 0
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]
                    prediction = self.predict(data_point)       
                    error = np.square(prediction - target)      
                    total_error = total_error + error
                total_errors.append(total_error)      
                
            for i in total_errors:
                if i <= 10^(-5):
                    break  

        return total_errors, self.weights, self.bias

    def test_function(self, test_data, final_weights, final_bias):
        test_data_pred = []
        prediction = np.dot(test_data, final_weights) + final_bias
        for i in range(len(test_data)):
            if prediction[i] <= 0.8 :     # optimizing results
                test_data_pred.append(0)
                print("Setosa")
            elif (prediction[i] > 0.8).any() and (prediction[i] <= 1.1).any() :
                test_data_pred.append(1)
                print("Versicolor")
            elif prediction[i] >= 1.1:
                test_data_pred.append(2)  
                print("Virginica")  
        return test_data_pred

    def new_data_prediction(self, new_data, final_weights, final_bias):

        new_data_result = np.dot(new_data, final_weights) + final_bias
        if new_data_result <= 0.69:         # optimizing results
            new_data_result = 0
            print("Setosa")               
        elif (new_data_result> 0.69) and (new_data_result <= 1.5):           
            new_data_result = 1 
            print("Versicolor")            
        elif new_data_result >= 1.5:
            new_data_result = 2
            print("Virginica") 
        return new_data_result

learning_rate = 0.01
iterations = 100000
model = NeuralNetwork(learning_rate, iterations)
joblib.dump(model, "my_first_model")
loaded_model = joblib.load("my_first_model")
