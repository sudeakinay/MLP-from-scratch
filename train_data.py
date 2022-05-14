#############################  MY NEURAL NETWORK MODEL (TRAIN) #############################
import numpy as np                      # allows scienftific computing                                   
import joblib                           # allows load models

class NeuralNetwork:
    def __init__(self, learning_rate, iterations,  input_dimension, hidden_layers):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.input_dimension = input_dimension
        self.hidden_layers = hidden_layers

        if np.any(learning_rate <= 0):
            raise ValueError("learning_rate must be greater than zero")
        if  np.any(iterations <= 0):
            raise ValueError("number of iteration must be greater than zero")

        self.gradient_hidden_to_output = np.random.randn()
        self.gradient_input_to_hidden = np.random.randn()
        self.weights_input_to_hidden = np.random.uniform(-1, 1, (input_dimension, hidden_layers))
        self.weights_hidden_to_output = np.random.uniform(-1, 1, (hidden_layers))
        
        pre_hidden = np.zeros(hidden_layers)
        post_hidden = np.zeros(hidden_layers)
        self.pre_hidden = pre_hidden
        self.post_hidden = post_hidden  
       
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
 
#################################### INPUT TO HIDDEN #################################      

    def predict_input_to_hidden(self, input_vector): ##
        layer_ex = np.dot(input_vector, self.weights_input_to_hidden) + self.gradient_input_to_hidden
        layer_next = self._LReLU(layer_ex)
        prediction = layer_next
        return layer_ex, prediction

    def _computer_gradients_input_to_hidden(self, input_vector, target): ##
        layer_ex = np.dot(input_vector, self.weights_input_to_hidden) + self.gradient_input_to_hidden
        layer_next = self._LReLU(layer_ex)
        prediction = layer_next

        derror_dprediction =  (prediction - target)
        dprediction_dexlayer = self._LReLU_deriv(layer_ex)
        dexlayer_dbias = 1
        dexlayer_dweights = (0 * self.weights_input_to_hidden) + (1 * input_vector)

        derror_dbias = ( derror_dprediction * dprediction_dexlayer * dexlayer_dbias )
        derror_dweights = ( derror_dprediction * dprediction_dexlayer * dexlayer_dweights)
        
        return derror_dweights, derror_dbias

    def _update_parameters_input_to_hidden(self, derror_dbias, derror_dweights): ##
        self.gradient_input_to_hidden = self.gradient_input_to_hidden - (derror_dbias * self.learning_rate)
        self.weights_input_to_hidden = self.weights_input_to_hidden - ( derror_dweights * self.learning_rate)
        return self.weights_input_to_hidden, self.gradient_input_to_hidden


####################################  HIDDEN TO OUTPUT #################################      

    def _update_parameters_hidden_to_output(self, derror_dbias, derror_dweights): ##
        self.gradient_hidden_to_output = self.gradient_hidden_to_output - (derror_dbias * self.learning_rate)
        self.weights_hidden_to_output = self.weights_hidden_to_output - ( derror_dweights * self.learning_rate)
        return self.weights_hidden_to_output, self.gradient_hidden_to_output


    def predict_hidden_to_output(self, input_vector): ##
        layer_ex = np.dot(input_vector, self.weights_hidden_to_output) + self.gradient_hidden_to_output
        layer_next = self._LReLU(layer_ex)
        prediction = layer_next
        return layer_ex, prediction

############################################# TRAIN ##################################
    def train(self, input_vectors, targets, iterations, input_dimension, hidden_layers):
        total_errors = []
        for current_iteration in range(iterations) :
            for sample in range(len(input_vectors[:,0])):
                target = targets[sample]          
                for node in range(self.hidden_layers):
                    self.pre_hidden[node], self.post_hidden[node] = self.predict_input_to_hidden(input_vectors[node])
                for hidden_node in range(hidden_layers):
                    derror_dweights , derror_dbias = self._computer_gradients_input_to_hidden(self.post_hidden[hidden_node], target)
                    for input_node in range(input_dimension):
                        input_value = input_vectors[sample, input_node]
                        derror_dweights_input, derror_dbias_input = self._computer_gradients_input_to_hidden(input_value, self.pre_hidden[hidden_node])
                        self.weights_input_to_hidden[input_node, hidden_node]  = self._update_parameters_input_to_hidden(derror_dbias_input, derror_dweights_input)
                    self.weights_hidden_to_output[hidden_node] = self._update_parameters_hidden_to_output(derror_dbias, derror_dweights)
            
        if current_iteration % 100 == 0: 
            total_error = 0
            for data_instance_index in range(len(input_vectors)):
                data_point = input_vectors[data_instance_index]
                target = targets[data_instance_index]
                prediction = self.predict_hidden_to_output(data_point)      
                error = np.square(prediction - target)      
                total_error = total_error + error
            total_errors.append(total_error)      
                
        for i in total_errors:
            if i <= 10^(-6):
                break  

        return total_errors, self.weights_hidden_to_output, self.weights_input_to_hidden, self.gradient_hidden_to_output, self.gradient_input_to_hidden


######################################## TEST FUNCTIONS ##################################

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
input_dimension = 4
hidden_layers = 2
model = NeuralNetwork(learning_rate, iterations, input_dimension, hidden_layers)
joblib.dump(model, "my_first_model")
loaded_model = joblib.load("my_first_model")
