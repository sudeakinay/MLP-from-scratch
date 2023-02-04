### MLP FROM SCRATCH ###


import numpy as np
from sklearn.datasets import load_iris
import random
import matplotlib.pyplot as plt

class MLP():
    def __init__(self,learning_rate, input_size, layers, output_size):
        self.input_size = input_size
        self.layers = layers
        self.output_size = output_size
        self.learning_rate = learning_rate

        np.random.seed(0)

        model = {}  # directionary
        self.model = model
        # First Layer
        self.model['W1'] = np.random.randn(self.input_size, self.layers[0])
        self.model['b1'] = np.zeros((1, self.layers[0]))
        # Second Layer
        self.model['W2'] = np.random.randn(self.layers[0], self.layers[1])
        self.model['b2'] = np.zeros((1, self.layers[1]))
        # Third/Output Layer
        self.model['W3'] = np.random.randn(self.layers[1], self.output_size)
        self.model['b3'] = np.zeros((1, self.output_size)) 

        self.activation_outputs = None
        self.math_eq = Math_Eq()

    def forward(self, feature):
        W1, W2, W3 = self.model['W1'], self.model['W2'], self.model['W3']
        b1, b2, b3 = self.model['b1'], self.model['b2'], self.model['b3']

        Z1 = np.dot(feature, W1) + b1
        activated1 = self.math_eq.ReLU(Z1)

        Z2 = np.dot(feature, W2) + b2
        activated2 = self.math_eq.tanh(Z2)

        Z3 = np.dot(feature, W3) + b3
        output = self.math_eq.ReLU(Z3)

        # self.activation_outputs = (activated1, activated2, output)
        return activated1, activated2, output

    def backward(self, feature, label):
        W1, W2, W3 = self.model['W1'], self.model['W2'], self.model['W3']
        b1, b2, b3 = self.model['b1'], self.model['b2'], self.model['b3']
        feature_shape = feature.shape[0]

        # activated1, activated2, output = self.activation_outputs
        activated1, activated2, output = self.forward(feature)
        self.delta3 = np.subtract(output, label)
        self.deriv_W3 = np.dot(activated2.T, self.delta3)
        self.deriv_b3 = np.sum(self.delta3, axis=0)

        self.delta2 = (1 - np.square(activated2)) * np.dot(self.delta3, W3.T)
        self.deriv_W2 = np.dot(activated1.T, self.delta2)
        self.deriv_b2 = np.sum(self.delta2, axis=0)

        self.delta1 = (1 - np.square(activated1)) * np.dot(self.delta2, W2.T)
        self.deriv_W1 = np.dot(feature.T, self.delta1)
        self.deriv_b1 = np.sum(self.delta1, axis=0)

        # update the model parameters using gradient descent
        self.model['W1'] -= self.learning_rate * self.deriv_W1
        self.model['b1'] -= self.learning_rate * self.deriv_b1

        self.model['W2'] -= self.learning_rate * self.deriv_W2
        self.model['b2'] -= self.learning_rate * self.deriv_b2

        self.model['W3'] -= self.learning_rate * self.deriv_W3
        self.model['b3'] -= self.learning_rate * self.deriv_b3

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output, axis = 1)

    def summary(self):
        self.W1, self.W2, self.W3 = self.model['W1'], self.model['W2'], self.model['W3']
        activated1, activated2, output = self.activation_outputs

        print('W1', self.W1.shape)
        print('activated1', activated1.shape)

        print('W2', self.W3.shape)
        print('activated2', activated2.shape)

        print('W3', self.W3.shape)
        print('output', output.shape)

    def loss(self, y_oht, predict):
        loss = - np.mean(y_oht - predict)
        return loss

    def one_hot_encoding(self, y, depth):
        y_shape = y.shape[0]
        y_oht = np.zeros((y_shape, depth))
        y_oht[np.arange(y_shape), y] = 1
        return y_oht

    def train(self, feature, label, epochs, learning_rate, logs = True):
        training_loss = []
        classes = 3
        label_OHT = self.one_hot_encoding(label, classes)

        for ix in range(epochs):
            predict = self.forward(feature)
            loss = self.loss(label, predict)
            print(label_OHT)
            loss = label_OHT - predict
            training_loss.append(1)
            self.backward(feature, label_OHT, learning_rate)
            
            if(logs and ix %50 == 0):
                print('Epoch: %d Loss: %.4f' % (ix, loss))

            for i in training_loss:
                if i <= 10^(-5):
                    break  

        return training_loss


class Math_Eq():
    def __init__(self) -> None:
        pass

    def sigmoid(self, x):
        return {(lambda x : 1/ (1+ np.exp(-1*x)))}
    def deriv_sigmoid(self, x):
        return {(lambda x: self.x*(1-x))}
        
    def tanh(self, x):
        return {(lambda x: np.tanh(x))}
    def deriv_tanh(self, x):
        return {(lambda x: 1-x**2)}

    def ReLU(self, x):
        return {(lambda x: x*(x>0))}
    def deriv_ReLU(self, x):
        return {(lambda x: 1 * (x >0))}

    def softmax(self, a):
        e_pa = np.exp(a)
        ans = e_pa / np.sum(e_pa, axis=1, keepdims=True)
        return ans


iris_data = load_iris()
# print(iris_data['DESCR'])
def separate_data():
    A = iris_dataset[0:40]
    tA = iris_dataset[40:50]
    B = iris_dataset[50:90]
    tB = iris_dataset[90:100]
    C = iris_dataset[100:140]
    tC = iris_dataset[140:150]

    train = np.concatenate((A, B, C))
    test = np.concatenate((tA, tB, tC))
    return train, test

train_porcent = 80
test_porcent = 20
iris_dataset = np.column_stack((iris_data.data, iris_data.target.T)) # join X and Y
iris_dataset = list(iris_dataset)
random.shuffle(iris_dataset)

Filetrain, Filetest = separate_data()

train_X = np.array([i[:4] for i in Filetrain])
train_y = np.array([i[4] for i in Filetrain]).astype(int)
test_X = np.array([i[:4] for i in Filetrain])
test_y = np.array([i[4] for i in Filetrain]).astype(int)


epochs = 500
learning_rate = 0.01
model = MLP(learning_rate, 4, [4, 4], 3)
# forward = model.forward([train_data])
# print(forward)
# model.summary()

losses = model.train(train_X, train_y, epochs, learning_rate)
plt.plot(losses) 
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
























iris_data = load_iris()
#print(iris_data['DESCR'])
def separate_data():
    A = iris_dataset[0:40]
    tA = iris_dataset[40:50]
    B = iris_dataset[50:90]
    tB = iris_dataset[90:100]
    C = iris_dataset[100:140]
    tC = iris_dataset[140:150]

    train = np.concatenate((A, B, C))
    test = np.concatenate((tA, tB, tC))
    return train, test

train_porcent = 80
test_porcent = 20
iris_dataset = np.column_stack((iris_data.data, iris_data.target.T)) # join X and Y
iris_dataset = list(iris_dataset)
random.shuffle(iris_dataset)

Filetrain, Filetest = separate_data()

train_X = np.array([i[:4] for i in Filetrain])
train_y = np.array([i[4] for i in Filetrain])
test_X = np.array([i[:4] for i in Filetrain])
test_y = np.array([i[4] for i in Filetrain])

# # Plot our training samples
# plt.subplot(1, 2, 1)
# plt.scatter(train_X[:, 0], train_X[:, 1], c = train_y, cmap = cm.viridis)
# plt.xlabel(iris_data.feature_names[0])
# plt.ylabel(iris_data.feature_names[1])

# plt.subplot(1, 2, 2)
# plt.scatter(train_X[:, 2], train_X[:, 3], c = train_y, cmap = cm.viridis)
# plt.xlabel(iris_data.feature_names[2])
# plt.ylabel(iris_data.feature_names[3])
# #plt.show()

# # Plot our training samples
# plt.subplot(1, 2, 1)
# plt.scatter(test_X[:, 0], test_X[:, 1], c = test_y, cmap = cm.viridis)
# plt.xlabel(iris_data.feature_names[0])
# plt.ylabel(iris_data.feature_names[1])

# plt.subplot(1, 2, 2)
# plt.scatter(test_X[:, 2], test_X[:, 3], c = test_y, cmap = cm.viridis)
# plt.xlabel(iris_data.feature_names[2])
# plt.ylabel(iris_data.feature_names[3])
# #plt.show()

