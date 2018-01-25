import numpy as np                              # Import numpy - for operations on matrices
import matplotlib.pyplot as plt                 # Import pyplot for plotting graphs
from sklearn.datasets import load_digits        # Import MNIST dataset using sklearn

from sklearn.preprocessing import StandardScaler        # For scaling the data
from sklearn.model_selection import train_test_split    # For splititng data into test and train sets

# Let's start by defining some functions for the process:

#one-hot encoding of the output data
def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))                 # Since we have ten digits so we'll have one hot encoded vectors of length 10
    for i in range(len(y)):                         # Traverse over the data in y
        y_vect[i, y[i]] = 1                         # For the given data row, assign the 'target value'-th column in the row, the value '1'
    return y_vect

def sigmoid(x):                                     # Sigmoid function
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):                               # Derivative of sigmoid function
    return sigmoid(x) * (1 - sigmoid(x))

#set up random values for the weights and the biases
import numpy.random as r
def initialize_weights_bias(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))                    # Initialize weigths - For 'L' layers, we need weights for L-1 layers - input layer is excluded
        b[l] = r.random_sample((nn_structure[l],))                                      # Initialize bias
    return W, b


#set up the mean accumulation
def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))                       # Initialize matrix to hold mean value for weights - same dimension as 'W'
        tri_b[l] = np.zeros((nn_structure[l],))                                         # Initialize matrix to hold mean value for bias - same dimension as 'b'
    return tri_W, tri_b


#set up the feedforward function
def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):                             # Traverse over the loop for all layers except the output layer
        # if it is the first layer, then the input is x, otherwise,
        # the input is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l]                       # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[l+1] = sigmoid(z[l+1])                                # h^(l) = f(z^(l))
    return h, z


#calculate delta for the output layer (error)
def calculate_out_layer_delta(y, h_out, z_out):
    return -(y-h_out) * sigmoid_deriv(z_out)                    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))


#calculate delta for the hidden layers
def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * sigmoid_deriv(z_l)



#the main training function:
def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.25):                # Iterate over the entire train set over 3000 times for training, keep the learning rate alpha to 0.25
    W, b = initialize_weights_bias(nn_structure)                            # Initialize weights and bias
    cnt = 0                                                                 # Keep a track of no of epochs
    m = len(y)                                                              # Length of 'y' - equal to no of samples
    avg_cost_func = []                                                      # Keep a track of avg_cost in every iteration - to plot graph in the end
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:                                                   # Perform gradient descent iter_num times
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))               # Print ITeration number after every 1000 iterations
        tri_W, tri_b = init_tri_values(nn_structure)                        # Define the matrix to hold mean values
        avg_cost = 0                                                        # Initialize average cost to 0
        for i in range(len(y)):                                             # Traverse over every sample in training set
            delta = {}                                                      # Initialize dict to hold values for derivative
            # perform the feed forward pass and return the stored h and z values, to be used in the
            # gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):                                      # Check if it is the last layer
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])    # Calculate output from the output layer
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))                   # Add value to the accumulated average cost
                else:
                    if l > 1:                                                       # Check if it is layer greater than '1', not input layer, greater than input layer
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])   # Calculate delta for the hidden layer
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))      # Accumulate error in Weights 'W'
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]                                                              # Accumulate error in bias 'b'
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):                       # Traverse over each layer to perform gradient descent
            W[l] += -alpha * (1.0/m * tri_W[l])                             # Update weights
            b[l] += -alpha * (1.0/m * tri_b[l])                             # Update bias

        # Calculate average cost
        avg_cost = 1.0/m * avg_cost                             # Find average cost
        avg_cost_func.append(avg_cost)                          # Append to avg_cost_fun list - for plotting graph
        cnt += 1
    return W, b, avg_cost_func


# Function to predict values through the trained model
def predict_y(W, b, X, n_layers):
    m = X.shape[0]                                                  # Get the number of samples in X using shape
    y = np.zeros((m,))                                              # Define a matrix of zeros for the output 'y'
    for i in range(m):                                              # Traverse over all the samples in the test set passed to this function
        h, z = feed_forward(X[i, :], W, b)                          # Pass the values through the feed_forward function
        y[i] = np.argmax(h[n_layers])                               # Pick the highest value from the output vector as the answer - map it to 1 and map the rest to 0 - This will then be a one-hot encoded vector
    return y




# Let's start using the above functions now

digits = load_digits()                          # Load the digits
print(digits.data.shape)                        # Print the shape of dataset

# plot a sample digit picture
plt.gray()                                      # Initialize a gray figure
plt.matshow(digits.images[1])                   # Display a sample digit - matshow displays an array as a matrix
plt.show()                                      # Show the plot
print digits.data[1,:]                          # print the digit value as present in the database

#scale the data to improve convergence of nueral network
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)               # This will scale the arrays of digits so that the mean becomes zero and standard deviation is reduced, this wa convergence occurs comparatively faster


#split the data into test and train (60% training data, 40% test data)
y = digits.target                                                           # Set 'y' to targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)    # Split the data into test and train - 40 percent test


y_v_train = convert_y_to_vect(y_train)                              # Convert out output 'y' for test set to one hot encoded vectors
y_v_test = convert_y_to_vect(y_test)                                # Convert out output 'y' for train set to one hot encoded vectors
# print y_train[0], y_v_train[0]

nn_structure = [64, 30, 10]                                         # Define the number of nodes in each layer - 64 nodes in input layer, 30 nodes in thehidden layer, 10 nodes in the outputlayer

W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)    # Start training the neural network on the train dataset

plt.plot(avg_cost_func)                                             # Plot the avg cost function for the training carried out
plt.ylabel('Average J')                                             # Label y-axis of plot
plt.xlabel('Iteration number')                                      # Label x-axis for plot
plt.show()                                                          # Display the plot

from sklearn.metrics import accuracy_score                          # Calculate accuracy on the test dataset
y_pred = predict_y(W, b, X_test, 3)                                 # Predict values for the test set using the trained model
print "Accuracy: " , accuracy_score(y_test, y_pred)*100             # Calculate and print accuracy - compare the predicted and actual values for the test set
