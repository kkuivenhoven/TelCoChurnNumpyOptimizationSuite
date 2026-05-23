import numpy as np
import pandas as pd
import math

def master_training(training_type):
	if training_type == 'gd':
		print("GD")
	elif training_type == 'momentum':
		print("Momentum")
	elif training_type == 'rmsprop':
		print("RMSProp")
	elif training_type == 'adam':
		print("Adam")
	else:
		print("Not possible")

def initialize_momentum(parameters):
	v = {}
	L = len(parameters) // 2
	for l in range(1, L + 1):
		v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
		v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
	return v

def initialize_rmsprop(parameters):
	s = {}
	L = len(parameters) // 2
	for l in range(1, L + 1):
		s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
		s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
	return s
		
def initialize_adam(parameters):
	s = {}
	v = {}
	L = len(parameters) // 2
	for l in range(1, L + 1):
		s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
		s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
		v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
		v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
	return v, s
		
		
def create_mini_batches(X, Y):
	batches = []
	length = X.shape[1]
	ending_value = ceil(length/64)
	# must shuffle incase raw data is ordered in some way
	shuffled_indices = np.random.permutation(length)
	X_shuffled = X[:, shuffled_indices]
	Y_shuffled = Y[:, shuffled_indices]
	for i in range(1, ending_value + 1):
		mini_batch_X = X_shuffled[:, (i-1)*64:i*64]
		mini_batch_Y = Y_shuffled[:, (i-1)*64:i*64]
		batches.append((mini_batch_X, mini_batch_Y))
	return batches

# hardcode a fixed random seed such that every optimizer starts with the 
# exact same weight matrices (W, b)
np.random.seed(42)

dataFrame = pd.read_csv('WA-Customer-Churn.csv')
dataFrame["TotalCharges"] = pd.to_numeric(dataFrame["TotalCharges"].str.strip(), errors='coerce').fillna(0).astype(float)
dataFrame = dataFrame.drop('customerID', axis=1)

dataFrame = pd.get_dummies(dataFrame)
dataFrame = dataFrame.astype(float)

Y_df = dataFrame["Churn_Yes"] # use this column since we need the model to predict 
 							  # the churning/cancelation ("positive" event)

dataFrame = dataFrame.drop('Churn_No', axis=1)
dataFrame = dataFrame.drop('Churn_Yes', axis=1)

# Shape is (m, nx) where m is rows (customers) and nx is columns (features) 
# For this, need (nx, m)

# Transpose such that each column represents a customer and each row represents a feature
Y_df = Y_df.values.reshape(-1, 1)
Y = Y_df.T
# print("dataframe shape")
# print(dataframe.shape)
X_df = dataFrame.T
X = X_df.to_numpy()
print("X shape")
print(X.shape)
print("Y shape")
print(Y.shape)

# [Input, Hidden1, Hidden2, Output]
layer_dims = [45, 20, 7, 1]
learning_rate = 0.01

create_mini_batches(X_df)

master_training('gd')
master_training('momentum')
master_training('rmsprop')
master_training('adam')
master_training('blah')
# Need to run Mini-Batch Gradient Descent against Momentum, RMSprop and Adam
# ever single one of them will loop through the exact same collection of size-64 batches

# need tracking structures to store the "history" of your gradients
# do this by initializizing dictionaries:
# v for velocity
# s for squared variance (standard deviation)
# these must mirror the exact shapes of your weights (W) and biases (b)
