"""
Deep Learning Optimizer Benchmark Suite

This script implements a multi-layer neural network from scratch using NumPy to predict customer churn.
This project serves as a benchmark to study the behaviors of 4 different optimizers: Adam, RMSProp,
Momentum, and Gradient Descent. The behaviors evaluated are convergence speed, cost reduction and 
the structural behavior. 

The code handles custom mini-batch slicing, data normalization to prevent gradient explosion and 
outputs comparative performance metrics across varying epoch milestones.
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def sigmoid(z):
	s = 1 / (1 + np.exp(-z))
	return s

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

# don't use learning_rate - this was a temporary placeholder for Course 1
# the better one is to use the deep network standard introduced in Course 2
# 	this is the He Initialization. For ReLu..use what is written below
def initialize_parameters(layer_dimensions):
	parameters = {}
	L = len(layer_dimensions)
	for l in range(1, L):
		# parameters["W" + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l-1]) * learning_rate
		# He Initialization (or Xavier variant) - automatically scales your starting weights based on how many neurons 
		# 										  are in the previous layer so the network doesn't stall
		parameters["W" + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l-1]) * np.sqrt(2 / layer_dimensions[l-1])
		parameters["b" + str(l)] = np.zeros((layer_dimensions[l], 1))
	return parameters
		
def create_mini_batches(X, Y):
	batches = []
	length = X.shape[1]
	ending_value = math.ceil(length/64)
	# must shuffle incase raw data is ordered in some way
	shuffled_indices = np.random.permutation(length)
	X_shuffled = X[:, shuffled_indices]
	Y_shuffled = Y[:, shuffled_indices]
	for i in range(1, ending_value + 1):
		mini_batch_X = X_shuffled[:, (i-1)*64:i*64]
		mini_batch_Y = Y_shuffled[:, (i-1)*64:i*64]
		batches.append((mini_batch_X, mini_batch_Y))
	return batches

def forward_propagation(X, layer_dims, parameters):
	# A. Forward propagation: calculate predictions for this batch
	cache = {}
	cache["A" + str(0)] = X
	for l in range(1, len(layer_dims)-1):
		cache["Z" + str(l)] = np.dot(parameters["W" + str(l)], cache["A" + str(l-1)]) + parameters["b" + str(l)]
		cache["A" + str(l)] = np.maximum(0, cache["Z" + str(l)])
	size = ((len(layer_dims))-1)
	cache["Z" + str(size)] = np.dot(parameters["W" + str(size)], cache["A" + str(size-1)]) + parameters["b" + str(size)]
	cache["A" + str(size)] = sigmoid(cache["Z" + str(size)])
	return cache, size

def predict(parameters, X, layer_dims):
	local_cache, size = forward_propagation(X, layer_dims, parameters)
	last_A_value = local_cache["A" + str(size)]
	predictions = (last_A_value > 0.5).astype(int)
	return predictions

def master_training_loop(X, Y, layer_dims, epochs, learning_rate, training_type):
	v = {}
	s = {}
	# initialize network parameters W and b for all layers
	parameters = initialize_parameters(layer_dims)
	beta = 0.9
	beta1 = 0.9
	beta2 = 0.9
	epsilon = 1e-8
	t = 0
	cost_metrics = []

	if training_type == 'momentum':
		v = initialize_momentum(parameters)
		beta = 0.9
	if training_type == 'rmsprop':
		s = initialize_rmsprop(parameters)
		beta = 0.999
	if training_type == 'adam':
		v, s = initialize_adam(parameters)

	# The grand optimization loop
	for i in range(1, epochs + 1):
		# Generate a fresh set of scrambled mini-batches for this epoch
		batches = create_mini_batches(X, Y)
		number_of_batches = len(batches)
		cost_total = 0

		# Loop through every tiny chunk of customers (ever mini batch)
		for X_mini_batch, Y_mini_batch in batches:
			t += 1

			# A. Forward propagation: calculate predictions for this batch
			cache, size = forward_propagation(X_mini_batch, layer_dims, parameters)

			# B. Compute cost: see how off the predictions were for this batch
			# Adding 1e-15 is a trick to prevent the code from crashing with NaN if your model accidentally 
			# predicts an exact 0.0 or 1.0
			inner = (Y_mini_batch * np.log(cache["A" + str(size)] + 1e-15)) + (1 - Y_mini_batch)*(np.log(1-cache["A" + str(size)] + 1e-15))
			cost = (-1) * (1/X_mini_batch.shape[1]) * np.sum(inner)
			cost_total += cost

			# C. Backward propagation: Calculate the immediate raw gradients (dW, db)
			grads = {}
			grads["dZ" + str(size)] = (cache["A" + str(size)] - Y_mini_batch)
			for l in range(len(layer_dims) - 1, 0, -1):
				grads["dW" + str(l)] = (1/X_mini_batch.shape[1]) * np.dot(grads["dZ" + str(l)], cache["A" + str(l-1)].T)
				grads["db" + str(l)] = (1/X_mini_batch.shape[1]) * np.sum(grads["dZ" + str(l)], axis = 1, keepdims=True)
				if l > 1:
					grads["dA" + str(l-1)] = np.dot(parameters["W" + str(l)].T, grads["dZ" + str(l)])
					relu_derivative_gate = (cache["Z" + str(l-1)] > 0).astype(float)
					dZ_prev = grads["dA" + str(l-1)] * relu_derivative_gate
					grads["dZ" + str(l-1)] = dZ_prev

			for l in range(1, len(layer_dims)):
				if training_type == 'gd':
					parameters["W" + str(l)] = parameters["W" + str(l)] - np.dot(learning_rate, grads["dW" + str(l)])
					parameters["b" + str(l)] = parameters["b" + str(l)] - np.dot(learning_rate, grads["db" + str(l)])
				# Momentum tracks the velocity (v) of the gradients. It's like a heavy ball rolling down a hill,
				# using past momentum to smooth out bumps.
				elif training_type == 'momentum':
					v["dW" + str(l)] = (beta * v["dW" + str(l)]) + ((1-beta) * grads["dW" + str(l)])
					v["db" + str(l)] = (beta * v["db" + str(l)]) + ((1-beta) * grads["db" + str(l)])
					# D. Update parameters: feed the raw gradients and the caches into your optimizer
					parameters["W" + str(l)] = parameters["W" + str(l)] - np.dot(learning_rate, v["dW" + str(l)])
					parameters["b" + str(l)] = parameters["b" + str(l)] - np.dot(learning_rate, v["db" + str(l)])
				# RMSprop tracks the squared gradients (s). It acts like an automatic speed limiter, slowing down
				# adjustments in wildly volatile direcitons so the network doesn't overshoot.
				elif training_type == 'rmsprop':
					s["dW" + str(l)] = (beta2 * s["dW" + str(l)]) + (1 - beta2) * (grads["dW" + str(l)] ** 2)
					s["db" + str(l)] = (beta2 * s["db" + str(l)]) + (1 - beta2) * (grads["db" + str(l)] ** 2)
					# D. Update parameters: feed the raw gradients and the caches into your optimizer
					parameters["W" + str(l)] = parameters["W" + str(l)] - (np.dot(learning_rate, grads["dW" + str(l)])/(np.sqrt(s["dW" + str(l)]) + epsilon))
					parameters["b" + str(l)] = parameters["b" + str(l)] - (np.dot(learning_rate, grads["db" + str(l)])/(np.sqrt(s["db" + str(l)]) + epsilon))
				elif training_type == 'adam':
					v["dW" + str(l)] = (beta1 * v["dW" + str(l)]) + ((1-beta1) * grads["dW" + str(l)])
					v["db" + str(l)] = (beta1 * v["db" + str(l)]) + ((1-beta1) * grads["db" + str(l)])
					s["dW" + str(l)] = (beta2 * s["dW" + str(l)]) + (1 - beta2) * (grads["dW" + str(l)] ** 2)
					s["db" + str(l)] = (beta2 * s["db" + str(l)]) + (1 - beta2) * (grads["db" + str(l)] ** 2)
					v_dW_corrected = (v["dW" + str(l)]/(1-beta1 ** t))
					v_db_corrected = (v["db" + str(l)]/(1-beta1 ** t))
					s_dW_corrected = (s["dW" + str(l)]/(1-beta2 ** t))
					s_db_corrected = (s["db" + str(l)]/(1-beta2 ** t))
					# D. Update parameters: feed the raw gradients and the caches into your optimizer
					parameters["W" + str(l)] = parameters["W" + str(l)] - (np.dot(learning_rate, v_dW_corrected)/(np.sqrt(s_dW_corrected) + epsilon))
					parameters["b" + str(l)] = parameters["b" + str(l)] - (np.dot(learning_rate, v_db_corrected)/(np.sqrt(s_db_corrected) + epsilon))
		cost_metrics.append(cost_total/number_of_batches)

	return parameters, cost_metrics, cache


# hardcode a fixed random seed such that every optimizer starts with the 
# exact same weight matrices (W, b)
np.random.seed(42)

dataFrame = pd.read_csv('WA-Customer-Churn.csv')
dataFrame["TotalCharges"] = pd.to_numeric(dataFrame["TotalCharges"].str.strip(), errors='coerce').fillna(0).astype(float)
totalChargesMinVal = dataFrame["TotalCharges"].min()
totalChargesMaxVal = dataFrame["TotalCharges"].max()
tenureMinVal = dataFrame["tenure"].min()
tenureMaxVal = dataFrame["tenure"].max()
monthlyChargesMinVal = dataFrame["MonthlyCharges"].min()
monthlyChargesMaxVal = dataFrame["MonthlyCharges"].max()

dataFrame["TotalCharges"] = ((dataFrame["TotalCharges"] - totalChargesMinVal)/(totalChargesMaxVal - totalChargesMinVal))
dataFrame["tenure"] = ((dataFrame["tenure"] - tenureMinVal)/(tenureMaxVal - tenureMinVal))
dataFrame["MonthlyCharges"] = ((dataFrame["MonthlyCharges"] - monthlyChargesMinVal)/(monthlyChargesMaxVal - monthlyChargesMinVal))

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
X_df = dataFrame.T
X = X_df.to_numpy()

# [Input, Hidden1, Hidden2, Output]
layer_dims = [45, 20, 7, 1]
learning_rate = 0.01
epochs = 25
size = ((len(layer_dims))-1)

split_index = int(X.shape[1] * 0.8)
X_train = X[:, :split_index]
Y_train = Y[:, :split_index]

X_test = X[:, split_index:]
Y_test = Y[:, split_index:]

parameters_gd, cost_metrics_gd, cache_gd = master_training_loop(X_train, Y_train, layer_dims, epochs,learning_rate, 'gd')
parameters_momentum, cost_metrics_momentum, cache_momentum = master_training_loop(X_train, Y_train, layer_dims, epochs, learning_rate, 'momentum')
parameters_rmsprop, cost_metrics_rmsprop, cache_rmsprop = master_training_loop(X_train, Y_train, layer_dims, epochs, learning_rate, 'rmsprop')
parameters_adam, cost_metrics_adam, cache_adam = master_training_loop(X_train, Y_train, layer_dims, epochs, learning_rate, 'adam')

predictions_gd = predict(parameters_gd, X_test, layer_dims)
accuracy_gd = np.mean(predictions_gd == Y_test)

predictions_momentum = predict(parameters_momentum, X_test, layer_dims)
accuracy_momentum = np.mean(predictions_momentum == Y_test)

predictions_rmsprop = predict(parameters_rmsprop, X_test, layer_dims)
accuracy_rmsprop = np.mean(predictions_rmsprop == Y_test)

predictions_adam = predict(parameters_adam, X_test, layer_dims)
accuracy_adam = np.mean(predictions_adam == Y_test)

epochs_gd = np.arange(len(cost_metrics_gd))
epochs_momentum = np.arange(len(cost_metrics_momentum))
epochs_rmsprop = np.arange(len(cost_metrics_rmsprop))
epochs_adam = np.arange(len(cost_metrics_adam))

plt.plot(epochs_gd, cost_metrics_gd, label='Gradient Descent')
plt.plot(epochs_momentum, cost_metrics_momentum, label='Momentum')
plt.plot(epochs_rmsprop, cost_metrics_rmsprop, label='RMSProp')
plt.plot(epochs_adam, cost_metrics_adam, label='Adam')

print("================================================")
print("         CHURN ENGINE BENCHMARK RESULTS"         )
print("================================================")
print(f"Gradient Descent Accuracy: {accuracy_gd*100:.2f}%")
print(f"Momentum Accuracy: {accuracy_momentum*100:.2f}%")
print(f"RMSProp Accuracy: {accuracy_rmsprop*100:.2f}%")
print(f"Adam Accuracy: {accuracy_adam*100:.2f}%")
print("================================================")

plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.legend()
plt.show()

