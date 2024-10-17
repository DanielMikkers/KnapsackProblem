from Classes import GenData, KnapsackSolver, NeuralNetworkDP
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import time

np.set_printoptions(edgeitems=10,linewidth=180)

# inititiate the values for creating data
n = 100
W = 1000
train_data_num = 3750
test_data_num = 100
normalizer = 4*W

#creating the data
data = GenData(n, W, correlation= 'strong', numObservations=train_data_num)
vw_arr = data.generate()
x_train = np.zeros_like(vw_arr).astype(float)

#normlizing the value and weight vectors
x_train[:,:,0] = vw_arr[:,:,0]/np.max(vw_arr[:,:,0], axis=1)[:,np.newaxis]
x_train[:,:,1] = vw_arr[:,:,1]/np.max(vw_arr[:,:,1], axis=1)[:,np.newaxis]

#creating training tables
y_train = KnapsackSolver().knapsack_DP_Ntable(n, W, vw_arr)

#reshaping tables to vectors
y_train_reshape = np.reshape(y_train, (train_data_num, (n+1)*(W+1)))
y_train_reshape = np.reshape(y_train, (train_data_num, (n+1)*(W+1))).astype(float)

#normalizing DP tables 
y_train_norm = y_train_reshape/np.max(y_train_reshape, axis=1)[:,np.newaxis]

#taking the maximum value of each table which has to be multiplied with the array
#to recreate the table
maxx = np.max(y_train_reshape, axis=1)
maxx = maxx.reshape(1,maxx.shape[0])

#normilizing the maxx values with some outside chosen constant to make sure
#that it doesnt have a problem with large values
maxx_norm = maxx/normalizer

#adding the max normalized values to the training array
y_train_new = np.concatenate((y_train_norm, maxx_norm.T), axis=1)

#model architecture
model = models.Sequential()

#flatten the input (n, 2) to a single vector
model.add(layers.Flatten(input_shape=(n, 2)))
model.add(layers.Dense(64, activation=tf.keras.layers.LeakyReLU(negative_slope=0.1)))
model.add(layers.Dense(128, activation=tf.keras.layers.LeakyReLU(negative_slope=0.1)))
model.add(layers.Dense(256, activation=tf.keras.layers.LeakyReLU(negative_slope=0.1)))
model.add(layers.Dense((n + 1) * (W + 1) +1))
model.compile(optimizer='adam', loss='mse')

#model summary to visualize the architecture
model.summary()

#fit model
history = model.fit(x_train, y_train_new, batch_size=32, epochs=30, validation_split=0.2)

#generate test data
data_test = GenData(n, W, correlation= 'strong', numObservations=test_data_num)
x_test = data_test.generate()
x_test = x_test.astype(int)
x_test_norm = np.zeros_like(x_test).astype(float)
x_test_norm[:,:,0] = x_test[:,:,0]/np.max(x_test[:,:,0], axis=1)[:,np.newaxis]
x_test_norm[:,:,1] = x_test[:,:,1]/np.max(x_test[:,:,1], axis=1)[:,np.newaxis]
y_test = KnapsackSolver().knapsack_DP_Ntable(n, W, x_test).astype(int)

#predicting
pred = model.predict(x_test_norm)
pred_table = pred[:,:-1]
pred_val = pred[:,-1]*normalizer
pred_table_pred = pred_table*pred_val[:,np.newaxis]
pred_table_pred = pred_table_pred.reshape(test_data_num,n+1,W+1)
pred_table_pred = pred_table_pred.astype(int)

#mean absolute error on whole table
print(np.mean(np.absolute(y_test-pred_table_pred)))

#mean absolute error on last value
print(np.mean(np.absolute(y_test[:,-1,-1]-pred_table_pred[:,-1,-1])))

#execution time of the predictions of the model
n_pred = np.arange(10,1001,30).astype(int)
exec_times = np.zeros_like(n_pred)

for i in range(np.size(n_pred)):
    data_test = GenData(n, W, correlation= 'strong', numObservations=n_pred[i])
    x_test = data_test.generate()
    x_test = x_test.astype(int)
    x_test_norm_pred = np.zeros_like(x_test).astype(float)
    x_test_norm_pred[:,:,0] = x_test[:,:,0]/np.max(x_test[:,:,0], axis=1)[:,np.newaxis]
    x_test_norm_pred[:,:,1] = x_test[:,:,1]/np.max(x_test[:,:,1], axis=1)[:,np.newaxis]

    
    start = time.time()
    pred = model.predict(x_test_norm_pred)
    pred_table = pred[:,:-1]
    pred_val = pred[:,-1]*normalizer
    pred_table_pred = pred_table*pred_val[:,np.newaxis]
    pred_table_pred = pred_table_pred.reshape(n_pred[i],n+1,W+1)
    pred_table_pred = pred_table_pred.astype(int)
    end = time.time()
    exec_times[i] = end - start

#plot loss
plt.figure(dpi=300)
plt.plot(history.history['loss'], label='loss', color='g')
plt.plot(history.history['val_loss'], label = 'validation loss', color='r')
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Epoch-1')
plt.grid(True, which="both", ls="-", color='0.65')
plt.xticks(np.arange(0, 30, 5))
plt.legend()
plt.savefig('loss.png')
plt.show()

#plot execution time predictions
plt.figure(dpi=300)
plt.plot(n_pred, exec_times, color='b')
plt.xscale('log')
plt.ylabel('Execution time [s]')
plt.xlabel('Number of knapsack samples of 100 objects and capacity 1000')
plt.grid(True, which="both", ls="-", color='0.65')
plt.savefig('exec_time_pred.png')
plt.show()