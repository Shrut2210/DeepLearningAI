import numpy as np
from seaborn import load_dataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist
import warnings

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

(X_full, y_full), (_, _) = mnist.load_data()

filter_indices = np.where((y_full == 0) | (y_full == 1))
X_filtered = X_full[filter_indices]
y_filtered = y_full[filter_indices]

X_resized = np.array([np.resize(img, (20, 20)) for img in X_filtered])

X_unrolled = X_resized.reshape(X_resized.shape[0], -1)

X = X_unrolled[:1000]
y = y_filtered[:1000]

print(X.shape)  
print(y.shape)

warnings.simplefilter(action='ignore', category=FutureWarning)

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1)

for i,ax in enumerate(axes.flat):
    random_index = np.random.randint(m)
    
    X_random_reshaped = X[random_index].reshape((20,20)).T
    
    ax.imshow(X_random_reshaped, cmap='gray')
    
    ax.set_title(y[random_index])
    ax.set_axis_off()
    
    
model = Sequential(
    [               
        tf.keras.Input(shape=(400,)),   
        
        Dense(units=25, activation='sigmoid', name='layer_1'),        
        Dense(units=15, activation='sigmoid', name='layer_2'),
        Dense(units=1, activation='sigmoid', name='layer_3')

        
    ], name = "my_model" 
)       

model.summary()

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X,y,
    epochs=20
)

prediction = model.predict(X[0].reshape(1,400))  
print(f" predicting a zero: {prediction}")
prediction = model.predict(X[500].reshape(1,400))  
print(f" predicting a one:  {prediction}")

if prediction >= 0.5:
    yhat = 1
else:
    yhat = 0
print(f"prediction after threshold: {yhat}")


m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) 

for i,ax in enumerate(axes.flat):
    random_index = np.random.randint(m)
    
    X_random_reshaped = X[random_index].reshape((20,20)).T
    
    ax.imshow(X_random_reshaped, cmap='gray')
    
    prediction = model.predict(X[random_index].reshape(1,400))
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    
    ax.set_title(f"{y[random_index]},{yhat}")
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=16)
plt.show()