import tensorflow as tf
import numpy as np
import unittest

def create_training_set() :
    n_bedrooms = np.array([1,2,3,4,5,6], dtype=np.float32)
    price_in_hundreds_of_thousands = np.array([2, 2.5, 3, 3.5, 4, 4.5], dtype=np.float32)
    
    return n_bedrooms, price_in_hundreds_of_thousands

features, targets = create_training_set()

print(f"Features have shape: {features.shape}")
print(f"Targets have shape: {targets.shape}")

# unittest.test_create_training_data(create_training_set())

def define_and_compile_model() :
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(1, input_dim=1, activation="linear")
        ]
    )
    
    model.compile(optimizer="sgd", loss="mse")
    
    return model

untrained_model = define_and_compile_model()

untrained_model.summary()

def train_model() :
    
    n_bedrooms, price = create_training_set()
    model = define_and_compile_model()
    
    model.fit(n_bedrooms, price ,epochs=500)
    
    return model

trained_model = train_model()

new_n_bedrooms = np.array([7.0])
predicted_price = trained_model.predict(new_n_bedrooms, verbose=False).item()
print(f"Your model predicted a price of {predicted_price:.2f} hundreds of thousands of dollars for a {int(new_n_bedrooms.item())} bedrooms house")