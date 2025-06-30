import os  
import numpy as np 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import dpctl
from sklearnx import patch_sklearn, config_context
patch_sklearn()
from sklearn.model_selection import train_test_split  # Import train_test_split

# Initialize variables
is_init = False
size = -1

label = []
dictionary = {}
c = 0

# Load the data
for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
        if not(is_init):
            is_init = True 
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c += 1

# Convert labels to integers
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# Convert labels to categorical
y = to_categorical(y)

# Shuffle the dataset
X_new = X.copy()
y_new = y.copy()
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

# Rearrange the dataset based on shuffled indices
for counter, i in enumerate(cnt): 
    X_new[counter] = X[i]
    y_new[counter] = y[i]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# Define the model
input_shape = (X.shape[1],)  # Update the input shape to a tuple

ip = Input(shape=input_shape)

# Model architecture with Batch Normalization and Dropout
m = Dense(512, activation="relu")(ip)
m = BatchNormalization()(m)
m = Dropout(0.2)(m)  # Dropout for regularization
m = Dense(256, activation="relu")(m)

# Output layer
op = Dense(y.shape[1], activation="softmax")(m) 

# Create the model
model = Model(inputs=ip, outputs=op)

# Compile the model
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])

# Fit the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))  # Use validation data

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
