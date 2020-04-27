import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import keras
import keras.models as models
import keras.layers as layers
import pickle
import random
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.image as mpimg
from keras.preprocessing import image
IMG_size = 20

X = pickle.load(open('X.pickle','rb'))
Y = pickle.load(open('Y.pickle','rb'))

X = np.asarray(X)
X = X/255
Y= np.asarray(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

## Define the callback list
filepath = '/Users/keenan/PycharmProjects/personalProjects/venv/my_model_file.hdf5' # define where the model is saved
callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor = 'val_loss', # Use accuracy to monitor the model
        patience = 2 # Stop after one step with lower accuracy
    ),
    keras.callbacks.ModelCheckpoint(
        filepath = filepath, # file where the checkpoint is saved
        monitor = 'val_loss', # Don't overwrite the saved model unless val_loss is worse
        save_best_only = True # Only save model if it is the best
    )
]

nn = models.Sequential()
nn.add(layers.Conv2D(64, activation = 'relu', kernel_size = 3, input_shape = (IMG_size, 2*IMG_size, 3)))
nn.add(layers.MaxPooling2D(pool_size = (3,3)))
nn.add(layers.Dropout(rate = 0.5))
nn.add(layers.Conv2D(32,activation = 'relu', kernel_size = 2, input_shape = (IMG_size/3,2*IMG_size/3,1)))
nn.add(layers.MaxPooling2D(pool_size =(2,2)))
nn.add(layers.Conv2D(16, kernel_size= (2,2)))
nn.add(layers.Dropout(rate=0.5))
nn.add(layers.Flatten())
nn.add(layers.Dense(132, activation = 'relu', input_shape=(IMG_size/7*IMG_size*2/7, )))
nn.add(layers.Dense(10, activation = 'softmax'))
nn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = nn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)

nn.save_weights('/Users/keenan/PycharmProjects/personalProjects/venv/my_model_file.hdf5')
nn.load_weights('/Users/keenan/PycharmProjects/personalProjects/venv/my_model_file.hdf5')
nn.save('/Users/keenan/PycharmProjects/personalProjects/venv/shapes_cnn.h5')


img_path = '/Users/keenan/PycharmProjects/personalProjects/venv/tomato/Tomato_Septoria_leaf_spot/1a02cfc4-375a-4be1-82a6-fa7ec5117ced___Matt.S_CG 6049.JPG'
img = image.load_img(img_path, target_size=(28, 28))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

plt.imshow(img_tensor[0])
plt.show()
img_tensor = X[0]
img_tensor = np.expand_dims(img_tensor, axis=0)
layer_outputs = [layer.output for layer in nn.layers[:7]]
# Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=nn.inputs, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
# Returns a list of five Numpy arrays: one array per layer activation
#plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

layer_names = []
for layer in nn.layers[:12]:
    layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

images_per_row= 16

for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
    n_features = layer_activation.shape[-1]  # Number of features in the feature map
    size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):  # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,  # Displays the grid
            row * size: (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')


epochs =1
def plot_accuracy():
    train_accuracy = history.history['accuracy']
    test_accuracy = history.history['val_accuracy']
    x = list(range(1,epochs+1))
    plt.plot(x,train_accuracy,color = 'red')
    plt.plot(x,test_accuracy, color='blue')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('accuracy(red) and test_accuracy(blue) vs epochs')

plot_accuracy()
plt.savefig('/Users/keenan/PycharmProjects/personalProjects/venv/dataPlots/accuracyplot.png')
plt.show()


