from keras.layers import Input, Lambda, Dense Flatten
from keras.models import model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocessed inputs
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metric import confusion_matrix
import numpy as np
import matplotlib.pyplot as plot
from glob import glob

IMAGE_SIZE = [100,100]

epochs = 5
batch_size = 16

train_path = ''
test_path = ''

image_files = glob(train_path + '/*/*/jp*g')
test_image_files = glob(test_path + '/*/*/jp*g')

folders = glob(train_path + '/*')

plt.imshow(image.img_to_array(image.load_img(np.random.choise(image_files))))
plt.show()

vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation = 'softmax')(x)

model = Model(inputs=vgg.input, outputs = prediction)
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metric = ['accuracy'])

gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input
)

train_gen = gen.flow_from_directory(train_path, target_size = IMAGE_SIZE, shuffle = True, batch_size = batch_size)
test_gen = gen.flow_from_directory(test_path, target_size = IMAGE_SIZE, shuffle = True, batch_size = batch_size)

r = model.fit_generator(train_gen, validation_data = test_gen, epochs = epochs, steps_per_epoch = len(image_files) // batch_size, validation_steps = len(test_image_files)// batch_size)

plt.plot(r.history['loss'], label = 'train loss')
plt.plot(r.history['val_loss'], label = 'validation loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label = 'train_acc')
plt.plot(r.history['val_accuracy'], label = 'test accuracy')
plt.legend()
plt.show()

print("and they all lived happily ever after")
