import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random
import tensorflow as tf
import keras.utils
DATADIR = '/Users/keenan/PycharmProjects/personalProjects/venv/tomato'
CATEGORIES = ['Tomato_Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_yellowLeaf__Curl_Virus',
              'Tomato_Bacterial_spot', 'Tomato_Early_blight','Tomato_healthy',
              'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
              'Tomato_Spider_mites_Two_spotted_spider_mite']

IMG_size = 50

doubleData_training_data = []
RGB_training_data = []
HSV_training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:

                img_array = cv2.imread(os.path.join(path, img))

                new_array = cv2.resize(img_array, (IMG_size, IMG_size))
                hsv = cv2.cvtColor(new_array, cv2.COLOR_BGR2HSV)

                HSV_training_data.append(hsv)
                RGB_training_data.append(new_array)

                doubleData = np.append(new_array,hsv,axis = 1)


                doubleData_training_data.append([doubleData, class_num])
                #training_data.append([hsv, class_num])

            except:
                Exception


    random.shuffle(doubleData_training_data)


create_training_data()


X = []
Y = []
for features, label in doubleData_training_data:
    X.append(features)
    Y.append(tf.keras.utils.to_categorical(label,10))

X = np.array(X).reshape(-1,IMG_size,(2*IMG_size),3)



import pickle

pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('Y.pickle', 'wb')
pickle.dump(Y, pickle_out)
pickle_out.close()

