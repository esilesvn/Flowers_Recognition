import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib import style
import seaborn as sns
#from PIL import Image
import cv2

#ENCODING THE LABELS
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

#SPLITTING THE DATA
from sklearn.model_selection import train_test_split

#BUILDING THE NEURAL NETWORK
#from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras import layers
from keras import models
#from keras import optimizers
#from keras.layers import Dropout, Flatten,Activation
#from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense
#from keras.optimizers import SGD
#import tensorflow as tf


#Initialization of working directory
current = os.getcwd()
flowers_path = '/flowers'
flower_types = os.listdir(current + flowers_path) #returns the list of species based on the sub repo

### DATA EXPLORATION : 
def generate_df():
    flowers = []

    for species in flower_types:
        # Get all the file names
        all_flowers = os.listdir(current + flowers_path + '/' + species)
        # Add them to the list
        for flower in all_flowers:
            flowers.append((species, str(current + flowers_path + '/' + species) + '/' + flower))
    
    # Build a dataframe        
    flowers = pd.DataFrame(data=flowers, columns=['category', 'image'], index=None)
    return flowers

def check_count(flowers):
    flowers.category.value_counts()
    sns.countplot(flowers.category)

### DATA PREPARATION 
    
def image_array_transf(flowers):
    #Transforming the images into a single array
    dim = (150,150)
    images_array = np.array([np.array(cv2.resize(cv2.imread(img,cv2.IMREAD_COLOR),dim)) for img in flowers.image])
    
    return images_array

def encoding_labels(flowers):
    #labels_array = np.array([label for label in flowers.labels])
    labels_ls = [label for label in flowers.category]
    Encod = LabelEncoder()
    labels = Encod.fit_transform(labels_ls)
    labels_array = to_categorical(labels, 5)

    return labels_array
    

### SPLITTING THE DATA

def data_split(images_array, labels_array):
    X_train, X_test, y_train, y_test = train_test_split(images_array,
                                                        labels_array,
                                                        test_size=0.33,random_state=42)
    X_train = X_train /255
    X_test = X_test /255
    
    return X_train, X_test, y_train, y_test


### CONVOLUTIONAL NEURAL NETWORK
def CNN_model():
    model = models.Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
     
    
    model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(5, activation = "softmax"))
    return model

def compile_model(model):
    #initialization of the optimizer, loss function and metrics
    model.compile(optimizer='Adam' ,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
    
def model_fit(X_train, y_train, model):   
    #fitting the model
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.3, shuffle = True)
    return history

def display_accuracy(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    
def history_metrics(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(current + f'/Accuracy of the model.png')
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(current + f'/Loss of the model.png')
    
### PREDICTIONS  : 
def predict(model, X_test):
    probability_model = models.Sequential([model, 
                                           layers.Softmax()])
    predictions = probability_model.predict(X_test)
    return predictions



if __name__ == '__main__' :
    
    #data exploration
    flowers = generate_df()
    check_count(flowers)
    
    #data preparation
    images_array = image_array_transf(flowers)
    labels_array = encoding_labels(flowers)
    X_train, X_test, y_train, y_test = data_split(images_array, labels_array)
    
    #model
    model = CNN_model()
    compile_model(model)
    history = model_fit(X_train, y_train, model)
    display_accuracy(model, X_test, y_test)
    history_metrics(history)
    
    #predictions
    predictions = predict(model, X_test)
    
    
    
    
    
    
