
"""
Created on Thu Jul  1 14:57:28 2020
binary classification with CNN
@author: parsa khorrami
"""

#libraries 
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.models import Sequential 
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras import backend as K 
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D



# for plotting images (optional)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# getting data
base_dir = 'data1'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_fruit = os.path.join(train_dir, 'fruit')
train_person = os.path.join(train_dir, 'person')
validation_fruit = os.path.join(validation_dir, 'fruit')
validation_person = os.path.join(validation_dir, 'person')

num_fruit_tr = len(os.listdir(train_fruit))
num_person_tr = len(os.listdir(train_person))
num_fruit_val = len(os.listdir(validation_fruit))
num_person_val = len(os.listdir(validation_person))

total_train = num_fruit_tr + num_person_tr
total_val = num_fruit_val + num_person_val

#initialising data
BATCH_SIZE = 32
IMG_SHAPE = 300 # square image
EPOCHS = 5 # initial EPOCHE if user don't inter its EPOCHS 
ACC=float(input ("accuracy(0 => base on initial EPOCHS ):")) # desired user EPOCHS, 0 = initial EPOCHS mean: EPOCHS=5.


#generators

#prevent memorization
train_image_generator = ImageDataGenerator(
    rescale=1./255,    
    )

validation_image_generator = ImageDataGenerator(
    rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')

val_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=validation_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')


#showing one sample of training images
xtbatches , ytbatches = next(train_data_gen)
for i in range (0,3):
    image1 = xtbatches[i]
    plt.imshow(image1)
    plt.show()

#showing one sample of validation images
xvbatches , yvbatches = next(val_data_gen )
for i in range (0,3):
    image1 = xvbatches[i]
    plt.imshow(image1)
    plt.show()

#callback
callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.001,
                                         patience=1, mode="min", baseline=ACC)

# model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)), # RGB
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5), # 1/2 of neurons will be turned off randomly
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(2, activation='softmax') #[0, 1] or [1, 0]  also you can use ""sigmoid"" function 

    ])



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# if we don't have progress, running code will stop
if  ACC == 0 :
  history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
    )
else: 
  history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE))),
    callbacks=[callback]
    )


# analysis

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = input("Please enter runed epochs :")
epochs_range = range(int(epochs))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



#Running the Model


img_path = 'C:/Users/parsa/..../Ptoject_files/input/fruit/fruit.jpg' # input your file path

img = load_img(img_path, target_size=(IMG_SHAPE, IMG_SHAPE))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)

# announce class of desired image
print(classes[0][0])
if classes[0][0]>0.5:
    print(" your image is a fruit ")
else:
    print("your image is a person ")


#Visualizing Intermediate Representations (optional)

successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
train_person_names = os.listdir(train_person)
train_fruit_names = os.listdir(train_fruit)
person_img_files = [os.path.join(train_person, f) for f in train_person_names]
fruit_img_files = [os.path.join(train_fruit, f) for f in train_fruit_names]
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
img_path = random.choice(person_img_files )
img = load_img(img_path, target_size=(IMG_SHAPE, IMG_SHAPE))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)
x /= 255 # Rescale by 1/255
# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)
# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers[1:]]
# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.figure()
plt.show()
    
