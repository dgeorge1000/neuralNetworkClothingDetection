'''
To see how the code works, you need to have a project interpreter (file->settings->project->project interpreter). One possibility would be
anaconda. You also need to add the python file to the configuration which is found on the top right side of the screen in PyCharm. The code
might take a little time to run, so even if there are red warning lines let the code run. In the end there will be a file from matplotlib
that will show the input data, actual answer, and predicted answer. Everytime you close that window a new image will appear as performed in
this code.
'''


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# fashion dataset
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# to get the labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# to get the greyscale value to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    # flatten the data so instead of passing 28 arrays of 28 pixels, we compress it to 728
    keras.layers.Flatten(input_shape=(28,28)),
    # 128 neurons in the hidden layer, the activation function is rectify linear unit
    keras.layers.Dense(128, activation="relu"),
    # 10 output layers, 0->1 for each given class of clothes
    keras.layers.Dense(10, activation="softmax")
])

# loss function to know how the model is doing in terms of accuracy
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# epochs is how many times the model sees the information, randomly picks from the data but in a different order
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

# to do it on one image do model.predict( [test_image[7]] )
prediction = model.predict(test_images)

# test the model to see the prediction and the actual
for i in range(20):
    # creates a basic grid
    plt.grid(False)
    # show the image with greyscale
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    # show the label of the image
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    # gets the largest value from the input value, and prints the label that the model predicts
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    # show the images
    plt.show()

    # to see several in a row
    '''
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    plt.show()
    '''


