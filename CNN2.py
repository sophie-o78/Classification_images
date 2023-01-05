import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, MaxPooling2D, Conv2D
from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
import numpy as np


#on charge X et y
X = np.load('MNIST_X_28x28.npy')
y = np.load('MNIST_y.npy')



#on divise les echantillons entre le set d'entreinement et le set de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#on redimensionne l'image pour l'operation de convolution
x_train= np.array(X_train).reshape((X_train.shape[0],28,28,1))#1 couleur
x_test= np.array(X_test).reshape((X_test.shape[0],28,28,1))

#set de validation
x_val = x_train[46000:56000]
y_val = y_train[46000:56000]
x_train = x_train[:46000]
y_train = y_train[:46000]



y_train = to_categorical(y_train, num_classes=10) #remet en array
y_test = to_categorical(y_test, num_classes=10)
##
print (len(y_train))
print (len(x_train))
x_train = x_train /255.0 #on normalise en divisant par la valeur maximale (ici intensite lumineuse de 255, qui correspond qu blanc)
x_test = x_test /255.0

#plt.imshow(x_train[0])
#plt.show()

model = Sequential()
#Bloc convolutionnel
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=x_train.shape[1:])) #64 noyaux, filtres de 3x3
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))



#connected layer 1
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
##

#derniere couche connectee
model.add(Dense(10)) # 10 = nombre de classes en sortie
model.add(Activation("softmax")) #softmax pour probabilite d'etre dans la classe x

model.compile(optimizer='adam',
              loss= keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, validation_data= (x_val, y_val))

print("Entraînement fini")
#model.summary()

print (np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()

print (np.argmax(predictions[128]))
plt.imshow(x_test[128])



