from tensorflow import keras
from sklearn.model_selection import train_test_split
from utils.utils import format_data


image_size = 28*28

data = format_data(image_size=image_size)
images = data['images']
np_labels_onehot = data['np_labels_onehot']

x_train, x_test, y_train, y_test = train_test_split(images, np_labels_onehot, train_size=55000)

model = keras.Sequential()
model.add(keras.layers.Dense(input_shape=(image_size,), units=128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=128)
model.evaluate(x_test, y_test)
model.save('../model')
