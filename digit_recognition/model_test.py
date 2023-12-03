import numpy as np
from random import randint
from keras.models import load_model
from utils.utils import format_data, plot_image


model = load_model('../model')
data = format_data()
images = data['images']

random_index = randint(55001, 60000)
test_image = images[random_index].reshape((1, -1))
plot_image(test_image)
predicted_results = model.predict(test_image)
result = np.argmax(predicted_results)
print(f'НА КАРТИНКЕ ЦИФРА {result}')
