import numpy as np
from pyod.models.copod import COPOD

from utils.utils import format_data, plot_image


cols = 2
rows = 784 // cols
ratio = 2.65

inds = [49, 53, 55131, 55163, 55165, 55223, 55271, 55285, 55291, 55333]
searcher = COPOD()
data = format_data()
images = np.array([d.reshape((rows, cols)) for d in (data['images'])])

for k in inds:
    test_image = images[k].reshape((rows, cols))
    data_anomalies = searcher.decision_function(test_image)
    plot_image(test_image)

    for i in range(rows):
        if data_anomalies[i] < ratio:
            test_image[i] = 0.
    plot_image(test_image)
