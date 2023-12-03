from utils.utils import plot_image, format_data, clear_anomalies


data = format_data()
images = data['images']

inds = [49, 53, 67, 55131, 55163, 55165, 55223, 55271, 55285, 55291, 55333]
for j in inds:
    test_image = images[j].reshape((28, 28))
    count = 0
    plot_image(test_image)
    for i in range(len(test_image)):
        clear_row = clear_anomalies(test_image[i])
        test_image[i] = clear_row
    plot_image(test_image)
