import numpy
import skimage.io, skimage.color, skimage.feature
import os

dataset_dir = "../data/Fruit360"
fruits = ["apple", "raspberry", "mango", "lemon"]
# Number of samples in the datset used = 492+490+490+490=1,962
# 360 is the length of the feature vector.
dataset_features = numpy.zeros(shape=(1962, 360))
outputs = numpy.zeros(shape=(1962))

idx = 0
class_label = 0
for fruit_dir in fruits:
    curr_dir = os.path.join(os.path.sep, fruit_dir)
    all_imgs = os.listdir(os.path.join(dataset_dir+curr_dir))
    for img_file in all_imgs:
        if img_file.endswith(".jpg"): # Ensures reading only JPG files.
            fruit_data = skimage.io.imread(fname=os.path.sep.join([dataset_dir, curr_dir, img_file]), as_gray=False)
            fruit_data_hsv = skimage.color.rgb2hsv(rgb=fruit_data)
            hist = numpy.histogram(a=fruit_data_hsv[:, :, 0], bins=360)
            dataset_features[idx, :] = hist[0]
            outputs[idx] = class_label
            idx = idx + 1
    class_label = class_label + 1

# Saving the extracted features and the outputs as NumPy files.
numpy.save("../data/dataset_features.npy", dataset_features)
numpy.save("../data/outputs.npy", outputs)
