

import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2

#=====================#
# load images
#=====================#

DIR = 'Images'
CATEGORIES = ['Dog', 'Cat']

for category in CATEGORIES:
    path = os.path.join(DIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break


#===================#
# normalize
#==================#

IMG_SIZE = 50 

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()


#=================#
# preprocessing
#=================#

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                plt.imshow(img_array, cmap='gray')
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e: 
                pass
            

create_training_data()

print(len(training_data))

# balancing
import random 
random.shuffle(training_data)

for sample in training_data[0:10]:
    print(sample[0], sample[1])

# build features and labels
features = []
label = []

for x, y in training_data:
    features.append(x)
    label.append(y)

# transform in numpy
features = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# save data as pickle for future usage
import pickle 

picout = open('features.pickle', 'wb')
pickle.dump(features, picout)
picout.close()

picout = open('label.pickle', 'wb')
pickle.dump(label, picout)
picout.close()

pickle_in = open('features.pickle', 'rb')
features = pickle.load(pickle_in)