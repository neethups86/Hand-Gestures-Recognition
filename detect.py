# -*- coding: utf-8 -*-
"""Detect.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BK3Iaap487PjUh1zweXFKv23vZEkMuOe
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
from google.colab.patches import cv2_imshow

#CATEGORIES = ["1finger", "2fingers", "3fingers", "4fingers", "5fingers", "handsopen", "nofinger", "thumbsup", "yoyo"]

from IPython.display import Image, display

filepath = '/content/drive/My Drive/Dataset/Dataset/0/IMG_1118.JPG'
display(Image(filepath, width=255, height=255))

def prepare(filepath):
  print(filepath)
  IMG_SIZE = 255
  img_array = cv2.imread(filepath)
  #cv2_imshow(filepath)
  #cv2_imshow(filepath)
  new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), 3)
  return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
from google.colab.patches import cv2_imshow
from keras.preprocessing import image

classes = ["1finger", "2fingers", "3fingers", "4fingers", "5fingers", "handsopen", "nofinger", "8", "9"]
#classes = ["1finger", "2fingers", "3fingers", "4fingers", "5fingers", "handsopen"]
from IPython.display import Image, display
img = '/content/drive/My Drive/Dataset/Dataset/0/IMG_1118.JPG'
display(Image(img, width=224, height=224))

#img = image.load_img('/content/drive/My Drive/tensorflow-for-poets-2-master/tf_files/imp/cylinder/110_04-_1_rep_CORE_GEN_36626158_1_1.jpg',target_size=(224,224,3))
#img = image.img_to_array(img)
#img = img/255
#classes = np.array(train.columns[2:])
model = tf.keras.models.load_model('/content/drive/My Drive/detecthandee.h5')
prediction = model.predict([prepare(img)])
top_3 = prediction[0][0] * 100
top_4 = prediction[0][1] * 100
top_5 = prediction[0][2] * 100
top_6 = prediction[0][3] * 100
top_7 = prediction[0][4] * 100
top_8 = prediction[0][5] * 100
top_9 = prediction[0][6] * 100
top_10 = prediction[0][7] * 100
top_11 = prediction[0][8] * 100
top_12 = prediction[0][9] * 100
print("1finger -",top_3, "%")
print("2fingers",top_4, "%")
print("3fingers",top_5, "%")
print("4fingers",top_6, "%")
print("5fingers",top_7, "%")
print("6fingers",top_8, "%")
print("7",top_9, "%")
print("8",top_10, "%")
print("9",top_11, "%")
print("10",top_12, "%")