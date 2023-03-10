# -*- coding: utf-8 -*-
"""TO_CSV.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GTGLt652rol727qyIjvr1PhohHL1TYIF
"""

from google.colab import drive
drive.mount('/gdrive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import csv
import os

cols = [i for i in range(10305)]
file = open('FACE_DATA_BIG.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(cols)
persons = []
for i in range(40):
  s = f's{i+1}'
  persons.append(s)

for person in persons:
    # print(person)
    for filename in os.listdir(f'/gdrive/MyDrive/FACE_DATA/{person}'):
        # print(filename)
        pname = f'/gdrive/MyDrive/FACE_DATA/{person}/{filename}'
        img =  cv2.imread(pname,cv2.IMREAD_GRAYSCALE)
        image = list(cv2.resize(img, (112, 92)).reshape(10304))
        image.append(person)
        file = open('FACE_DATA_BIG.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(image)

image_df = pd.read_csv('FACE_DATA_BIG.csv')
image_df.head()

image_df.shape

shuffled_df = image_df.sample(frac = 1)

shuffled_df.to_csv('FACE_DATA_SHUFFLED_BIG.csv', index = False)

image_shuff = pd.read_csv('FACE_DATA_SHUFFLED_BIG.csv')
image_shuff.head()

