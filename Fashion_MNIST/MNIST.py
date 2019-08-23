from __future__ import absolute_import, division, print_function, unicode_literals

#导入TensorFlow和tf.keras
import tensorflow as tf
from  tensorflow import keras

#导入辅助库
import numpy as np
import matplotlib.pyplot as plt

import os
import gzip

from DrawFunction import *

print(tf.__version__)

#载入数据集 远程下载数据集
#fashion_mnist = keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#加载本地数据集
def load_data(data_folder):
    files = [
        'train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder,fname))

    #解析训练标签
    with gzip.open(paths[0], 'rb') as lbpath:
        train_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    #解析训练图像
    with gzip.open(paths[1], 'rb') as imgpath:
        train_iamges = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(train_labels), 28, 28)

    #解析测试标签
    with gzip.open(paths[2], 'rb') as lbpath:
        test_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    #解析测试图像
    with gzip.open(paths[3], 'rb') as imgpath:
        test_iamges = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(test_labels), 28, 28)

    return (train_iamges, train_labels), (test_iamges, test_labels)

"""
def plot_iamge(i, predictions_array,true_label,img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{},{:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                        color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
"""


(train_images, train_labels), (test_images, test_labels) = load_data('data/fashion/')

#标注标签
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

#查看数据
print(train_images.shape)

print(len(train_labels))

print(train_labels)

print(test_images.shape)

print(len(test_labels))

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#进网络之前进行归一化
train_images = train_images / 255.0
test_images  = test_images / 255.0

#显示训练集中前25个图像，验证数据是否正确
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation=tf.nn.relu),
        keras.layers.Dense(10,activation=tf.nn.softmax)
    ]
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images,train_labels,epochs=5)

test_loss, test_acc = model.evaluate(test_images,test_labels)

print('Test accuracy: ', test_acc)

predictions = model.predict(test_images)

print(predictions[0])

print(np.argmax(predictions[0]))

i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1,2,1)
plot_iamge(i,predictions,test_labels,test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_iamge(i, predictions,test_labels,test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,test_labels)
plt.show()

rows = 5
cols = 3
images = rows * cols

plt.figure(figsize=(2*2*cols, 2*rows))
for i in range(images):
    plt.subplot(rows, 2 * cols, 2 * i + 1)
    plot_iamge(i, predictions, test_labels, test_images)
    plt.subplot(rows, 2 * cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()

#从测试集中获取数据
img = test_images[0]

print(img.shape)

#将图像添加到批次中，即使它是唯一的成员
img = np.expand_dims(img, 0)
print(img.shape)

predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10),class_names,rotation=45)
plt.show()

