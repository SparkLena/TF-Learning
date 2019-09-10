from keras.datasets import mnist
import matplotlib.pyplot as plt
#导入TensorFlow和tf.keras
import tensorflow as tf
import numpy as np

 # 构建模型
from keras.models import Sequential
from  keras.layers import Convolution2D, MaxPooling2D, Activation, Flatten, Dense
from keras.optimizers import Adam

print(tf.__version__)

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
x_train = x_train.reshape(-1, 28, 28) / 255
x_valid = x_valid.reshape(-1, 28, 28) / 255
# 可视化数据
plt.figure(figsize=(12, 8))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.title("label:{}".format(y_train[i]))
    plt.imshow(x_train[i], cmap='gray')
plt.show()

model = Sequential()

model.add(Convolution2D(
    batch_input_shape=(None, 28, 28, 1),  # 输入数据维度
    filters=32,  # 卷积核数目
    kernel_size=3,  # 卷积核大小
    strides=1,  # 步长
    padding='same',  # (3-1)/2
    data_format='channels_last'  # 通道位置，注意keras和torch不同，一般通道在最后
))  # 加入一个卷积层，输出(28, 28, 32)
model.add(Activation('relu'))  # 加入激活函数
model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_last', ))  # 输出(14, 14, 32)

model.add(Convolution2D(64, 3, strides=1, padding='same', data_format='channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_last'))  # 输出(8, 8, 64)

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))  # (1024)

model.add(Dense(10))
model.add(Activation('softmax'))  # (10) 这里是概率

model.compile(optimizer=Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 训练模型
history = model.fit(x_train.reshape(-1, 28, 28, 1), y_train, batch_size=64, epochs=10, validation_split=0.2, shuffle=True, verbose=True)

loss, accuracy = model.evaluate(x_valid.reshape(-1, 28, 28, 1), y_valid)
print(loss, accuracy)

result = model.predict(x_valid[:10].reshape(-1, 28, 28, 1))
plt.figure(figsize=(12, 8))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_valid[i], cmap='gray')
    plt.title("true:{}pred:{}".format(np.argmax(y_valid[i], axis=0), np.argmax(result[i], axis=0)))
plt.show()

