#主要练习保存和恢复模型

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels  = test_labels[:1000]

#数据归一化
train_images = train_images[:1000].reshape(-1, 28*28) / 255.0
test_images  = test_images[:1000].reshape(-1, 28*28) / 255.0

# Returns a short sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu,input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])
    return model

model = create_model()
model.summary()

#设计检查点对中间训练模型权重进行保存,
#用于对训练过程的阶段性保存
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir  = os.path.dirname(checkpoint_path)

#创建检查点的回调函数
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    verbose=1)

model = create_model()
model.summary()

model.fit(
    train_images,
    train_labels,
    epochs=20,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback]) #训练中响应检查点回调函数

#创建一个未训练的新模型
#利用之前保存的模型架构训练的权重
#对测试集进行评估
#1.直接利用生成的模型进行评估
model = create_model()
loss, acc = model.evaluate(test_images,test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

#2. 加载训练好的模型进行评估
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#对每一个检查点创建一个唯一的名字
#并设置创建频率
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir  = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    verbose=1,
    save_weights_only=True,
    #Save weights every 5-epochs
    period=5)

model = create_model()
model.fit(
    train_images,
    train_labels,
    epochs=50,
    callbacks=[cp_callback],
    validation_data=(test_images, test_labels),
    verbose=0)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)