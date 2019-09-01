import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)

imdb = keras.datasets.imdb

#在线下载数据
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Train entries: {}, labels: {}".format(len(train_data), len(train_labels)))

print(train_data[0])

#将单词映射到整数索引的字典
word_index = imdb.get_word_index()

#第一个索引被保留
word_index = {k:(v+3) for k, v in word_index.items()}

word_index["<PAD>"]    = 0
word_index["<START>"]  = 1
word_index["<UNK>"]    = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'?') for i in text])

decode_review(train_data[0])

#标准化影评的数据长度
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256)

vocab_size = 10000

#构建网络模型
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation=tf.nn.relu))
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))

model.summary()

#损失函数和优化器
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

x_value = train_data[:10000]
partial_x_train = train_data[10000:]

y_value = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_value,y_value),
    verbose=1)

result = model.evaluate(test_data,test_labels)

print(result)


history_dict = history.history
history_dict.keys()

#dict_keys(['loss', 'val_loss', 'val_acc', 'acc'])

acc      = history.history['acc']
val_loss = history.history['val_loss']
val_acc  = history.history['val_acc']
loss     = history.history['loss']

epochs   = range(1, len(acc) + 1)

#bo is for blue dot
plt.plot(epochs, loss, 'bo', label='Training loss')
#b is for solid blue line
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.figure()
acc_value = history_dict['acc']
val_acc_value = history_dict['val_acc']
plt.plot(epochs, acc_value, 'bo', label='Training acc')
plt.plot(epochs, val_acc_value, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()

plt.show()