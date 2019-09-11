import numpy as np
import tensorflow as tf
import xlrd
import matplotlib.pyplot as plt
import os
from sklearn.utils import check_random_state

#人为构造数据
n = 50
"""
np.arange([start, ]stop, [step, ]dtype=None)
start:可忽略不写，默认从0开始;起始值
stop:结束值；生成的元素不包括结束值
step:可忽略不写，默认步长为1；步长
dtype:默认为None，设置显示元素的数据类型
"""
XX = np.arange(n)

rs = check_random_state(0)

YY = rs.randint(-20, 20, size=(n, )) + 2.0 * XX

print(YY)
data = np.stack([XX, YY], axis=1)

print(data)

###################################
####定义标志位
###################################
#tf.app.flags.DEFINE_integer('num_epochs', 50, 'The number of epochs for training the model. Default=50')
#用于接收int型数值的变量
#“DEFINE_xxx”函数带3个参数，分别是变量名称，默认值，用法描述，例如：
tf.app.flags.DEFINE_integer('num_epochs', 50, 'The number of epochs for training the model. Default=50')
# 在FLAG structure中存储所有元素
FLAGS = tf.app.flags.FLAGS

#创建权重和偏置
#默认值被设置为零
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")

#为输入X和标签y创建占位符
def inputs():
    """
    Defining the place_holders.
    :return:
            Returning the data and label place holders
    """
    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")
    return X,Y

#创建预测函数
def inference(X):
    """
    Forward passing the X
    :param X: Input
    :return: X*w + b
    """
    return X * w + b

#定义损失函数
def loss(X, Y):
    """
    compute the loss by comparing the predicted value to the actual label
    :param X: The input 
    :param Y: The label
    :return:  The loss over the samples
    """
    #启动预测
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted)) / (2 * data.shape[0])

#训练函数
def train(loss):
    """
    :param loss:
    :return:
    """
    learning_rate = 0.0001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    #初始化变量w和b
    sess.run(tf.global_variables_initializer())

    #获取输入张量
    X, Y = inputs()

    #返回训练损失并创建训练操作节点
    train_loss = loss(X, Y)
    train_op   = train(train_loss)


    #训练模型
    for epoch_num in range(FLAGS.num_epochs):
        loss_value, _ = sess.run([train_loss, train_op],
                                 feed_dict={X:data[:,0], Y:data[:,1]})

        #显示每一个epoch的损失
        print('epoch %d,  loss= %f'%(epoch_num + 1, loss_value))

        #保存权重和偏置
        wcoeff, bias = sess.run([w, b])


###############################
#### Evaluate and plot ########
###############################
Input_values = data[:,0]
Labels = data[:,1]
Prediction_values = data[:,0] * wcoeff + bias

# # uncomment if plotting is desired!
plt.figure()
plt.plot(Input_values, Labels, 'ro', label='main')
plt.plot(Input_values, Prediction_values, label='Predicted')

# # Saving the result.
plt.legend()
plt.savefig('plot.png')
plt.show()
plt.close()





