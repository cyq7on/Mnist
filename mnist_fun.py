import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def get_accuracy(mnist=None, learning_rate=0.01, batch_size=100, total_steps=5000):
    # 读取mnist数据集，如果本地不存在则会自动下载
    if mnist is None:
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # 创建两个占位符，x为输入的图像像素，y为图像类别0~9
    x = tf.placeholder("float", [None, 784])
    y_ = tf.placeholder("float", [None, 10])
    # 定义变量，维度与图像像素和类别对应
    # w = tf.Variable(tf.zeros([784, 10]))
    # b = tf.Variable(tf.zeros([10]))

    w = tf.Variable(tf.truncated_normal([784, 10]))
    b = tf.Variable(tf.truncated_normal([10]))
    # softmax回归
    y = tf.nn.softmax(tf.matmul(x, w) + b)
    # 计算交叉熵
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    # 梯度下降得到步长
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    # 初始化并启动模型
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    for i in range(total_steps + 1):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    result = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    sess.close()
    return result
