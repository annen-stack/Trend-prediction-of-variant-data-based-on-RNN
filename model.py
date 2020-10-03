import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt

import pywt

class SeriesPredictor:

    def __init__(self, input_dim, seq_size, hidden_dim):
        # Hyperparameters
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        self.learning_rate = tf.Variable(0.01, trainable=False)

        # Weight variables and input placeholders


        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        # 5个小片段，有5个预测值
        self.y = tf.placeholder(tf.float32, [None, seq_size])



    def model(self, cell):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        outputs, states = tf.nn.dynamic_rnn(self.cell, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        # 返回长度为5的一维数组
        return out

    def train(self, train_x, train_y, test_x, test_y, rnnflag, gruflag, lstmflag,dropoutflag,learnflag,crossflag):

        if rnnflag:
            print('rnn')
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        if gruflag:
            print('gru')
            self.cell = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
        if lstmflag:
            print('lstm')
            self.cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_dim)

        if dropoutflag==True:
            self.cell = rnn.DropoutWrapper(self.cell, output_keep_prob=0.6)  # 4.dropout 防止过拟合
        else:
            self.cell = rnn.DropoutWrapper(self.cell, output_keep_prob=1)


        if crossflag:  #多层RNN？？？
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model(self.cell),labels=self.y))
        else:
            cost = tf.reduce_mean(tf.square(self.model(self.cell) - self.y))


        if learnflag:
            train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost) #自适应学习率变化
        # Auxiliary ops
        else:
            train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)#随机梯度下降

        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE)


         #   tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE)
            n=0
            test_errsum=[]
            train_errsum=[]
            for step in range(1000):  ##1.迭代次数，不是越大越好，需要找到最优值

                _, train_err = sess.run([train_op, cost], feed_dict={self.x: train_x, self.y: train_y})
                if step % 10 == 0:
                    test_err = sess.run(cost, feed_dict={self.x: test_x, self.y: test_y})

                    print('step:{}\t\ttrain err：{}\t\ttest err：{}'.format(step, train_err, test_err))
                    test_errsum.append(test_err)
                    train_errsum.append(train_err)
                    n=n+1
            print(n)
            save_path = self.saver.save(sess, './model/')
            print('Model saved to {}'.format(save_path))

        plt.figure()
        plt.plot(list(range(len(test_errsum))), test_errsum, color='r', label='test_err')
            # 真实值
        plt.plot(list(range(len(train_errsum))), train_errsum, color='g', label='train_err')
            # 加图例
        plt.legend()
        plt.xlabel('epoch(x10)')
        plt.ylabel('error_rate')

        plt.ylim(0,1)


        if 'train_test_err' is not None:
            plt.savefig('train_test_err')
        else:
            plt.show()

    def test(self, sess, test_x):
        self.saver = tf.train.Saver()

        tf.variable_scope(name_or_scope = '', reuse = tf.AUTO_REUSE)

        self.saver.restore(sess, './model/')
        output = sess.run(self.model(self.cell), feed_dict={self.x: test_x})
        return output
