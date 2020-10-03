from datetime import datetime

import numpy as np
import tensorflow as tf
from pandas import read_csv
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import math

def wgn(x, snr):
    Ps = np.sum(abs(x)**2)/len(x)
    Pn = Ps/(10**((snr/10)))
    noise = np.random.randn(len(x)) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise



def plot_error(actual_vals, predicted_vals, train_x, filename):
    plt.figure()
    num_train = len(train_x)
    width = 0.35
    error_values = abs(actual_vals[:len(predicted_vals)] - predicted_vals)

    index = list(range(num_train, num_train + len(predicted_vals)))

    # 绘制柱状图, 每根柱子的颜色为紫罗兰色


# 把数据切分成80%训练集、20%测试集
def split_data(data, percent_train=0.60):
    num_rows = len(data)
    train_data, test_data = [], []
    for idx, row in enumerate(data):
        if idx < num_rows * percent_train:
            train_data.append(row)
        else:
            test_data.append(row)
    return train_data, test_data


class SeriesPredictor:

    def __init__(self, input_dim, seq_size, hidden_dim):
        # Hyperparameters
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim

        # Weight variables and input placeholders
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        # 5个小片段，有5个预测值
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        # Cost optimizer
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))

        self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost)
        # Auxiliary ops
        self.saver = tf.train.Saver()

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """

        def lstm_cell(hidden_size, keep_prob):
            cell = rnn.BasicRNNCell(hidden_size, reuse=tf.get_variable_scope().reuse)

            return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        def get_init_cell(batch_size, rnn_size):
            num_layers = 1
            keep_prob = 0.5#3 0.6 5 100
            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(rnn_size, keep_prob) for _ in range(num_layers)],
                                          state_is_tuple=True)
            init_state = cell.zero_state(batch_size, tf.float32)
            init_state = tf.identity(init_state, name='init_state')
            return cell, init_state

        cell,state=get_init_cell(seq_size,self.hidden_dim)

        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)

        num_examples = tf.shape(self.x)[0]

        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])

        out = tf.matmul(outputs, W_repeated) + self.b_out

        out = tf.squeeze(out)

        return out

    def train(self, train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            # test_err出现3次浮动停下
            tr=[]
            tes=[]
            for step in range(1200):
                _, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})

                if step % 10 == 0:
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
                    tes.append(test_err)
                    tr.append(train_err)
                    # print('step:{}\t\ttrain err：{}\t\ttest err：{}'.format(step, train_err, test_err))
                    print(str(train_err)+',')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            save_path = self.saver.save(sess, './model0/')
            print('Model saved to {}'.format(save_path))
            plt.figure()
            plt.plot(list(range(len(tr))), tr, color='r', label='训练集损失',linestyle='--')
            plt.plot(list(range(len(tes))), tes, color='b', label='测试集损失')
            plt.xlabel('时间')
            plt.ylabel('损失值')
            print(tr)
            plt.ylim(0,1)
            plt.title('损失图像 (模型:LSTM 随机失活率:1)')
            plt.legend()
            plt.show()

    def test(self, sess, test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, './model0/')
        output = sess.run(self.model(), feed_dict={self.x: test_x})
        return output


def plot_results(train_x, predictions, actual, filename):
    plt.figure()
    num_train = len(train_x)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    # 训练集。plt.plot(x,y,format_string,**kwargs) x轴数据，y轴数据
    plt.plot(list(range(num_train)), train_x, color='b', label='训练集')

    plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='测试集')
    # 预测集
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='预测值')
    plt.title('数据图像（模型：LSTM 随机失活率：0.5）')
    plt.xlabel('时间')
    plt.ylabel('数据值')
    # 真实值
#    plt.plot(list(range(num_train, num_train + len(predictionrnn))), predictionrnn, color='y', label='predicted_datarnn')
    # 加图例
    plt.legend()

    # error=abs(actual-prediction)

    # plt.plot(list(range(num_train, num_train + len(actual))), actual, color='b', label='test_data')

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')
import csv
if __name__ == '__main__':
    # 序列长度，基于一个小片段去预测下一个值
    seq_size = 5
    predictor = SeriesPredictor(input_dim=1, seq_size=5, hidden_dim=120)

    series = read_csv('t.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    # 数据切分成训练集，测试集
    train_data, actual_vals = split_data(series.values)

    train_x, train_y = [], []
    for i in range(len(train_data) - seq_size - 1):
        # 训练数据和标签错开了1个单位
        train_x.append(np.expand_dims(train_data[i:i + seq_size], axis=1).tolist())
        train_y.append(train_data[i + 1: i + seq_size + 1])

    test_x, test_y = [], []
    for i in range(len(actual_vals) - seq_size - 1):
        test_x.append(np.expand_dims(actual_vals[i:i + seq_size], axis=1).tolist())
        test_y.append(actual_vals[i + 1: i + seq_size + 1])

    predictor.train(train_x, train_y, test_x, test_y)


    with tf.Session() as sess:
        predicted_vals = predictor.test(sess, test_x)[:, 0]
        #predicted_valsrnn = predictor.testrnn(sess, test_x)[:, 0]
        print('predicted_vals', np.shape(predicted_vals))
        plot_results(train_data, predicted_vals, actual_vals,'predictions.png')
        plt.show()
        # ## 误差图（误差杆状图、误差细节图，以及不同模型中的对比图）
        error_plot = abs(actual_vals[:len(predicted_vals)] - predicted_vals)
        error_plot=error_plot/actual_vals[:len(predicted_vals)]

        e =','.join([str(i) for i in error_plot])
        print(e)
        plt.plot(list(range(len(error_plot))), error_plot, color='r', label='error_lstm_plot')
        plt.title('error_rate')
    #   error_plotr = abs(actual_vals[:len(predicted_valsrnn)] - predicted_valsrnn)
     #  plt.plot(list(range(len(error_plotr))), error_plotr, color='b', label='error_rnn_plot')
        plt.show()
