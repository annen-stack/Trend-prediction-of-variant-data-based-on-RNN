import pywt

import math

import numpy as np
import tensorflow as tf
from keras.losses import mean_squared_error
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt


# ��������ģ��
# Ԥ����������

# ���������������У���a�����䣬��֮����a+b��ŷ��ֳ�����ֵ���������ϣ��Ǹ�ʱ��̫���ˡ�
# ��������a-cΪ��㣬��ǰȡN���㣬����RNNѵ����Ȼ��Ԥ�����ĵ㣬�Լ��������䣬��a���ҵĵ�xδ�����������У���˵��
# xΪ�쳣�㣬��ʼ�쳣��������ϣ����һ�㷢���쳣��ǰ�˺ü��졣


def load_data():
    seq = np.array(
          [112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104.,
          118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133.,
         114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184.,
         162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242.,
         209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264.,
         272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264.,
         302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270.,
         315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313.,
         318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356.,
         348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318.,
         362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360.,
         342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
         417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390.,
         432.], dtype=np.float32)
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    return seq


def split_data(data, percent_train=0.80):
    num_rows = len(data)
    train_data, test_data = [], []
    for idx, row in enumerate(data):
        #if idx < num_rows * percent_train:
        train_data.append(row)
        #else:
        test_data.append(row)
    return train_data, test_data


def plot_error(actual_vals, predicted_vals, train_x, filename):
    plt.figure()
    num_train = len(train_x)
    width = 0.35
    error_values = abs(actual_vals[:len(predicted_vals)] - predicted_vals)

    index = list(range(num_train, num_train + len(predicted_vals)))

    # ������״ͼ, ÿ�����ӵ���ɫΪ������ɫ
    plt.bar(index, error_values, width, label="error", color="#87CEFA")
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

        x1 = actual_vals
        x2 = predicted_vals
        MAPE = 0
        for i in range(len(x1)):
            m = abs((x1[i] - x2[i]) / x1[i])
            MAPE = MAPE + m
        print('ƽ�����԰ٷֱ���%d' % (MAPE / len(x1)))

        # 2.���������RMSE

        a = 0
        for i in range(len(x1)):
            m = (x1[i] - x2[i]) ** 2
            a = a + m
        print('��������%d' % (math.sqrt(a / len(x1))))

        # 3.ƽ���������(MAE)

        a = 0
        for i in range(len(x1)):
            m = abs(x1[i] - x2[i])
            a = a + m
        print('ƽ��������%d' % (a / len(x1)))


# �������зֳ�80%ѵ������20%���Լ�
# def split_data(data, percent_train=0.80):
#     num_rows = len(data)
#     train_data, test_data = [], []
#     for idx, row in enumerate(data):
#         if idx < num_rows * percent_train:
#             train_data.append(row)
#         else:
#             test_data.append(row)
#     return train_data, test_data


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
        # 5��СƬ�Σ���5��Ԥ��ֵ
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
        cell = rnn.BasicLSTMCell(self.hidden_dim)
        cell=rnn.DropoutWrapper(cell,output_keep_prob=0.8)
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        # ���س���Ϊ5��һά����
        return out

    def train(self, train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            max_patience = 3
            patience = max_patience
            # ��Сerrָ��Ϊ����
            min_test_err = float('inf')
            step = 0
            # test_err����3�θ���ͣ��
            trainr=[]
            while step<800:
                _, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                trainr.append(train_err)

                if step % 10 == 0:
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
                    print('step:{}\t\ttrain err��{}\t\ttest err��{}'.format(step, train_err, test_err))
                step += 1
            save_path = self.saver.save(sess, './model/')
            print('Model saved to {}'.format(save_path))
            plt.figure()
            print(trainr)
            plt.plot(list(range(len(trainr))),trainr,label='lr=0.01')
            plt.legend()
            plt.title('loss')
            plt.show()


    def test(self, sess, test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, './model/')
        output = sess.run(self.model(), feed_dict={self.x: test_x})
        return output


def plot_results(train_x, predictions, actual, filename):
    plt.figure()
    num_train = len(train_x)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    # ѵ������plt.plot(x,y,format_string,**kwargs) x�����ݣ�y������
   # plt.plot(list(range(num_train)), train_x, color='b', label='training data')
    # Ԥ�⼯
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='Ԥ��ֵ')
    # ��ʵֵ
    plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='���Լ�')
    # ��ͼ��
    plt.legend()
    plt.xlim(140,220)
    plt.xlabel('ʱ��')
    plt.ylabel('����ֵ')

    # error=abs(actual-prediction)

    # plt.plot(list(range(num_train, num_train + len(actual))), actual, color='b', label='test_data')

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


if __name__ == '__main__':
    # ���г��ȣ�����һ��СƬ��ȥԤ����һ��ֵ
    seq_size = 5
    predictor = SeriesPredictor(input_dim=1, seq_size=5, hidden_dim=100)
    data = load_data()
    # �����зֳ�ѵ���������Լ�
    train_data, actual_vals = split_data(data)

    train_x, train_y = [], []
    for i in range(len(train_data) - seq_size - 1):
        # ѵ�����ݺͱ�ǩ����1����λ
        train_x.append(np.expand_dims(train_data[i:i + seq_size], axis=1).tolist())
        train_y.append(train_data[i + 1: i + seq_size + 1])

    test_x, test_y = [], []
    for i in range(len(actual_vals) - seq_size - 1):
        test_x.append(np.expand_dims(actual_vals[i:i + seq_size], axis=1).tolist())
        test_y.append(actual_vals[i + 1: i + seq_size + 1])

    predictor.train(train_x, train_y, test_x, test_y)

    with tf.Session() as sess:
        # [:, 0]Ϊʲô��0�أ���Ӧ����-1ô��
        # 1����ѵ���õ�ģ��'./model/'Ԥ�⣬���Ե�һ��ֵҲ�ǻ���ǰ������Ԥ�⵽�ġ�
        # 2����ͼʱʹԤ��ֵ����ʵֵ���ж����ԡ�
        predicted_vals = predictor.test(sess, test_x)[:, 0]
        print('predicted_vals', np.shape(predicted_vals))
        plot_results(train_data, predicted_vals, actual_vals, 'predictions.png')
        plt.show()
        ## ���ͼ������״ͼ�����ϸ��ͼ���Լ���ͬģ���еĶԱ�ͼ��
        # rmse
        #error_plot = abs(actual_vals[:len(predicted_vals)] - predicted_vals)
        x2 = actual_vals
        x1 = predicted_vals
        MAPE = 0
        for i in range(len(x1)):
            m = abs((x1[i] - x2[i])/ x1[i])
            MAPE = MAPE + m
        print('ƽ�����԰ٷֱ���%.5f' % (MAPE / len(x1)))

        # 2.���������RMSE

        a = 0
        for i in range(len(x1)):
            m = (x1[i] - x2[i]) ** 2
            a = a + m
        print('��������%.5f' % (math.sqrt(a / len(x1))))

        # 3.ƽ���������(MAE)

        a = 0
        for i in range(len(x1)):
            m = abs(x1[i] - x2[i])
            a = a + m
        print('ƽ��������%.5f' % (a / len(x1)))

        plt.show()