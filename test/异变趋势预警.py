import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt


# 卫星数据模拟
# 预测结果分析：

# 首先卫星正常运行，在a点跳变，得之后在a+b点才发现超出阈值，发生故障，那个时候太晚了。
# 我们在以a-c为起点，往前取N个点，进行RNN训练，然后预测后面的点，以及置信区间，当a左右的点x未在置信区间中，则说明
# x为异常点，开始异常，立即阻断，则比一般发现异常提前了好几天。

# LSTM可以解决梯度消失的问题（不是梯度爆炸）
# LSTM:cell=memory+input，gate有四个，影响会永远都存在，不会被洗掉，除非forget gate关闭，所以没梯度消失的问题（if forget gate open）。
# RNN:每个时间点，memory里的秩序都会被洗掉
# GRU:simpler than LSTM,gate 只有两个门，参数比较少。当LSTM的overfitting严重，尝试GRU

def load_data():
    # seq = np.array(
    #     [112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104.,
    #      118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133.,
    #      114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184.,
    #      162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242.,
    #      209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264.,
    #      272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264.,
    #      302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270.,
    #      315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313.,
    #      318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356.,
    #      348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318.,
    #      362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360.,
    #      342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
    #      417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390.,
    #      432.], dtype=np.float32)
    # seq = np.array(
    #     [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
    #      1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
    #      -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.7,
    #      -0.7, -0.7, -0.7, -0.7, -0.7 - 0.7, -0.7, -0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, -1,
    #      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4.5, 4.6, 1.7, 4.1, 4.3, 1.7, 1.6, 3.9, 4.2, 1.9, 1.6, 3.9,
    #      4.2, 1.9, 1.8, 4.5, 4.6, 1.7, 4.1, 4.3, 1.7, 1.6, 3.9, 4.2, 1.9, 1.6, 3.9, 4.2, 1.9, 1.8, 3.9, 4.2, 1.9, 1.8,
    #      1.8, 3.9, 3.9, 4.2, 1.9, 1.8, 1.8, 3.9, 4.2, 1.9, 1.8, 2.6, 4.9, 4.3, 1.7, 2.6, 4.9, 4.3, 1.7, 4.1, 4.3, 1.7,
    #      4.1, 4.3, 1.7, 1.6, 4.3, 1.7, 1.6, 3.9, 4.2, 1.9, 3.8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    #      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     dtype=np.float32)
    seq=np.array([362., 405.,417., 415, 388, 400, 418., 419., 461., 472., 535., 622.,  508., 461., 390.,389.,
362., 405.,417., 415, 378, 402, 417., 419., 461., 472., 536., 626.,  508., 465., 393.,387.,
362., 405., 417., 425, 378, 401, 417., 417., 461., 472., 533., 626.,  504., 465., 383., 385.,
362., 405.,427., 415, 388, 400, 416., 419., 461., 480., 535., 615.,  508., 451., 390.,379.,
362., 395., 423., 414, 388, 400, 419., 419., 471., 476., 535., 623.,  508., 461., 390., 399.,
362., 395., 423., 414, 628, 400, 419., 419., 571., 596., 612., 523.,  408., 661., 390., 399.,
362., 405.,427., 416, 388, 402, 418., 419., 461., 469., 535., 620., 508., 461., 390.,389.,
                  362., 405., 417., 415, 378, 402, 417., 419., 461., 472., 536., 626., 508., 465., 393., 387.,

                  ])
    print(seq.mean(axis=0))
    print(seq.std(axis=0))
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)

    return seq


def split_data(data, percent_train=0.80):
    num_rows = len(data)
    train_data, test_data = [], []
    for idx, row in enumerate(data):
      #  if idx < num_rows * percent_train:
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

    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    plt.bar(index, error_values, width, label="error", color="#87CEFA")
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

    x1 = actual_vals

    x2 = predicted_vals

    MAPE = 0

    # 3.平均绝对误差(MAE)

    a = 0
    for i in range(len(x1)):
        m = abs(x1[i] - x2[i])
        a = a + m
    print('平均绝对误差：%d' % (a / len(x1)))


# 把数据切分成80%训练集、20%测试集



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
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.02).minimize(self.cost)

        # Auxiliary ops
        self.saver = tf.train.Saver()

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn.BasicLSTMCell(self.hidden_dim)
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        # 返回长度为5的一维数组
        return out

    def train(self, train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            max_patience = 3
            patience = max_patience
            # 最小err指定为无限
            min_test_err = float('inf')
            step = 0
            # test_err出现3次浮动停下
            for i in range(1200):
                _, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if i % 10 == 0:
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
                    print('step:{}\t\ttrain err：{}\t\ttest err：{}'.format(step, train_err, test_err))

            save_path = self.saver.save(sess, './model/')
            print('Model saved to {}'.format(save_path))

    def test(self, sess, test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, './model/')
        output = sess.run(self.model(), feed_dict={self.x: test_x})
        return output


def plot_results(train_x, predictions, actual, filename):
    plt.figure()
    num_train = len(train_x)
    # 训练集。plt.plot(x,y,format_string,**kwargs) x轴数据，y轴数据
    #  plt.plot(list(range(num_train)), train_x, color='b', label='training data')
    # 预测集
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(list(range(len(predictions))), predictions, color='r', label='预测值',linestyle='--')
    # 真实值
    plt.plot(list(range(len(actual))), actual, color='b', label='实际值')
    # 加图例
    plt.xlim(0,118)
    plt.xlabel('时间')
    plt.ylabel('数据值')
    plt.legend()

    # error=abs(actual-prediction)

    # plt.plot(list(range(num_train, num_train + len(actual))), actual, color='b', label='test_data')

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

if __name__ == '__main__':
    # 序列长度，基于一个小片段去预测下一个值
    seq_size = 5
    predictor = SeriesPredictor(input_dim=1, seq_size=5, hidden_dim=80)
    data = load_data()
    # 数据切分成训练集，测试集
    train_data, actual_vals = split_data(data)

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
        # [:, 0]为什么是0呢？不应该是-1么？
        # 1、用训练好的模型'./model/'预测，所以第一个值也是基于前面序列预测到的。
        # 2、画图时使预测值和真实值具有对照性。
        predicted_vals = predictor.test(sess, test_x)[:, 0]
        print('predicted_vals', np.shape(predicted_vals))
        plot_results(train_data, predicted_vals, actual_vals, 'predictions.png')
        #   plt.show()
        ## 误差图（误差杆状图、误差细节图，以及不同模型中的对比图）
        error_plot = abs(actual_vals[:len(predicted_vals)] - predicted_vals)
        arr = []
        beta = 0.7
        arr.append(error_plot[0])
        for i in range(len(error_plot) - 1):
            a = beta * error_plot[i] + (1 - beta) * error_plot[i + 1]
            arr.append(a)
        l=[]
        for i in range(len(arr)):
            l.append(3.234)

        error_plot = abs(actual_vals[:len(predicted_vals)] - predicted_vals)
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(list(range(len(error_plot))), error_plot, color='r', label='预测误差')
        plt.plot(list(range(len(l))), l, color='y', label='阈值')
        plt.legend()
        plt.title('预测误差图像')
        plt.xlabel('时间')
        plt.ylabel('误差值')
        plt.xlim(0, 118)
        plt.show()


