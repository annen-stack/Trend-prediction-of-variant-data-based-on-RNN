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
    seq = np.array([18.41406098,  20.01108275,  20.54899315,  17.84945733,
        17.72776385,  14.1537172 ,  23.10528481,  18.11915952,
        19.71313976,  16.64865462,  21.81541276,  18.28434935,
        18.21543113,  18.36871213,  19.4070513 ,  19.49161073,
        17.96165062,  16.48878499,  12.29733141,  20.15662625,
        21.97888679,  23.73588206,  12.54155697,  12.98412254,
        18.06058713,  17.67810285,  14.82233521,  16.62752209,
        22.97215539,  19.93602746,  17.52079068,  16.67079317,
        14.23088385,  15.42183135,  15.98280906,  12.94370615,
        14.23207748,  18.78742476,  16.12837062,  18.2866613 ,
        13.80231232,  22.18141293,  15.10876194,  15.00921027,
        22.7356711 ,  13.976639  ,  15.79635406,  19.09124675,
        16.2888073 ,  15.93899519,  18.40157807,  17.57938482,
        19.32482386,  20.20012669,  15.77692586,  20.06723439,
        20.7213266 ,  18.75133459,  16.2915961 ,  21.4301795 ,
        18.16542688,  21.84835057,  20.57135685,  21.2444208 ,
        22.00577539,  16.86768415,  18.75192791,  15.54229744,
        17.523965  ,  20.44078186,  20.46155089,  14.30536421,
        24.67500145,  17.96560828,  19.55985065,  17.96987717,
        15.28434428,  23.64448589,  17.28952403,  18.15556391,
        18.3814342 ,  13.53739248,  21.16321907,  16.40625388,
        15.1952001 ,  17.56801854,  20.82500777,  20.38651622,
        18.13207849,  15.66685181,  15.38258139,  18.27302851,
        12.90825876,  24.36757854,  22.79603245,  13.78681248,
        18.83593826,  16.28059936,  18.01169338,  18.22774022,
        21.05217927,  16.37170363,  24.66670284,  18.56423315,
        17.39078487,  16.93866112,  19.35778982,  21.64733672,
        17.89816696,  20.75773157,  22.27059156,  18.95634818,
        15.53904084,  13.59782212,  14.7830166 ,  20.4095604 ,
        19.89422918,  14.87868821,  18.84421262,  17.36461803,
        17.10876633,  16.14936192,  14.4689037 ,  19.35275089,
        14.0053305 ,  15.44144652,  16.24734164,  14.98210153,
        15.1706537 ,  17.10690462,  19.61480617,  14.69877929,
        17.21783674,  10.67028784,  22.01573784,  22.05383771,
        15.57466636,  19.03467803,  20.51363819,  20.82733793,
        15.32407136,  17.73254849,  14.90374616,  15.68674334,
        17.89671283,  13.65046155,  19.13619982,  14.83240759,
        21.46895624,  15.07755705,  14.71905099,  18.62084201,
        15.08517501,  22.24668955,  12.56864533,  17.21532993,
        18.18930228,  17.85085462,  15.80326154,  13.91871032,
        20.53377274,  18.53947121,  17.55642243,  13.83441715,
        19.79443369,  18.6782417 ,  16.92731304,  18.57288801,
        17.37447799,  17.71379487,  11.45989472,  18.14092096,
        21.34533108,  17.37452632,  15.30697234,  15.77154346,
        25.30594801,  10.87968487,  15.43815045,  19.88612035,
        18.0126976 ,  19.05637821,  18.65933967,  15.26428959,
        12.19738483,  16.3465842 ,  22.74048406,  17.07615893,
        23.61975043,  21.42917636,  20.0124415 ,  18.92879279,
        17.44128688,  19.54572884,  17.38213741,  21.11431209,
        15.95667166,  14.75645489,  19.38825566,  14.81232212,18.83593826,  16.28059936,  18.01169338,  18.22774022,
        21.05217927,  16.37170363,  24.66670284,  18.56423315,
        17.39078487,  16.93866112,  19.35778982,  21.64733672,
        17.89816696,  20.75773157,  22.27059156,  18.95634818,
        15.53904084,  13.59782212,  14.7830166 ,  20.4095604 ,
        19.89422918,  14.87868821,  18.84421262,  17.36461803,
        17.10876633,  16.14936192,  14.4689037 ,  19.35275089,
        14.0053305 ,  15.44144652,  16.24734164,  14.98210153,
        15.1706537 ,  17.10690462,  19.61480617,  14.69877929,
        17.21783674,  10.67028784,  22.01573784,  22.05383771,
        15.57466636,  19.03467803,  20.51363819,  20.82733793,
        15.32407136,  17.73254849,  14.90374616,  15.68674334,
        17.89671283,  13.65046155,  19.13619982,  14.83240759,
        21.46895624,  15.07755705,  14.71905099,  18.62084201,
        15.08517501,  22.24668955,  12.56864533,  17.21532993,
        18.18930228,  17.85085462,  15.80326154,  13.91871032,],
        dtype=np.float32)        # [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
        #  1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
        #  -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.7,
        #  -0.7, -0.7, -0.7, -0.7, -0.7 - 0.7, -0.7, -0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, -1,
        #  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4.5, 4.6, 1.7, 4.1, 4.3, 1.7, 1.6, 3.9, 4.2, 1.9, 1.6, 3.9,
        #  4.2, 1.9, 1.8, 4.5, 4.6, 1.7, 4.1, 4.3, 1.7, 1.6, 3.9, 4.2, 1.9, 1.6, 3.9, 4.2, 1.9, 1.8, 3.9, 4.2, 1.9, 1.8,
        #  1.8, 3.9, 3.9, 4.2, 1.9, 1.8, 1.8, 3.9, 4.2, 1.9, 1.8, 2.6, 4.9, 4.3, 1.7, 2.6, 4.9, 4.3, 1.7, 4.1, 4.3, 1.7,
        #  4.1, 4.3, 1.7, 1.6, 4.3, 1.7, 1.6, 3.9, 4.2, 1.9, 3.8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        #  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    return seq
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
    for i in range(len(x1)):
        m = abs((x1[i] - x2[i]) / x1[i])
        MAPE = MAPE + m
    print('平均绝对百分比误差：%d' % (MAPE / len(x1)))

    # 2.均方根误差RMSE

    a = 0

    # 3.平均绝对误差(MAE)

    a = 0
    for i in range(len(x1)):
        m = abs(x1[i] - x2[i])
        a = a + m
    print('平均绝对误差：%d' % (a / len(x1)))


# 把数据切分成80%训练集、20%测试集
def split_data(data, percent_train=0.80):
    num_rows = len(data)
    train_data, test_data = [], []
    for idx, row in enumerate(data):

        train_data.append(row)

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
            for i in range(1000):
                _, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if i % 10 == 0:
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
                    print('step:{}\t\ttrain err：{}\t\ttest err：{}'.format(step, train_err, test_err))

            save_path = self.saver.save(sess, './model11/')
            print('Model saved to {}'.format(save_path))

    def test(self, sess, test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, './model11/')
        output = sess.run(self.model(), feed_dict={self.x: test_x})
        return output


def plot_results(train_x, predictions, actual, filename):
    plt.figure()
    num_train = len(train_x)
    # 训练集。plt.plot(x,y,format_string,**kwargs) x轴数据，y轴数据
    #  plt.plot(list(range(num_train)), train_x, color='b', label='training data')
    # 预测集
    plt.plot(list(range(len(predictions))), predictions, color='b', label='实际值')
    # 真实值
    plt.plot(list(range(len(actual))), actual, color='g', label='预测值',linestyle='--')
    plt.xlim(0,220)
    plt.xlabel('时间')
    plt.ylabel('数据值')
    # 加图例
    plt.legend()

    # error=abs(actual-prediction)

    # plt.plot(list(range(num_train, num_train + len(actual))), actual, color='b', label='test_data')

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


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


if __name__ == '__main__':
    # 序列长度，基于一个小片段去预测下一个值
    seq_size = 5
    predictor = SeriesPredictor(input_dim=1, seq_size=5, hidden_dim=100)
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

   # predictor.train(train_x, train_y, test_x, test_y)

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

        arr_mean = np.mean(arr[100:151])
        arr_var = np.var(arr[100:151])

        m = arr_mean + 0.2 * arr_var
        n = []
        for i in range(len(arr[100:151])):
            n.append(m)

        error_plot = abs(actual_vals[:len(predicted_vals)] - predicted_vals)
        plt.figure()
        plt.plot(list(range(len(error_plot))), error_plot, color='r', label='error_plot')
        plt.title('the error value')
        plt.xlabel('time')
        plt.ylabel('error_value')
        plt.show()

        plt.figure()
        plt.plot(list(range(len(arr))), arr, color='r', label='error_smoothed_plot')
        plt.title('the smoothed error rate image')
        plt.xlabel('time')
        plt.ylabel('error_rate')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(list(range(100, 151)), arr[100:151], color='r', label='error_plot')
        plt.plot(list(range(100, 151)), n, color='y', label='threshold')
        plt.xlabel('time')
        plt.ylabel('error_rate')
        plt.legend()
        plt.show()



#unitied