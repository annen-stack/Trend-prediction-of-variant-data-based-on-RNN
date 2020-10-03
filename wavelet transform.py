
import numpy as np
import pywt
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets

def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def xiaobo(self):
    origin = self.datarec
    wavefunc = 'db6'

    lv = 4
    m = 4
    n = 4

    nx = wgn(origin, 30)
    origin = np.array(origin)
    nx = np.array(nx)
    origin_withnoise = origin + nx

    index = []
    data = []
    for i in range(len(origin_withnoise) - 1):
        X = float(i)
        Y = float(origin_withnoise[i])
        index.append(X)
        data.append(Y)

        # 打包为函数，方便调节参数。  lv为分解层数；data为最后保存的dataframe便于作图；index_list为待处理序列；wavefunc为选取的小波函数；m,n则选择了进行阈值处理的小波系数层数

        # 分解
    coeff = pywt.wavedec(data, wavefunc, mode='sym', level=lv)  # 按 level 层分解，使用pywt包进行计算， cAn是尺度系数 cDn为小波系数

    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0  # sgn函数

    # 去噪过程
    for i in range(m, n + 1):  # 选取小波系数层数为 m~n层，尺度系数不需要处理
        cD = coeff[i]
        for j in range(len(cD)):
            Tr = np.sqrt(2 * np.log(len(cD)))  # 计算阈值
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) - Tr  # 向零收缩
            else:
                coeff[i][j] = 0  # 低于阈值置零

        # 重构
    self.datarec1 = pywt.waverec(coeff, wavefunc)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(len(origin_withnoise)), origin_withnoise)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title("The signal after removing the wild value")
    plt.subplot(2, 1, 2)
    plt.plot(range(len(self.datarec1)), self.datarec1)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title("De-noised signal using wavelet techniques")

    filename = 'xiaobo.png'
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

    xiaobo_plot = QtGui.QPixmap('xiaobo.png').scaled(self.label3.width(), self.label3.height())

    self.label3.setPixmap(xiaobo_plot)