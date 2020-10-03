import sys

import pandas as pd

from model import SeriesPredictor
from PyQt5 import QtGui
from pyqt import Ui_MainWindow
import yuchuli
import tensorflow as tf
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication, QFileDialog
import matplotlib.pyplot as plt
import numpy as np
import math
import pywt
from pandas import read_csv

import guzhang_test22

import sys, os
import numpy as np
from PIL import Image
from matplotlib import pyplot

import guzhang_test22


import matplotlib.pyplot as plt
from keras.models import load_model
import sys, os


from PyQt5 import QtCore, QtGui, QtWidgets


import tensorflow as tf
from tensorflow.contrib import rnn

from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication
from PyQt5.QtWidgets import QLabel, QMessageBox, QPushButton, QFrame
from PyQt5.QtGui import  QPixmap, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QSize, QFile
from keras.models import load_model
import math
import pywt
from pandas import read_csv
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class MainWindow(QMainWindow, Ui_MainWindow, SeriesPredictor):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.series = read_csv('xiaoboo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,
                               date_parser=guzhang_test22.parser)
        self.setupUi(self)
        self.setWindowTitle('卫星遥测数据预测demo')
        self.center()
        self.datarec = np.ones((1,144), dtype=float)
        self.a = []
        self.b=[]
        self.datarec1 = np.ones((1, 144), dtype=float)
        # configure
        self.n_lag = 90
        self.n_seq = 25
        self.n_test = 20  # 取测试点  24
        self.n_epochs = 1500
        self.n_batch = 1
        self.n_neurons = 2
        self.time=0
        self.value=0
        self.zhi=90
        self.openfile_name=None

        self.clearDataArea()

    def clearDataArea(self):
        self.label1.clear()
        self.label3.clear()
        self.label2.clear()
        self.label4.clear()


    def center(self):
        # 获得窗口
        framePos = self.frameGeometry()
        # 获得屏幕中心点
        scPos = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        framePos.moveCenter(scPos)
        self.move(framePos.topLeft())


    def changedvalue(self,value):
        self.zhi=value

    def btnshanbing_click(self):
        self.n_lag = int(self.lineEdit11.text())
        print(self.n_lag)
        self.n_seq = int(self.lineEdit12.text())
        print(self.n_seq)
        m = int(self.lineEdit13.text())
        self.n_test=len(self.series)-2-m
        print(self.n_test)


    def openfile(self):
        fname = QFileDialog.getOpenFileName(self, '选择文件', '', 'Excel files(*.csv)')

        self.dataframe = read_csv(fname[0], usecols=[1], engine='python', skipfooter=3)


    def yezhi(self):
        #original data.csv

        dataset = self.dataframe.values
        dataset = dataset.astype('float32')
        org_l=dataset #原始数据

        self.datarec= yuchuli.preprocessl(org_l)



        for i in range(len(dataset)):
            self.a.append(dataset[i][0])
        print(self.a)
        for i in range(len(self.datarec)):
            self.b.append(self.datarec[i][0])
        print(self.b)
        c=[]
        if len(self.a)>150:
            c=[]
        else:
            for i in range(len(dataset)):
                if self.b[i]!=self.a[i]:
                    c.append(i)

        print(c)
        plt.subplot(2, 1, 1)
        plt.plot(range(len(org_l)), org_l)
        plt.xlabel('time')
        plt.ylabel('value')
        plt.title("original data")
        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.b)), self.b)
        plt.xlabel('time')
        plt.ylabel('value')
        plt.title("The signal after removing the wild value")


        filename = 'yezhi.png'
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

        yezhi_plot = QtGui.QPixmap('yezhi.png').scaled(self.label1.width(), self.label1.height())

        self.label1.setPixmap(yezhi_plot)

        print(self.a)

        self.textEdit3.setPlainText("遥测数据出现野值的时间发生在：%s" %c)





    def xiaobo(self):
        origin = self.b
        wavefunc = 'db6'

        lv = 4
        m = 4
        n = 4

        # nx = wgn(origin, 30)
        # origin = np.array(origin)
        # nx = np.array(nx)
        # origin_withnoise = origin + nx

        index = []
        data = []
        for i in range(len(origin) - 1):
            X = float(i)
            Y = float(origin[i])
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
        plt.plot(range(len(origin)), origin)
        plt.xlabel('时间')
        plt.ylabel('数据值')
        plt.title("The original signal")
        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.datarec1)), self.datarec1)
        plt.xlabel('time')
        plt.ylabel('value')
        plt.title("De-noised signal ")

        filename = 'xiaobo.png'
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

        seq = np.array(self.datarec1, dtype=np.int)

        index = pd.date_range(start='2000-01', periods=len(seq),freq='m')
        time = pd.Series(seq, index=index)
        time.to_csv('xiaoboo.csv', index=True)

        xiaobo_plot = QtGui.QPixmap('xiaobo.png').scaled(self.label2.width(), self.label2.height())

        self.label2.setPixmap(xiaobo_plot)

    def train11(self):
        print(0)
        scaler, train, test = guzhang_test22.prepare_data(self.series, self.n_test, self.n_lag, self.n_seq)
        print(9)

        # make forecasts
        model=load_model('my_model.h5')
        #model = guzhang_test22.fit_lstm(train, self.n_lag, self.n_seq, self.n_batch, self.n_epochs, self.n_neurons)

        print(1)
        forecasts = guzhang_test22.make_forecasts(model, self.n_batch, train, test, self.n_lag, self.n_seq)
        print(2)
        # inverse transform forecasts and test
        forecasts = guzhang_test22.inverse_transform(self.series, forecasts, scaler, self.n_test + 2)
        print(3)
        actual = [row[self.n_lag:] for row in test]
        actual = guzhang_test22.inverse_transform(self.series, actual, scaler, self.n_test + 2)
        print(4)

        series = self.series
        n_test=self.n_test
        plt.figure()
            # plot the entire dataset in blue
        plt.plot(series.values,label='实际值')
            # forecasts[0][1] = 360
            # forecasts[0][2] = 330
        print(5)

        xiaxian = []
        shangxian = []

        # for i in range(len(forecasts[0])):
        #     xiaxian.append(forecasts[0][i] - self.zhi)
        #     shangxian.append(forecasts[0][i] + self.zhi)

        n = len(forecasts[0])
        sigma = np.std(forecasts[0], ddof=1)
        if self.zhi==90:
            z=1.25
        if self.zhi==95:
            z=1.6449

        zsigman = z* float(sigma) / math.sqrt(n)
        for i in range(len(forecasts[0])):
            # xiaxian.append(forecasts[0][i] - zhi)
            # shangxian.append(forecasts[0][i] + zhi)
            shangxian.append(forecasts[0][i] + 5.1*zsigman+1)
            xiaxian.append(forecasts[0][i] - 5.1*zsigman)

            # plot the forecasts in red
        off_s = len(series) - n_test - 1
        off_e = off_s + len(forecasts[0]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[0]
        plt.plot(xaxis, yaxis, color='red',label='预测值',linewidth='3')

        xaxis1 = [x for x in range(off_s, off_e)]
        yaxis1 = [series.values[off_s]] + shangxian
        plt.plot(xaxis1, yaxis1, color='y', linestyle='--',label='上限')


        xaxis2 = [x  for x in range(off_s, off_e)]
        yaxis2 = [series.values[off_s]] + xiaxian
        plt.plot(xaxis2, yaxis2, color='g', linestyle='-.',label='下限')

        for i in range(off_s,off_e):
            if series.values[i]<yaxis2[i-off_s]:
                self.time=i
                self.value=abs(series.values[i]-yaxis2[i-off_s])
                break
            elif series.values[i]>yaxis1[i-off_s]:
                self.time=i
                self.value=abs(series.values[i]-yaxis1[i-off_s])
                break
            else:
                self.time=0
                self.value=0

        print(self.time)
        plt.xlabel('时间')
        plt.ylabel('数据值')
        plt.xlim(80,142)

        plt.legend()
        print(6)


        filename = '3.png'
        print(7)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
        print(8)


        A=guzhang_test22.evaluate_forecasts(actual, forecasts,  self.n_seq)
        print(9)

        yuzhi_plot = QtGui.QPixmap('3.png').scaled(self.label3.width(), self.label3.height())
        self.label3.setPixmap(yuzhi_plot)

        wucha_plot = QtGui.QPixmap('4.png').scaled(self.label4.width(), self.label4.height())
        self.label4.setPixmap(wucha_plot)


    def xianshi(self):

        self.textEdit1.setPlainText("遥测数据出现异常状态的时间发生在：%d~%d" % (self.time,self.time+1))
        self.textEdit2.setPlainText("异常值偏离正常值%.2f~%.2f" %(self.value-1,self.value+1))


if __name__ == "__main__":
    tf.reset_default_graph()

    app = QApplication(sys.argv)
    Gui = MainWindow()
    Gui.show()

    sys.exit(app.exec_())