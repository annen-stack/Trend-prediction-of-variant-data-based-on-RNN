import sys, os
import numpy as np
from PIL import Image
from matplotlib import pyplot

import guzhang_test22


import matplotlib.pyplot as plt
from keras.models import load_model
import sys, os


from PyQt5 import QtCore, QtGui, QtWidgets
from untitled import Ui_MainWindow

import tensorflow as tf
from tensorflow.contrib import rnn

from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication
from PyQt5.QtWidgets import QLabel, QMessageBox, QPushButton, QFrame
from PyQt5.QtGui import  QPixmap, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QSize
from keras.models import load_model
import math
import pywt
from pandas import read_csv



class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setupUi(self)
        self.center()
        self.clearDataArea()

        self.series = read_csv('some.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, dateyu_parser=guzhang_test22.parser)
        # configure
        self.n_lag = 3 # 3
        self.n_seq = 8 # 8
        self.n_test = 15  # 取测试点24
        self.n_epochs = 100
        self.n_batch = 1
        self.n_neurons = 60#60
        self.time=0
        self.value=0
        self.zhi=0
        # prepare data


    def clearDataArea(self):
        self.label1.clear()


    def center(self):
        # 获得窗口
        framePos = self.frameGeometry()
        # 获得屏幕中心点
        scPos = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        framePos.moveCenter(scPos)
        self.move(framePos.topLeft())

        # 窗口关闭事件
    def changedvalue(self,value):
        self.zhi=value


    def train11(self):
        print(0)
        scaler, train, test = guzhang_test22.prepare_data(self.series, self.n_test, self.n_lag, self.n_seq)
        print(9)

        model = guzhang_test22.fit_lstm(train, self.n_lag, self.n_seq, self.n_batch, self.n_epochs, self.n_neurons)

        # make forecasts
        #model=load_model('my_model.h5')

        print(1)
        forecasts = guzhang_test22.make_forecasts(model, self.n_batch, train, test, self.n_lag, self.n_seq)
        print(2)
        # inverse transform forecasts and test
        forecasts = guzhang_test22.inverse_transform(self.series, forecasts, scaler, self.n_test + 2)
        print(3)
        actual = [row[self.n_lag:] for row in test]
        actual = guzhang_test22.inverse_transform(self.series, actual, scaler, self.n_test + 2)
        print(4)
        # evaluate forecasts

        # plot forecasts

        series = self.series
        n_test=self.n_test+2
        plt.figure()
            # plot the entire dataset in blue
        plt.plot(series.values,label='actual data')
            # forecasts[0][1] = 360
            # forecasts[0][2] = 330
        print(5)

        xiaxian = []
        shangxian = []

        for i in range(len(forecasts[0])):
            xiaxian.append(forecasts[0][i] - self.zhi)
            shangxian.append(forecasts[0][i] + self.zhi)

            # plot the forecasts in red
        off_s = len(series) - n_test - 1
        off_e = off_s + len(forecasts[0]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[0]
        plt.plot(xaxis, yaxis, color='red',label='predicted data',linestyle='-.')

        # xaxis1 = [x - 1 for x in range(off_s, off_e)]
        # yaxis1 = [series.values[off_s]] + shangxian
        # plt.plot(xaxis1, yaxis1, color='y', linestyle='--',label='upper limit')
        #
        # xaxis2 = [x + 1 for x in range(off_s, off_e)]
        # yaxis2 = [series.values[off_s]] + xiaxian
        # plt.plot(xaxis2, yaxis2, color='g', linestyle='-.',label='lower limit')

        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend()
        print(6)


        filename = 'yuzhi.png'
        print(7)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
        print(8)


        self.time, self.value = guzhang_test22.evaluate_forecasts(actual, forecasts, self.n_lag, self.n_seq, shangxian, xiaxian)

        yuzhi_plot = QtGui.QPixmap('yuzhi.png').scaled(self.label1.width(), self.label1.height())
        self.label1.setPixmap(yuzhi_plot)




    def xianshi(self):
        self.textEdit1.setPlainText("遥测数据出现异常状态的时间发生在：%d" % self.time)
        self.textEdit2.setPlainText("异常值偏离正常值%.2f~%.2f" %(self.value-1,self.value+1))





if __name__ == "__main__":
    tf.reset_default_graph()

    app = QApplication(sys.argv)
    Gui = MainWindow()
    Gui.show()

    sys.exit(app.exec_())