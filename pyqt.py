# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'G.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QSlider


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("卫星遥测数据预测demo")
        MainWindow.resize(1512,892)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")



        self.pushButton_1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_1.setGeometry(QtCore.QRect(170,350, 93, 28))
        self.pushButton_1.setObjectName("pushButton")


        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(720, 350, 93, 28))
        self.pushButton_2.setObjectName("pushButton")


        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(1200, 350, 93, 28))
        self.pushButton_3.setObjectName("pushButton_2")

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(770, 690, 113, 28))
        self.pushButton_4.setObjectName("pushButton_3")

        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(10,10,93,28))
        self.pushButton_5.setObjectName("pushButton")
        self.pushButton_5.setText("0.打开文件")
        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.label16 = QtWidgets.QLabel(self.centralwidget)
        self.label16.setGeometry(QtCore.QRect(200, 540, 72, 15))
        self.label16.setObjectName("label16")
        self.label17 = QtWidgets.QLabel(self.centralwidget)
        self.label17.setGeometry(QtCore.QRect(130, 570, 131, 16))
        self.label17.setObjectName("label17")
        self.label18 = QtWidgets.QLabel(self.centralwidget)
        self.label18.setGeometry(QtCore.QRect(130, 610, 121, 16))
        self.label18.setObjectName("label18")
        self.label19 = QtWidgets.QLabel(self.centralwidget)
        self.label19.setGeometry(QtCore.QRect(150, 650, 72, 15))
        self.label19.setObjectName("label19")
        self.lineEdit11 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit11.setGeometry(QtCore.QRect(270, 570, 113, 21))
        self.lineEdit11.setObjectName("lineEdit11")

        self.lineEdit12 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit12.setGeometry(QtCore.QRect(270, 600, 113, 21))
        self.lineEdit12.setObjectName("lineEdit12")
        self.lineEdit13 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit13.setGeometry(QtCore.QRect(270, 640, 113, 21))
        self.lineEdit13.setObjectName("lineEdit13")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(210, 700, 93, 28))
        self.pushButton.setObjectName("pushButton")


        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(10, 40, 481, 261))
        self.label1.setMouseTracking(False)
        self.label1.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label1.setFrameShape(QtWidgets.QFrame.Box)
        self.label1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label1.setLineWidth(1)
        self.label1.setMidLineWidth(0)
        self.label1.setText("")
        self.label1.setObjectName("label1")


        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(520,40,481,261))
        self.label2.setMouseTracking(False)
        self.label2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label2.setFrameShape(QtWidgets.QFrame.Box)
        self.label2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label2.setLineWidth(1)
        self.label2.setMidLineWidth(0)
        self.label2.setText("")
        self.label2.setObjectName("label2")


        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setGeometry(QtCore.QRect(1020,40,481,261))
        self.label3.setMouseTracking(False)
        self.label3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label3.setFrameShape(QtWidgets.QFrame.Box)
        self.label3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label3.setLineWidth(1)
        self.label3.setMidLineWidth(0)
        self.label3.setText("")
        self.label3.setObjectName("label3")

        self.label15 = QtWidgets.QLabel(self.centralwidget)
        self.label15.setGeometry(QtCore.QRect(1020,330,141,15))
        self.label15.setObjectName("label14")

        self.label4 = QtWidgets.QLabel(self.centralwidget)
        self.label4.setGeometry(QtCore.QRect(1020,440,481, 261))
        self.label4.setMouseTracking(False)
        self.label4.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label4.setFrameShape(QtWidgets.QFrame.Box)
        self.label4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label4.setLineWidth(1)
        self.label4.setMidLineWidth(0)
        self.label4.setText("")
        self.label4.setObjectName("label1")

        self.label14 = QtWidgets.QLabel(self.centralwidget)
        self.label14.setGeometry(QtCore.QRect(1020,730,141,15))
        self.label14.setObjectName("label14")

        self.label6 = QtWidgets.QLabel(self.centralwidget)
        self.label6.setGeometry(QtCore.QRect(620,530,131,16))
        self.label6.setObjectName("label6")
        self.label7 = QtWidgets.QLabel(self.centralwidget)
        self.label7.setGeometry(QtCore.QRect(620,610,131,16))
        self.label7.setObjectName("label7")
        self.label8 = QtWidgets.QLabel(self.centralwidget)
        self.label8.setGeometry(QtCore.QRect(620,440, 131, 16))
        self.label8.setObjectName("label8")

        self.label9 = QtWidgets.QLabel(self.centralwidget)
        self.label9.setGeometry(QtCore.QRect(740,460,72,15))
        self.label9.setObjectName("label9")
        self.label10 = QtWidgets.QLabel(self.centralwidget)
        self.label10.setGeometry(QtCore.QRect(800,460,72,15))
        self.label10.setObjectName("label10")
        self.label12 = QtWidgets.QLabel(self.centralwidget)
        self.label12.setGeometry(QtCore.QRect(20,460, 161,16))
        self.label12.setObjectName("label11")


        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(740,440,72,22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider.setMinimum(90)
        # 设置最大值
        self.horizontalSlider.setMaximum(95)
        self.horizontalSlider.setValue(90)
        self.horizontalSlider.setTickPosition(QSlider.TicksBelow)
        self.horizontalSlider.setTickInterval(5)

        self.textEdit1 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit1.setGeometry(QtCore.QRect(720,510,251,51))
        self.textEdit1.setObjectName("textEdit1")

        self.textEdit2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit2.setGeometry(QtCore.QRect(720,600,251,51))
        self.textEdit2.setObjectName("textEdit2")

        self.textEdit3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit3.setGeometry(QtCore.QRect(150,440,251,51))
        self.textEdit3.setObjectName("textEdit2")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.pushButton_1.clicked.connect(MainWindow.yezhi)
        self.pushButton_2.clicked.connect(MainWindow.xiaobo)
        self.pushButton_3.clicked.connect(MainWindow.train11)
        self.pushButton_4.clicked.connect(MainWindow.xianshi)
        self.pushButton_5.clicked.connect(MainWindow.openfile)

        self.horizontalSlider.valueChanged[int].connect(MainWindow.changedvalue)
        self.pushButton.clicked.connect(MainWindow.btnshanbing_click)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_4.setText(_translate("MainWindow", "4.显示异常范围"))
        self.pushButton_3.setText(_translate("MainWindow", "3.开始预测"))
        self.pushButton_2.setText(_translate("MainWindow", "2.小波变换"))
        self.pushButton_1.setText(_translate("MainWindow", "1.野值去除"))
        self.label6.setText(_translate("MainWindow", "异常时间范围"))
        self.label7.setText(_translate("MainWindow", "异常偏离值"))
        self.label8.setText(_translate("MainWindow", "置信度调整"))
        self.label9.setText(_translate("MainWindow", "90"))
        self.label10.setText(_translate("MainWindow", "95"))
        self.label12.setText(_translate("MainWindow", "野值出现时刻"))
        self.label14.setText(_translate("MainWindow", "相关系数阈值法"))
        self.label15.setText(_translate("MainWindow", "置信区间估计法"))
        self.label16.setText(_translate("MainWindow", "参数设置"))
        self.label17.setText(_translate("MainWindow", "输入时间窗口长度"))
        self.label18.setText(_translate("MainWindow", "输出时间窗口长度"))
        self.label19.setText(_translate("MainWindow", "测试点"))
        self.pushButton.setText(_translate("MainWindow", "确定"))
