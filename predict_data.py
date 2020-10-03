from PyQt5 import QtGui
from pandas import DataFrame, np
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array
import matplotlib.pyplot as plt
from keras.models import load_model

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')


# convert time series into supervised learning problem 从数字列表转到输入和输出模式列表
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):  # [d0 d1 ... d99   d100 d101 d102]把数据集整个都化成该形式
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# create a differenced series
# 我们可以引入一个名为difference()的函数使数据平稳。这将把一系列的值转换成一系列的差异，这是一种更简单的表示方法。
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# transform series into train and test sets for supervised learning
# 首先对数据进行差异处理并重新调整数据，然后将其转换为监督学习问题并训练测试集，
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:-n_test + 1]  # 取数据集中的某一点作为测试点
    return scaler, train, test


# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        print('step:%d' % i)
        model.reset_states()
    model.save('my_model_some.h5')
    return model


# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]


# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)

    return forecasts


# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    return inverted


# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted


# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_seq):
    global time, value
    mse = []
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        mse.append(rmse)
        print('t+%d RMSE: %f' % ((i + 1), rmse))




    # beta = 0.8
    # threshold=[]
    # la=3
    # for i in range(len(mse)-1):#平滑处理
    #     mse[i+1]=mse[i]*beta+(1-beta)*mse[i+1]
    #
    # lb=5
    # i=0
    # z=0.2
    #
    # while i*la+lb<len(mse):
    #     #lb>la
    #     t = np.mean(mse[i * la:(i + 1) * la], dtype=float) + z * np.std(mse[i * la:(i + 1) * la], dtype=float)
    #     for j in range(i*la,(i+1)*la):
    #         threshold.append(t)
    #     i=i+1
    # t = np.mean(mse[i * la:(i+1)*la],dtype=float) + z * np.std(mse[i * la:(i+1)*la],dtype=float)
    # for j in range(i*la,len(mse)):
    #     threshold.append(t)
    #
    #
    #
    threshold=[]
    for i in range(len(mse)):
        threshold.append(62)
    plt.figure()

    plt.plot(list(range(len(series.values) - n_test, len(series.values) - n_test + len(threshold))),threshold,label='threshold')
    plt.plot(list(range(len(series.values) - n_test, len(series.values) - n_test + len(mse))),mse,label='mse')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('error value')
    filename = '4.png'

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

    # for i in range(len(mse)):
    #     if mse[i]>threshold[i]:
    #         break
    # print(i+len(series.values) - n_test)


    #
    # for i in range(len(forecasts[0])):
    #     # if  shangxian[i]<series.values[i+len(series.values) - n_test] :
    #     #     time = i + len(series.values) - n_test
    #     #     value = abs(shangxian[i]-(series.values[i+len(series.values) - n_test]))
    #     #     print("遥测数据出现异常状态的时间发生在：%d"%(i + len(series.values) - n_test))
    #     #     print("异常值偏离正常值%.5f"%(abs(shangxian[i]-(series.values[i+len(series.values) - n_test]))))
    #     #     break
    #     if xiaxian[i] > series.values[i + len(series.values) - n_test+1]:
    #         time = i + len(series.values) - n_test+1
    #         value = abs(xiaxian[i] - (series.values[i + len(series.values) - n_test+1]))
    #         print("遥测数据出现异常状态的时间发生在：%d"%(i + len(series.values) - n_test+1))
    #         print("异常值偏离正常值%.5f"%(abs(xiaxian[i]-(series.values[i+len(series.values) - n_test]+1))))
    #         break
    #
    return 0


# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    plt.figure()
    plt.plot(series.values,label='实际值')
    # forecasts[0][1] = 360
    # forecasts[0][2] = 330

    xiaxian = []
    shangxian = []
    zhi = 5



    n=len(forecasts[0])
    sigma=np.std(forecasts[0],ddof=1)

    zsigman=zhi*float(sigma)/sqrt(n)
    print('******')
    print(zsigman)
    print('******')
    for i in range(len(forecasts[0])):
        # xiaxian.append(forecasts[0][i] - zhi)
        # shangxian.append(forecasts[0][i] + zhi)
        shangxian.append(forecasts[0][i]+zsigman)
        xiaxian.append(forecasts[0][i]-zsigman)



    # plot the forecasts in red
    off_s = len(series) - n_test - 1
    off_e = off_s + len(forecasts[0]) + 1
    xaxis = [x for x in range(off_s, off_e)]
    yaxis = [series.values[off_s]] + forecasts[0]
    plt.plot(xaxis, yaxis, color='red',label='预测值')

    xaxis1 = [x-1 for x in range(off_s, off_e)]
    yaxis1 = [series.values[off_s]] + shangxian
    plt.plot(xaxis1, yaxis1, color='y',linestyle='--',label='上限')

    xaxis2 = [x+1 for x in range(off_s, off_e)]
    yaxis2 = [series.values[off_s]] + xiaxian
    plt.plot(xaxis2, yaxis2, color='g',linestyle='--',label='下限')

    plt.xlabel('时间')
    plt.ylabel('数据值')
    plt.legend()
    # show the plot
  #  plt.show()

    # filename = 'yu.png'
    # if filename is not None:
    #     plt.savefig(filename)
    # else:
    #     plt.show()

    return shangxian,xiaxian

#
series = read_csv('xiaoboo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# # configure
n_lag = 90
n_seq = 25
n_test = 24  # 取测试点  24
n_epochs = 1500
n_batch = 1
n_neurons = 2
# # prepare data
# scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
# # fit model
# # model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# # make forecasts
# model = load_model('my_model.h5')
# forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
# # inverse transform forecasts and test
# forecasts = inverse_transform(series, forecasts, scaler, n_test + 2)
# actual = [row[n_lag:] for row in test]
# actual = inverse_transform(series, actual, scaler, n_test + 2)
# evaluate forecasts

# plot forecasts
#shangxian,xiaxian=plot_forecasts(series, forecasts, n_test + 2)
# pyplot.plot(list(range(len(forecasts[0]))),forecasts[0]+10,color='b')



#time,value = evaluate_forecasts(actual, forecasts, n_lag, n_seq,shangxian,xiaxian)
