from scipy import interpolate
import matplotlib.pyplot as plt

import numpy as np
def preprocess1(data):
    x=data
    out = []
    # 可信度计算
    for i in range(len(x) - 2):
        i = i + 1
        if abs(x[i] - x[i - 1]) < abs(x[i] - x[i + 1]):
            out.append(abs(x[i] - x[i - 1]))
        else:
            out.append(abs(x[i] - x[i + 1]))
            # out18 个数除去首尾

    c1 = 0.02
    c2 = 0.04
    c3 = 0.06
    m = []
    # 模糊化
    u1 = []
    u2 = []
    u3 = []
    u4 = []
    for r in range(len(out)):
        if 0 <= out[r] <= c1:
            u1.append(1 - out[r] / c1)
            u2.append(out[r] / c1)
            u3.append(0)
            u4.append(0)
            m.append([out[r], u1[r], u2[r], u3[r], u4[r]])
        if c1 < out[r] <= c2:
            u1.append(0)
            u2.append((c2 - out[r]) / c1)
            u3.append((out[r] - c1) / c1)
            u4.append(0)
            m.append([out[r], u1[r], u2[r], u3[r], u4[r]])
        if c2 < out[r] <= c3:
            u4.append((out[r] - c2) / c1)
            u3.append((c3 - out[r]) / c1)
            u2.append(0)
            u1.append(0)
            m.append([out[r], u1[r], u2[r], u3[r], u4[r]])
        if out[r] > c3:
            u4.append(1)
            u3.append(0)
            u2.append(0)
            u1.append(0)
            m.append([out[r], u1[r], u2[r], u3[r], u4[r]])

    d1 = 0
    d2 = 0.33
    d3 = 0.66
    d4 = 1
    reliab = []  # 可信度

    value = []
    label_y = []
    value.append(x[0])
    label_y.append(0)
    thresh = 0.5
    label_y_y = []
    # 模糊可信度判断,计算量化的可信度
    for r in range(len(out)):
        z = d1 * u4[r] + d2 * u3[r] + d3 * u2[r] + d4 * u1[r]
        reliab.append([out[r], z])
        if z < thresh:
            # print(r+1)
            print('野值为')
            print(x[r + 1])
            # value.append(x[r+2])
            label_y_y.append(r + 1)

        #   x[r + 1] = x[r]#将野值用前后值代替
        # a=r+1则为x序列中x[a]野值
        else:
            value.append(x[r + 1])
            label_y.append(r + 1)

    value.append(x[len(x) - 1])  # 去除野值后的值，包括首尾元素,空值用前后值来代替
    label_y.append(label_y[-1] + 1)

    # 空值填充

    f = interpolate.interp1d(label_y, value, kind="cubic")
    for i in range(len(label_y_y)):
        value_new = f(label_y_y[i])
        print(value_new)
        value.insert(label_y_y[i], value_new)

    plt.figure()
    plt.plot(list(range(len(value))), value)  # 去除野值后的序列
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title('The data after the null value is filled')
    plt.show()

    # 平滑
    a = list(range(len(value)))
    f = interpolate.interp1d(a, value, kind="cubic")
    b = []
    for i in range(len(value)):
        value_new = f(a[i])
        b.append(value_new)

    plt.figure()
    plt.plot(list(range(len(b))), b)  # 平滑
    plt.xlabel('time')
    plt.ylabel('value')
    plt.title('Smooth data')
    plt.show()


































from pandas import read_csv

def preprocessl(data):
    # seq = np.array(
    #         [112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104.,
    #          118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133.,
    #          114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184.,
    #          162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242.,
    #          209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264.,
    #          272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264.,
    #          302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270.,
    #          315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313.,
    #          318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356.,
    #          348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318.,
    #          362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360.,
    #          342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
    #          417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390.,
    #          432.], dtype=np.float32)
    # seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    # return seq
    s=[data[0]]
    for i in range(1,len(data)-1):
        k=(data[i-1]+data[i+1])/2
        if data[i]>1.6*k :
            s.append(k)
            continue
        if data[i]<0.42*np.mean(data):
            s.append(k)
            continue
        s.append(data[i])
    s.append(data[-1])


    #dataframe = read_csv('errordata.csv', usecols=[1], engine='python', skipfooter=3)
    return s