import numpy as np
import pywt
import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-

import scipy.io as scio

origin=[35.22770024, 106.33075265, 156.97401559, 140.10927066,
        135.47636366, 114.80200457, 153.04006305,  90.64669945,
        109.39685005,  90.34224163, 122.19051108, 117.50439793,
         74.55647211, 125.66662794, 116.85663118, 111.66552962,
        155.24522974, 152.96968027, 183.94269962, 200.93894616,
        164.30901929, 152.27990827, 137.90374966, 180.9713031 ,
        161.65512055, 199.42814403, 178.79129695, 161.32457064,
        231.5467129 , 160.16009005, 198.90007639, 231.19534779,
        189.24330564, 211.1726706 , 182.19425388, 176.20496011,
        143.78055116, 199.83436341, 144.60825942, 189.55661365,
        211.29535811, 225.45645944, 260.78870075, 313.3606688 ,
        241.3376357 , 200.34211063, 194.45815272, 208.89198745,
        149.98467738, 181.42560731, 278.86407387, 233.47455281,
        249.38481471, 207.77249739, 266.25808598, 293.01021822,
        189.34697658, 223.16574212, 173.63548735, 168.11628319,
        202.68633839, 121.62934272, 160.52081311, 185.80768342,
        199.33137228, 222.94386855, 241.61002737, 303.08798314,
        203.20356979, 232.85765972, 230.54733367, 228.92206248,
        245.2588066 , 219.76450302, 215.10633611, 267.46634951,
        303.62888241, 375.83857264, 348.94278152, 290.04370422,
        317.82216224, 287.16930687, 229.05673374, 237.78806986,
        354.85164098, 194.41667712, 299.1509784 , 266.74310432,
        295.73551126, 385.12479967, 417.83167911, 380.12427238,
        382.19512344, 327.78378735, 293.65833096, 291.10289114,
        351.70032662, 324.12474802, 443.80703656, 385.22108737,
        395.49225787, 423.85425896, 419.9064899 , 454.00725634,
        377.47728292, 327.2610879 , 324.4298496 , 316.51276776,
        308.67104026, 303.67352974, 356.36896316, 362.32670507,
        329.6577401 , 446.10611153, 493.73070283, 551.35349072,
        396.53903298, 354.29324744, 269.00254637, 288.51580609,
        370.86779115, 333.26788274, 423.63104195, 364.75685088,
        486.79793156, 476.27807452, 584.64616434, 551.05779302,
        457.31243205, 431.15181643, 381.451487  , 365.79107495,
        363.35574451, 352.57462807, 369.44793375, 463.0849706 ,
        457.60621761, 522.11962434, 574.36629047, 633.28486001,
        523.3793889 , 457.86321307, 425.74130413, 401.41071872,]


wavefunc = 'db4'

lv = 4
m = 4
n = 4
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
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
coeffs = pywt.wavedec(data, wavefunc, mode='sym', level=lv)  # 按 level 层分解，使用pywt包进行计算， cAn是尺度系数 cDn为小波系数



ya4 = pywt.waverec(np.multiply(coeffs, [1, 0, 0, 0, 0]).tolist(), 'db4') #低频
yd4 = pywt.waverec(np.multiply(coeffs, [0, 1, 0, 0, 0]).tolist(), 'db4')  #高频
yd3 = pywt.waverec(np.multiply(coeffs, [0, 0, 1, 0, 0]).tolist(), 'db4')
yd2 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 1, 0]).tolist(), 'db4')
yd1 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 1]).tolist(), 'db4')
x=range(len(origin))
y=origin

plt.figure()
plt.plot(x,y)
plt.xlabel('时间')
plt.ylabel('数据值')
plt.title('原始数据')
plt.figure(figsize=(12, 12))
plt.subplot(511)
plt.plot(x, ya4)
plt.title('第四级近似分量')
plt.subplot(512)
plt.plot(x, yd4)
plt.title('第四级细节分量')
plt.subplot(513)
plt.plot(x, yd3)
plt.title('第三级细节分量')
plt.subplot(514)
plt.plot(x, yd2)
plt.title('第二级细节分量')
plt.subplot(515)
plt.plot(x, yd1)
plt.title('第一级细节分量')
plt.tight_layout()
plt.show()
#


sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0  # sgn函数
print(1)
# 去噪过程
for i in range(m, n + 1):  # 选取小波系数层数为 m~n层，尺度系数不需要处理
    cD = coeffs[i]
    for j in range(len(cD)):
        Tr = np.sqrt(2 * np.log(len(cD)))  # 计算阈值
        if cD[j] >= Tr:
            coeffs[i][j] = sgn(cD[j]) - Tr  # 向零收缩
        else:
            coeffs[i][j] = 0  # 低于阈值置零

    # 重构
print(2)
print(len(origin))
datarec1 = pywt.waverec(coeffs, wavefunc)
print(3)
plt.figure()
plt.subplot(2,1,1)
plt.plot(range(len(origin)), origin)
plt.xlabel('时间')
plt.ylabel('数据值')
plt.title("原始信号")
plt.subplot(2, 1, 2)
plt.plot(range(len(datarec1)), datarec1)
plt.xlabel('时间')
plt.ylabel('数据值')
plt.title("降噪信号")
plt.show()