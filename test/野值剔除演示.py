from scipy import interpolate
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
x = [10.0247, 10.0470, 10.0647, 10.0761, 15.08, 10.0761, 10.0647, 10.047, 10.0247, 10, 9.9753, 9.956, 9.9353, 9.9239,
     18.92, 9.9239, 9.9353, 9.953, 9.9753, 10]
plt.figure()
plt.plot(list(range(len(x))), x)
plt.title('原始数据')
plt.xlabel('时间')
plt.ylabel('数据值')
plt.show()

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
    print('cubic 插值填充')
    print(value_new)
    value.insert(label_y_y[i], value_new)

plt.figure()
plt.plot(list(range(len(value))), value)  # 去除野值后的序列
plt.xlabel('时间')
plt.ylabel('数据值')
plt.title('空值填充后的数据')
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
plt.xlabel('时间')
plt.ylabel('数据值')
plt.title('平滑后的数据')
plt.show()



