import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

a1=[7.267746,
9.769373,
1.648982,
0.65723956,
0.35539,
0.19131222,
0.09336452,
0.06441229,
0.054478236,
0.050516363,
0.04831108,
0.047184195,
0.0464161,
0.045766957,
0.045249004,
0.044801038,
0.044402365,
0.0440484,
0.043731093,
0.04344494,
0.043185767,
0.042949967,
0.042734593,
0.042537082,
0.042355224,
0.0421871,
0.04203113,
0.041885883,
0.041750137,
0.041622855,
0.041503176,
0.041390292,
0.04128352,
0.04118226,
0.041086018,
0.040994316,
0.040906772,
0.040822994,
0.040742666,
0.04066556,
0.040591378,
0.04051987,
0.040450856,
0.040384173,
0.040319584,
0.040257018,
0.040196273,
0.040137265,
0.040079843,
0.040023938,
0.039969426,
0.039916255,
0.0398643,
0.03981352,
0.03976381,
0.039715175,
0.039667495,
0.0396207,
0.039574802,
0.039529726,
0.039485384,
0.03944177,
0.039398868,
0.03935656,
0.039314877,
0.03927375,
0.039233126,
0.039192967,
0.039153278,
0.039113995,
0.039075103,
0.039036505,
0.038998265,
0.038960267,
0.03892253,
0.038884982,
0.038847614,
0.038810436,
0.038773317,
0.038736317,
0.038699366,
0.038662456,
0.03862556,
0.038588632,
0.038551677,
0.038515367,
0.038802706,
0.05182118,
0.04205362,
0.03753696,
0.050866686,
0.04358565,
0.041071195,
0.040058006,
0.03960313,
0.039360765,
0.039202485,
0.03907913,
0.03897105,
0.038871344,
0.03877857,
0.038692065,
0.03861053,
0.038532935,
0.038458772,
0.038387522,
0.03831872,
0.038252056,
0.038187217,
0.038123917,
0.038061947,
0.038001057,
0.037941054,
0.037881825,
0.03782318,
0.03776494,
0.03770704,
0.03764937,
0.037591763,
0.03753419,]


a2=[0.8753365,
0.09298632,
0.07640456,
0.054533236,
0.046265468,
0.044113886,
0.04349946,
0.042793952,
0.04222334,
0.04176272,
0.041348424,
0.040960543,
0.0405768,
0.040199894,
0.03982842,
0.03946587,
0.03911987,
0.03879521,
0.03848825,
0.03818613,
0.037869137,
0.037527896,
0.037176922,
0.036820196,
0.03643627,
0.036004093,
0.035578363,
0.035251904,
0.035018638,
0.03484076,
0.03464899,
0.034475937,
0.03440357,
0.034170434,
0.033993818,
0.033842806,
0.03377224,
0.03360025,
0.03338407,
0.033177942,
0.03295055,
0.03267061,
0.032385573,
0.03206325,
0.031634346,
0.03130103,
0.030927991,
0.031046936,
0.030541468,
0.030375391,
0.029890273,
0.029604623,
0.029319007,
0.029019782,
0.028707664,
0.030935703,
0.029325968,
0.028716749,
0.028283019,
0.027963107,
0.027660077,
0.027360225,
0.027087536,
0.026821814,
0.026555179,
0.026279025,
0.025992554,
0.025696652,
0.02541631,
0.027110815,
0.025764102,
0.02524737,
0.024837917,
0.024619618,
0.024420653,
0.02424791,
0.024080437,
0.023919657,
0.023795221,
0.024939997,
0.024147406,
0.023790114,
0.023478404,
0.02326245,
0.023157798,
0.023139216,
0.023284072,
0.023068048,
0.022878239,
0.022739148,
0.022642009,
0.02255826,
0.022476623,
0.022401305,
0.02236067,
0.022457939,
0.022562338,
0.02220122,
0.02209505,
0.022498127,
0.022278741,
0.022012377,
0.021817232,
0.022226848,
0.02256569,
0.021774905,
0.021626838,
0.021926979,
0.021914296,
0.021631945,
0.021418829,
0.021312067,
0.02120467,
0.021116365,
0.022178497,
0.022981707,
0.021464784,
0.021137455,
0.020901818,
0.02080403,]


a3=[2.9479651,
0.25477597,
0.11057569,
0.071856186,
0.051753107,
0.046705965,
0.045673806,
0.044922248,
0.044308007,
0.04378924,
0.043287426,
0.042841095,
0.04242742,
0.042051163,
0.04171329,
0.04141383,
0.04115218,
0.040925898,
0.04073077,
0.0405614,
0.040411852,
0.040276416,
0.040150333,
0.04003,
0.03991285,
0.0397971,
0.039681476,
0.039565112,
0.03944752,
0.03932865,
0.03920887,
0.039089072,
0.03897055,
0.038854897,
0.03874363,
0.038637787,
0.038537577,
0.038442243,
0.038350258,
0.038259763,
0.038168915,
0.038076147,
0.037980292,
0.037880495,
0.037776265,
0.037667442,
0.03755409,
0.03743633,
0.037314016,
0.03718635,
0.037051667,
0.036907557,
0.03675178,
0.036583826,
0.036406204,
0.03622374,
0.036043152,
0.03587692,
0.035733912,
0.035605606,
0.035484884,
0.03537175,
0.03526194,
0.035152126,
0.03504044,
0.034925714,
0.035076473,
0.03757237,
0.036254555,
0.035567142,
0.035058677,
0.034731917,
0.034570906,
0.03443201,
0.03429675,
0.03417147,
0.034049634,
0.03392814,
0.0338047,
0.03367602,
0.033538364,
0.033386618,
0.033212755,
0.033005375,
0.032763537,
0.032534223,
0.03235822,
0.03220656,
0.032058418,
0.03190803,
0.03175191,
0.03158606,
0.031406414,
0.031208215,
0.03098906,
0.031736825,
0.037196554,
0.032596167,
0.030903446,
0.03078975,
0.030509414,
0.030269152,
0.030092204,
0.029940542,
0.029802065,
0.029670212,
0.029542195,
0.029416269,
0.029290393,
0.029162442,
0.029030401,
0.02889258,
0.028747754,
0.02859526,
0.028800366,
0.03409806,
0.030797126,
0.02924031,
0.028674282,
0.028343715,]

plt.figure()
plt.plot(list(range(len(a1))),a1,label='RNN')
plt.plot(list(range(len(a2))),a2,label='LSTM',linestyle='--')
plt.plot(list(range(len(a3))),a3,label='GRU',linestyle='-.')
plt.title('损失')
plt.xlabel('迭代次数')
plt.ylabel('训练损失')
plt.legend()
plt.ylim(0,0.3)
plt.show()