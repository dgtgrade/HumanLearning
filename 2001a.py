# 공부를 위해서 작성한 거라 일부러 ML 라이브러리는 사용하지 않았다.
# mnist 로딩하는 라이브러리도 직접 작성 하였다.
# numpy, matplotlib 만 사용 하였다.

# python 3.5 환경에서 작성 하였다
#
# 실험용으로 대충 빨리 만들어본 코드라서 깔끔하지 않다.
#
# 그림 그리는 부분 코드가 인공지능 코드 보다 더 많다.
# 인공지능 부분만 보려면 윗부분에 모여있는 그림 그리기 관련 코드를 
# 건너뛰고 그 아래쪽 부분만 보면 된다.
#
# Color Palette from Color Schemer Online v2
# http://www.colorschemer.com/online.html
#
# 작성자:
# Facebook: https://facebook.com/dgtgrade
# Youtube: https://youtube.com/channel/UCEdT99nAs8nalv6Mafs9RiA 
# email: dgtgrade@gmail.com
#
import numpy as np
import time
import math

# 휴먼러닝 라이브러리
from mnist2ndarray import *

# 정해져 있는 값들
IMGSIZE=28
CMAX=255

# 설정값들
DO_PLOT = True # 그래프로 표시할지 여부

# numpy 어레이 데이터 출력 형태 설정
float_formatter = lambda x: "%9.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter},
    linewidth=120, threshold=np.inf)

##############################
# 그림 그리기 관련 코드 모음
##############################
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# 통계치에 대한 그래프 설정
ax_es_opts = {
    "mean": {
        "gpos": 2,
        "gposi": np.s_[5:7],
        "size": (IMGSIZE,IMGSIZE),
        "cmap": "gray", 
        "vmin": 0.0, 
        "vmax": 1.0}, 
    "std": {
        "gpos": 3,
        "gposi": np.s_[7:9],
        "size": (IMGSIZE,IMGSIZE),
        "cmap": "afmhot", 
        "vmin": 0.0, 
        "vmax": 1.0}, 
    "cov": {
        "gpos": 4,
        "gposi": np.s_[9:11],
        "size": (IMGSIZE*IMGSIZE,IMGSIZE*IMGSIZE),
        "cmap": "br1",
        "vmin": -0.1, 
        "vmax": 0.1}} 

# 커스텀 칼라
cmap_br1 = {'red':  ((0.0, 1.0, 1.0),
                    (0.5, 0.0, 0.0),
                    (1.0, 0.2, 0.2)),
         'green':   ((0.0, 0.2, 0.2),
                    (0.5, 0.0, 0.0),
                    (1.0, 0.8, 0.8)),
         'blue':    ((0.0, 0.25, 0.25),
                    (0.5, 0.0, 0.0),
                    (1.0, 1.0, 1.0))}

br1 = LinearSegmentedColormap('br1', cmap_br1)
plt.register_cmap(cmap=br1)

# 설정값들
PLT_PAUSETIME=0.1

# 그림 초기화
#
def fig_init():

    # bar, eimage, label, evidence
    global fig, gs, ax_b, ax_i, ax_l, ax_es, ax_es_i, n_labels
    global f_im, f_la, f_es, f_es_i

    fig = plt.figure(figsize=(32, 18))
    fig.patch.set_facecolor('#000000')

    gs = gridspec.GridSpec(5, 1+n_labels, height_ratios=[0.3,3,1,1,1])
    gs.update(wspace=0.05, hspace=0.02, 
        left=0.1, right=0.9, bottom=0.1, top=0.9)

    # 텍스트들
    for i, desc in enumerate(("trained", "training", 
        "mean", "standard\ndeviation", "covariance")):
        ax = plt.subplot(gs[i,0])
        ax.set_axis_off()
        ax.text(0.8, 0.5, desc,
            verticalalignment='center', horizontalalignment='right',
            transform=ax.transAxes, fontsize=30, color="#999999")

    ax = plt.subplot(gs[1,4])
    ax.set_axis_off()
    ax.text(0.5, 0.5, ">",
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, fontsize=30, color="#999999")

    # 프로그레스바
    ax_b = plt.subplot(gs[0,1:])
    ax_b.set_axis_off()


    # 학습중인 이미지
    ax_i = plt.subplot(gs[1,2:4])
    ax_i.set_axis_off()
    f_im = ax_i.imshow(np.zeros((IMGSIZE,IMGSIZE)),
        interpolation='none', cmap="gray", vmin=0, vmax=1.0)

    # 학습중인 라벨
    ax_l = plt.subplot(gs[1,1])
    ax_l.set_axis_off()
    f_la = None

    # 현재까지의 평균, 표준편차, 공분산들
    ax_es_i = {}
    f_es_i = {}
    for n in ax_es_opts:
        ax_es_i[n] = plt.subplot(gs[1,ax_es_opts[n]['gposi']])
        f_es_i[n] = ax_es_i[n].imshow(np.zeros(ax_es_opts[n]['size']),
            interpolation='none', cmap=ax_es_opts[n]['cmap'], 
            vmin=ax_es_opts[n]['vmin'], vmax=ax_es_opts[n]['vmax'])
        ax_es_i[n].set_axis_off()

    ax_es = {}
    f_es = {}
    for i in range(n_labels):
        ax_es[i] = {}
        f_es[i] = {}
        for n in ax_es_opts:
            ax_es[i][n] = plt.subplot(gs[ax_es_opts[n]['gpos'],i+1])
            f_es[i][n] = ax_es[i][n].imshow(np.zeros(ax_es_opts[n]['size']),
                interpolation='none', cmap=ax_es_opts[n]['cmap'], 
                vmin=ax_es_opts[n]['vmin'], vmax=ax_es_opts[n]['vmax'])

            ax_es[i][n].set_axis_off()

    plt.show(block=False)
    plt.pause(PLT_PAUSETIME)


# 학습 중 그림
#
def fig_learn(trained, img, label, labels=[]):

    global f_im, f_la

    if (len(labels)==0):
        labels = [label]

    # 프로그레스
    ax_b.barh(0, trained, facecolor="#33ccff",align="center")
    ax_b.set_xlim((0,m))
    ax_b.set_ylim((0,0))

    # 현재 이미지
    if (label != None):
        if (f_im != None): f_im.remove()
        img_2d = img.reshape(IMGSIZE,IMGSIZE)
        f_im = ax_i.imshow(img_2d, interpolation='none',
                cmap="gray", vmin=0, vmax=1.0)

        if (f_la != None): f_la.remove()
        f_la = ax_l.text(0.5, 0.5, "%s" % label,
                verticalalignment='center', horizontalalignment='center',
                transform=ax_l.transAxes, fontsize=50, color="white")

    # 현재 이미지 + 전체 이미지
    img_es = {}
    for n in ax_es_opts:

        if (label != None):
            img_es[n] = EoH[label][n].reshape(ax_es_opts[n]['size'])
            f_es_i[n].remove()
            f_es_i[n] = ax_es_i[n].imshow(img_es[n], 
                interpolation='none', cmap=ax_es_opts[n]['cmap'], 
                vmin=ax_es_opts[n]['vmin'], vmax=ax_es_opts[n]['vmax'])

        for l in labels:
            if (l in EoH):
                img_es[n] = EoH[l][n].reshape(ax_es_opts[n]['size'])
                f_es[l][n].remove()
                f_es[l][n] = ax_es[l][n].imshow(img_es[n], 
                    interpolation='none', cmap=ax_es_opts[n]['cmap'], 
                    vmin=ax_es_opts[n]['vmin'], vmax=ax_es_opts[n]['vmax'])


    plt.show(block=False)
    plt.pause(PLT_PAUSETIME)

# 테스트 그림 초기화
#
def fig_init_test():

    global fig, gs, ax_b, ax_i, ax_l, ax_a
    global f_im, f_la, f_an

    # 텍스트들
    ax = plt.subplot(gs[0:2,:])
    ax.cla()

    ax = plt.subplot(gs[0,0])
    ax.set_axis_off()
    ax.text(0.8, 0.5, "tested",
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes, fontsize=30, color="#999999")

    ax = plt.subplot(gs[1,0])
    ax.set_axis_off()
    ax.text(0.8, 0.5, "testing",
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes, fontsize=30, color="#999999")

    ax = plt.subplot(gs[1,4])
    ax.set_axis_off()
    ax.text(0.5, 0.5, ">",
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, fontsize=30, color="#999999")

    # 프로그레스바
    ax_b = plt.subplot(gs[0,1:])
    ax_b.set_axis_off()

    # 테스트중인 이미지
    ax_i = plt.subplot(gs[1,2:4])
    ax_i.set_axis_off()
    f_im = ax_i.imshow(np.zeros((IMGSIZE,IMGSIZE)),
        interpolation='none', cmap="gray", vmin=0, vmax=1.0)

    # 테스트중인 라벨
    ax_l = plt.subplot(gs[1,1])
    ax_l.set_axis_off()
    f_la = None

    # 답
    ax_a = {}
    f_an = {}
    for i in (0,1,2):
        ax_a[i] = plt.subplot(gs[1,5+i*2:7+i*2])
        f_an[i] = None
        ax_a[i].set_axis_off()


    plt.show(block=False)
    plt.pause(PLT_PAUSETIME)


# 테스트 중 그림
#
COLOR_SUCCESS = "#66ff33"
COLOR_FAIL = "#ff3366"
def fig_test(success, fail, img=None, label=None, ds=[]):

    global fig, gs, ax_b, ax_i, ax_l, ax_a
    global f_im, f_la, f_an
    global m_test

    ax_b.barh(0,success,facecolor=COLOR_SUCCESS,align="center")
    ax_b.barh(0,fail,left=success,facecolor=COLOR_FAIL,align="center")
    ax_b.set_xlim((0,m_test))
    ax_b.set_ylim((0,0))

    if (label != None):
        if (f_im != None): f_im.remove()
        img_2d = img.reshape(IMGSIZE,IMGSIZE)
        f_im = ax_i.imshow(img_2d, interpolation='none',
                cmap="gray", vmin=0, vmax=1.0)

        if (f_la != None): f_la.remove()
        f_la = ax_l.text(0.5, 0.5, "%s" % label,
                verticalalignment='center', horizontalalignment='center',
                transform=ax_l.transAxes, fontsize=50, color="white")

    for i, a in enumerate(ds[0:3]):
        if (i == 0):
            c = COLOR_SUCCESS if a == label else COLOR_FAIL
            fs = 150
        elif (a == label):
            c = "white"
            fs = 150
        else:
            c = "#999999"
            fs = 50

        if (f_an[i] != None): f_an[i].remove()
        f_an[i] = ax_a[i].text(0.5, 0.5, "%s" % a,
                verticalalignment='center', horizontalalignment='center',
                transform=ax_a[i].transAxes, fontsize=fs, color=c)

        
    plt.show(block=False)
    plt.pause(PLT_PAUSETIME)

# Naive Bayes: P(H|E) = P(E|H)/P(E) * P(H)
# 
#   P(H|E):         posterior probability
#       구하려는 값
#
#   P(E|H):         likelihood
#       multivariate normal distribution
#       P(E|H) = N(E, EoH_mean,EoH_cov)
#
#   P(H):           prior probability
#       스무딩 적용하면 다음과 같이 됨
#       (count(H)+1)/(count(all H)+1)
#
#   P(E):           evidence
#       특정 E에 대해서 argmax P(H|E)를 구할때 P(E)는 고정값이므로 필요 없음
#
EoH = dict()
Using_Features = np.ones(IMGSIZE*IMGSIZE, dtype=bool)

def update_EoH(labels=[]):
    t = 0

    if (len(labels)==0):
        labels = LABELS

    for d in labels:
        d_imgs = train_images[np.all([train_labels==d, trained], axis=0)]

        if (len(d_imgs)==0):
            continue

        if (len(d_imgs)>1):
            cov = np.cov(d_imgs[...,Using_Features].T)
        else:
            cov = np.zeros((IMGSIZE**2,IMGSIZE**2))

        EoH[d] = {
            "mean" : np.mean(d_imgs[...,Using_Features],axis=0),
            "std" : np.std(d_imgs[...,Using_Features],axis=0),
            "cov" : cov
        }


##############################
# 학습
##############################
#
train_images_2d = mnist2ndarray("data/train-images-idx3-ubyte")/CMAX
train_images = train_images_2d.reshape(len(train_images_2d),-1)
train_labels = mnist2ndarray("data/train-labels-idx1-ubyte")
trained = np.zeros(len(train_images), dtype=bool)

LABELS = np.unique(train_labels)
n_labels = len(LABELS)
assert len(train_images) == len(train_labels)
# 학습 데이터 수 
m = len(train_images)
m = int(m/1) # 디버깅시에 일부만 가지고 하면 편함

if DO_PLOT: fig_init()

FIG_LEARN_BOOTSTRAP=100
FIG_LEARN_INTERVAL=100
for i in range(m):
    trained[i] = True

    if (DO_PLOT and
        ((i < FIG_LEARN_BOOTSTRAP) or (i%FIG_LEARN_INTERVAL==0))):
            update_EoH()
            fig_learn(i, train_images[i], train_labels[i], LABELS)

l, C_H = np.unique(train_labels, return_counts=True)
P_H = (C_H+1) / (sum(C_H)+1)

update_EoH()
if DO_PLOT: fig_learn(m, None, None, LABELS)

##############################
# 피쳐 버리기
##############################
#
# 특정 H의 cov의 determinant가 0이 되어버리면,
# 해당 H의 P(H|E) 확률이 무한대로 올라가게 된다.
# 따라서 모든 H의 cov의 determinant가 절대 0이 되지 않도록 노력해야 한다.
# 몇가지 방법들이 있는 것 같은데 아직 공부를 안 해서...
# 일단 생각나는대로 해본다.
#
# 1. cov의 row중 row전체가 0 또는 0에 가까운 것이 있으면
#       그 row에 해당하는 feature는 버린다.
#       cov.T = cov이므로 column에 대해서는 할 필요 없다. 
# 
# 그런데 1에서 문제가 되는 row 들의 경우에 대부분 분산 이전에
#   값부터가 0이어서 문제인 것 같다.
#   만약 값들은 0이 아닌데 분산만 0에 가까운 경우라면,
#   determinant를 못구한다는 것 외에도
#   그 row가 주는 의미로 볼때, 버려도 되는 (도움이 안 되는) 것이 맞는걸까?
#
for d in LABELS:
    Min_CovRow = np.median(abs(EoH[d]["cov"][abs(EoH[d]["cov"])>0.0]))
    for i, row in enumerate(EoH[d]["cov"]):
        if (np.median(abs(row))<Min_CovRow/50):
            Using_Features[i] = False

# 버려진 피쳐들 제외 하고 새로 만들기
print ("Using_Features:",sum(Using_Features))
update_EoH()

# 2. 라벨과 관계 없이 전체 데이터에서 각 피처들의 분산을 구하고,
#   이 분산이 0에 가까운 녀석들은 버려버릴 수 있겠다.
#   이렇게 하면 cov이 좀 더 invertible 한 것에 가까워질 수 있겠다.
#   PCA가 이것과 유사한데 PCA는 피쳐들을 조합도 하는 것 같고,
#   이것은 조합 없이 그냥 피쳐들을 버리는 것이다.
#   
#   결과: 1 없이 이것만 해서는 잘 안 되는 것 같다.
#   추가: PCA를 사용하지 않은 이유는 조합해 버리면 그 조합 결과를 
#       시각적으로 확인하기 힘들 것 같아서 그랬는데, reconstruct 
#       하면 될 듯 하다.
#       3에서는 PCA를 구현해 봐야 하겠다.

#F_STD = np.std(train_images,axis=0)
#for i, var in enumerate(F_STD):
#    if (var < 0.2):
#        Using_Features[i] = False
#
#print ("Using_Features:",sum(Using_Features))
#EoH = make_EoH()

##############################
# 테스트
##############################
test_images_2d = mnist2ndarray("data/t10k-images-idx3-ubyte")/CMAX
test_images = test_images_2d.reshape(len(test_images_2d),-1)
test_labels = mnist2ndarray("data/t10k-labels-idx1-ubyte")
assert len(test_images) == len(test_labels)

# m = 테스트 데이터 수 
m_test = len(test_images)
m_test = int(m_test/100) # 디버깅시에 일부만 가지고 하면 편함

# 테스트 결과 맞는 경우
test_results = np.zeros(m_test, dtype=bool)
# 테스트 결과 (첫번째 후보는 아니고) 두번째 후보가 맞는 경우
test_results_2nd = np.zeros(m_test, dtype=bool)

# multivariate normal distribution
# P(E|H) = N(E, EoH_mean,EoH_cov)
def logN(img,mean,cov):

    const = 0.0 # 빠른 계산을 위해서 0으로 처리하는 부분

    detsign,logdet = np.linalg.slogdet(cov)
#    assert logdet > 0

    ln = -1/2*(logdet+
        np.dot((img-mean).T,np.linalg.inv(cov)).dot(img-mean)+const)

    return ln 


if DO_PLOT: fig_init_test()
    
FIG_TEST_BOOTSTRAP=100
FIG_TEST_INTERVAL=10

for i in range(m_test):

    P_HoE = np.zeros(n_labels)
    img_org = test_images[i]
    img = test_images[i,Using_Features]
    l = test_labels[i]

    # P(H|E) = P(E|H)/P(E) * P(H)
    # Bayes' Rule for Multiple Variables
    # P(H|E1,..,En) = P(E1,..,En|H)/P(E1,.,En)*P(H)
    #   E1,E2,...En이 완전히 독립적이라면:
    #       P(E1,..,En|H) = P(E1|H)*P(E2|H)*..*P(En|H)
    #       P(E1,..,En) = P(H)*P(E1|H)*..*P(En|H)+P(^H)*P(E1|^H)...*P(En|^H)
    #
    for j in LABELS:
        P_HoE[j] = (np.log(P_H[j]) + 
                (logN(img, EoH[j]["mean"], EoH[j]["cov"])))
            
    ds = np.argsort(P_HoE)[-5:][::-1]
    d = ds[0] # 결과

    if (d==l):
        test_results[i] = True
    
    successes = sum(1*test_results)
    tests = (i+1)

    print ("%d / %d (%3.1f)" % (successes, tests, successes/tests*100.0))
    
    if (DO_PLOT and 
        ((i < FIG_TEST_BOOTSTRAP) or 
        (i%FIG_TEST_INTERVAL==0) or i==m_test-1)): #마지막
        fig_test(successes, tests-successes, img_org, l, ds)

time.sleep(5)
