import matplotlib.pyplot as plt
from mnist2ndarray import *
import time
import math

# 정해져 있는 값들
IMGSIZE=28
#IMGSIZE=20
CMAX=255

# 설정값들
PLT_PAUSETIME=1
DO_PLOT = False # 그래프로 표시할지 여부
NP_LINEWIDTH = IMGSIZE*8+10 # 콘솔 한줄의 길이 설정

# numpy 어레이 데이터 출력 형태 설정
float_formatter = lambda x: "%9.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter},
    linewidth=NP_LINEWIDTH, threshold=np.inf)

def fig_init():

    if not DO_PLOT: return

    global fig, ax_i, ax_c, ax_n
    
    fig = plt.figure()
    fig.patch.set_facecolor('#aaaaaa')

    ax_i = plt.subplot2grid((2,2), (0,0))
    ax_c = plt.subplot2grid((2,2), (0,1))
    ax_n = plt.subplot2grid((2,2), (1,0))

    ax_i.set_position([0.1,0.2,0.7,0.7])
    ax_c.set_position([0.73,0.2,0.03,0.7])
    ax_n.set_position([0.05,0.05,0.8,0.1])

def fig_draw(i):

    if not DO_PLOT: return

    ax_i.cla()
    for ax in [ax_i.xaxis, ax_i.yaxis]:
        ax.set_ticks(np.arange(IMGSIZE))
        ax.set_ticks(np.arange(IMGSIZE)+0.5, minor=True)
        ax.set_ticklabels([])

    ax_i.grid(True, which='minor')
    # 그래프의 픽셀칼라값 = 255-픽셀데이터값
    # 즉, 픽셀데이터값이 255이면 픽셀칼라값을 0으로 해서 검은색이 되게 한다.
    im = ax_i.imshow(train_images[i], interpolation='none',
            cmap="gray_r", vmin=0, vmax=CMAX)

    ax_c.cla()
    cb = fig.colorbar(im, cax=ax_c, orientation='vertical')
    cb.set_ticks([0,int(CMAX/2),CMAX])

    ax_n.cla()
    ax_n.set_axis_off()
    ax_n.text(0.5, 0.5, "[%s]" % train_labels[i],
            verticalalignment='center', horizontalalignment='center',
            transform=ax_n.transAxes, fontsize=50)

    plt.show(block=False)
    plt.pause(PLT_PAUSETIME)

# Naive Bayes: P(H|E) = P(E|H)/P(E) * P(H)
# 
#   P(H|E):         posterior probability
#       구하려는 값
#
#   P(E|H):         likelihood
#       multivariate normal distribution
#       P(E|H) = N(E, EoH_mean,EoH_covM)
#
#   P(H):           prior probability
#       스무딩 적용하면 다음과 같이 됨
#       (count(H)+1)/(count(all H)+1)
#
#   P(E):           evidence
#       특정 E에 대해서 argmax P(H|E)를 구할때 P(E)는 고정값이므로 필요 없음

fig_init()




##############################
# 학습
##############################
#
train_images_2d = mnist2ndarray("data/train-images-idx3-ubyte")/CMAX
train_images = train_images_2d.reshape(len(train_images_2d),-1)
train_labels = mnist2ndarray("data/train-labels-idx1-ubyte")
#
#iris = np.genfromtxt("data/iris.tsv", skip_header=1, delimiter="\t",
#        dtype=[('sl','f4'),('sw','f4'),('pl','f4'),('pw','f4'),
#                ('sp','U32')])
#
#iris_names = {'I. setosa':0, 'I. versicolor':1, 'I. virginica':2}
#
#iris_features = np.array([list(x)[0:4] for x in iris])
#iris_labels = np.array([iris_names[x['sp']] for x in iris])
#
#train_images = iris_features
#train_labels = iris_labels

LABELS = np.unique(train_labels)
assert len(train_images) == len(train_labels)
# 학습 데이터 수 
m = len(train_images)
m_part = int(m/1) # 디버깅시에 일부만 가지고 하면 편함

N_TOTAL = m_part

for i in range(m_part):
    fig_draw(i)

l, C_H = np.unique(train_labels, return_counts=True)
P_H = (C_H+1) / (sum(C_H)+1)

Using_Features = np.ones(IMGSIZE*IMGSIZE, dtype=bool)

def make_EoH():
    EoH = dict()
    t = 0
    for d in LABELS:
        d_imgs = train_images[train_labels==d]
        EoH[d] = {
            "mean" : np.mean(d_imgs[...,Using_Features],axis=0),
            "covM" : np.cov(d_imgs[...,Using_Features].T)
        }
    return EoH

EoH = make_EoH()

##############################
# 피쳐 버리기
##############################
#
# 특정 H의 covM의 determinant가 0이 되어버리면,
# 해당 H의 P(H|E) 확률이 무한대로 올라가게 된다.
# 따라서 모든 H의 covM의 determinant가 절대 0이 되지 않도록 노력해야 한다.
# 몇가지 방법들이 있는 것 같은데 아직 공부를 안 해서...
# 일단 생각나는대로 해본다.
#
# 1. covM의 row중 row전체가 0 또는 0에 가까운 것이 있으면
#       그 row에 해당하는 feature는 버린다.
#       covM.T = covM이므로 column에 대해서는 할 필요 없다. 
# 
# 그런데 1에서 문제가 되는 row 들의 경우에 대부분 분산 이전에
#   값부터가 0이어서 문제인 것 같다.
#   만약 값들은 0이 아닌데 분산만 0에 가까운 경우라면,
#   determinant를 못구한다는 것 외에도
#   그 row가 주는 의미로 볼때, 버려도 되는 (도움이 안 되는) 것이 맞는걸까?
#
for d in LABELS:
    Min_CovRow = np.median(abs(EoH[d]["covM"][abs(EoH[d]["covM"])>0.0]))
    for i, row in enumerate(EoH[d]["covM"]):
        if (np.median(abs(row))<Min_CovRow/20):
            Using_Features[i] = False

# 버려진 피쳐들 제외 하고 새로 만들기
print ("Using_Features:",sum(Using_Features))
EoH = make_EoH()

##############################
# 테스트
##############################
test_images_2d = mnist2ndarray("data/t10k-images-idx3-ubyte")/CMAX
test_images = test_images_2d.reshape(len(test_images_2d),-1)
test_labels = mnist2ndarray("data/t10k-labels-idx1-ubyte")
assert len(test_images) == len(test_labels)

#test_images = iris_features
#test_labels = iris_labels

# m = 테스트 데이터 수 
m_test = len(test_images)

# 테스트 결과 맞는 경우
test_results = np.zeros(m_test, dtype=bool)
# 테스트 결과 (첫번째 후보는 아니고) 두번째 후보가 맞는 경우
test_results_2nd = np.zeros(m_test, dtype=bool)

# multivariate normal distribution
# P(E|H) = N(E, EoH_mean,EoH_covM)
def logN(img,mean,covM):

    const = 0.0 # 빠른 계산을 위해서 0으로 처리하는 부분

#    i = plt.imshow(covM)
#    plt.colorbar(i)
#    plt.show()

    detsign,logdet = np.linalg.slogdet(covM)
#    assert detsign >= 0

    ln = -1/2*(logdet+
        np.dot((img-mean).T,np.linalg.inv(covM)).dot(img-mean)+const)

    return ln 


for i in range(m_test):
    P_HoE = np.zeros(len(LABELS))
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
                (logN(img, EoH[j]["mean"], EoH[j]["covM"])))
            
    ds = np.argsort(P_HoE)[-5:][::-1]
    d,d_2nd = ds[0:2] # 결과, 두번째 후보

    if (d==l):
        test_results[i] = True
    if (d_2nd==l):
        test_results_2nd[i] = True
    
    successes = sum(1*test_results)
    suc_inc_2nd = sum(1*test_results+1*test_results_2nd)
    tests = (i+1)

    print ("%d / %d (%3.1f), %d / %d (%3.1f)" \
            % (successes, tests, successes/tests*100.0, \
            suc_inc_2nd, tests, suc_inc_2nd/tests*100.0))
    
#    if True and test_results[i] == False:
    if False:
        print ()
        print (img)
        print (l,d,ds)
        print (P_HoE)
        print ()
#        time.sleep(1)


