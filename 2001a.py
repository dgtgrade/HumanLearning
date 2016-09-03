import matplotlib.pyplot as plt
from mnist2ndarray import *
import time

# 정해져 있는 값들
IMGSIZE=28
CMAX=255
DIGITS=10

# 설정값들
PLT_PAUSETIME=1
DO_PLT = False # 그래프로 표시할지 여부
NP_LINEWIDTH = IMGSIZE*8+10 # 콘솔 한줄의 길이 설정

# numpy 어레이 데이터 출력 형태 설정
float_formatter = lambda x: "%7.1f" % x
np.set_printoptions(formatter={'float_kind':float_formatter},
    linewidth=NP_LINEWIDTH)

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

# 가정 H(Hypothesis)
#   예) 이 숫자는 7이다.
#
# 관찰된 증거 E(Evidence)
#   예) 이 픽셀은 칠해져있다. (픽셀데이터값이 255이다)
#
# 최종 목표 함수 = P(H|E)
#
# 베이지안 룰: P(H|E) = P(E|H)/P(E) * P(H)
# 
#   P(H|E):         posterior probability
#
#   P(E|H):         likelihood
#   P(H):           prior probability
#   P(E):           evidence
# 
#   P(E|H) = N(E and H) / N(H)
#   P(H)   = N(H) / N_TOTAL
#   P(E)   = N(E) / N_TOTAL

#fig_init()

##############################
# 학습
##############################
#
train_images = mnist2ndarray("data/train-images-idx3-ubyte")
train_labels = mnist2ndarray("data/train-labels-idx1-ubyte")
assert len(train_images) == len(train_labels)
# m = 학습 데이터 수 
m = len(train_images)
m_part = int(m/1) # 디버깅시에 일부만 가지고 하면 편함

N_EnH = np.zeros((DIGITS,IMGSIZE,IMGSIZE))
N_H = np.zeros(DIGITS)
N_E = np.zeros((IMGSIZE,IMGSIZE))

N_TOTAL = m_part

for i in range(m_part):
#    fig_draw(i)

    image = train_images[i]
    label = train_labels[i]

    N_H[label] += 1
    # 만약 0.3 이라면 (1이 아니라) 0.3 회수 만큼 관측된 것으로 함
    N_E += image/CMAX
    N_EnH[label] += image/CMAX 

    if False and label == 1:
        print (label)
        print ()
        print (N_H)
        print ()
        print (N_E)
        print ()
        print (N_EnH[1])
        print ()
        time.sleep(1)

##############################
# 확률 계산
##############################
#
# 간단한 것부터 계산
#
# 아래 모든 P들에 스무딩 적용함
# 즉 아래 모든 P들은 항상 0보다는 큼
# 게다가 1보다도 작음
# 아래쪽 테스트 부분 코드의 p_1ohx, p_0ohx, p_1x, p_0x 값들이 
# 0이 될 수 없게 됨
# https://en.wikipedia.org/wiki/Additive_smoothing
#
#   P(H)   = N(H) / N_TOTAL
#       스무딩: (N(H)+1) / (N_TOTAL+DIGITS)
P_H = (N_H+1)/(N_TOTAL+DIGITS)
#   P(E)   = N(E) / N_TOTAL
#       스무딩: (N(E)+1) / (N_TOTAL+2)
P_E = (N_E+1)/(N_TOTAL+2)
#   P(E|H) = N(E and H) / N(H)
#       스무딩: (N(E and H)+1) / ( N(H)+2)
P_EoH = np.zeros((DIGITS,IMGSIZE,IMGSIZE))
for i in range(DIGITS):
    P_EoH[i] = (N_EnH[i]+1) / (N_H[i]+2)

if False:
    print (P_H)
    print (P_E)
    print (P_EoH[1])
    print (P_EoH[5])
    time.sleep(10)

##############################
# 테스트
##############################
np.set_printoptions(linewidth=NP_LINEWIDTH)

test_images = mnist2ndarray("data/t10k-images-idx3-ubyte")
test_labels = mnist2ndarray("data/t10k-labels-idx1-ubyte")
assert len(test_images) == len(test_labels)
# m = 테스트 데이터 수 
m_test = len(test_images)

# 테스트 결과 맞는 경우
test_results = np.zeros(m_test, dtype=bool)
# 테스트 결과 (첫번째 후보는 아니고) 두번째 후보가 맞는 경우
test_results_2nd = np.zeros(m_test, dtype=bool)

for i in range(m_test):
    P_HoE = np.zeros(DIGITS)
    img = test_images[i]/CMAX
    l = test_labels[i]

    # P(H|E) = P(E|H)/P(E) * P(H)
    # Bayes' Rule for Multiple Variables
    # P(H|E1,E2,...,En) = P(E1,E2,...,En|H)/P(E1,E2,...,En)*P(H)
    #   E1,E2,...En이 완전히 독립적이라면:
    #       P(E1,..,En|H) = P(E1|H)*P(E2|H)*..*P(En|H)
    #       P(E1,..,En) = P(H)*P(E1|H)*..*P(En|H)+P(^H)*P(E1|^H)...*P(En|^H)
    #
    for j in range(DIGITS):

        # 일단 img의 픽셀값이 0.5 보다 크면 채워져있는 것으로 해본다.
        #
        # P(E1,..,En|H)
        CON = 0.5
        p_1oh = (img)*P_EoH[j] # 채워져있는 Evidence
        p_1ohx = p_1oh[img>CON].prod()

        p_0oh = (1.0-img)*(1-P_EoH[j]) # 비어있는 Evidence
        p_0ohx = p_0oh[img<=CON].prod()

        # P(E1,..,En)
        p_1 = (img)*P_E # 채워져있는 Evidence
        p_1x = p_1[img>CON].prod()

        p_0 = (1.0-img)*(1-P_E) # 비어있는 Evidence
        p_0x = p_0[img<=CON].prod()

        P_HoE[j] = P_H[j] * \
            p_1ohx * p_0ohx / \
            ((p_1x * p_0x))

            
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
    
    if False and test_results[i] == False:
        print ()
        print (img>CON)
        print (l,d,ds)
        print (P_HoE)
        print ()
#        time.sleep(1)


