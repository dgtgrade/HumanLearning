# 1003.py가 1002.py 와 다른점
#
# - 목표 함수 t의 그래프 이미지로부터 정답 데이터를 로딩할 수 있음
# 
# 딱히 참고한 코드 없음
#
# 매트릭스 연산을 위해서 numpy 사용하고,
# 그래프 출력을 위해서 matplotlib 사용하였음
# 이미지 로딩을 위해서 skimage 사용하였음
#
# 그 이외에 머신 러닝 관련한 라이브러리는 사용하지 않았음
#
# python 3.5 환경에서 작성 하였음
#
# 실험용으로 대충 빨리 만들어본 코드라서 깔끔하지 않음
#
# 게다가 python 사용한지 1주일도 안 되어서,
# 특별히 이상한 코드가 많을 것으로 추정됨
#
# Back Propagation 외에 요구되는 배경 지식은 없음
#
# 작성자:
# Facebook: https://facebook.com/dgtgrade
# Youtube: https://youtube.com/channel/UCEdT99nAs8nalv6Mafs9RiA 
# email: dgtgrade@gmail.com

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import sys

# Y 값 범위
y_MAX = 100

# 그래프 이미지로부터 목표 함수 t의 데이터 로딩
#
def img2dat(imgfp, lenX):


    img = color.rgb2gray(io.imread(imgfp))

    numY, numX = np.shape(img)

    # 어느 정도 어두운 값만 필터링한 점의 집합
    dots = np.where(img < 0.2)

    # f0[i][0]: X 좌표가 i인 모든점의 y 좌표값 합
    # f0[i][1]: X 좌표가 i인 모든점의 개수
    # f1[i]: X 좌표가 i인 모든점의 평균 y 좌표값
    f0 = np.zeros((numX,2))
    f1 = np.zeros(numX)

    for i in range(len(dots[0])):
        ypos = dots[0][i]
        xpos = dots[1][i]

        f0[xpos][0] += ypos
        f0[xpos][1] += 1
        f1[xpos] = f0[xpos][0]/f0[xpos][1]

        # y 값을 [0,y_MAX]로 스케일링
        # 이미지 배열에서의 y 좌표와 함수 그래프에서의 y 값은 반대
        f1[xpos] = (1 - f1[xpos]/numY)*y_MAX

    # 최종함수 f 생성
    #
    # f1의 인덱스 범위를 X의 인덱스 범위로 다시 맵핑
    f = np.zeros((lenX,1))
    for i in range(lenX):
        Xpos = int((numX-1) * (i / (lenX-1))) 
        f[i][0] = f1[Xpos]

    return f

# plt 기본 폰트사이즈
plt.rc('xtick', labelsize=30) 
plt.rc('ytick', labelsize=30) 

# 단순한 sigmoid 함수
Z_MAX = 100
def sigmoid(z):
    z = np.clip(z, -Z_MAX, Z_MAX) # overflow 에러 방지
    return 1.0 / (1.0 + np.exp(-z))

# 단순한 sigmoid 미분 함수
# a는 scalar 값 또는 ndarray
def d_sigmoid(a):
    return a * (1.0-a)

# 네트워크 구조는 다음과 같다.
# 입력레이어 (a1) --(th1)--> 히든레이어 (a2)
# 히든레이어 (a2) --(th2)--> 출력레이어 (a3)
# th1, th2 는 레이어 사이의 세타값 모음

# 목표 함수 t
def t(t_str, X):
    Y = eval(t_str)
    return Y
    
#t_str = "X*X"
#t_str = "8*X**2-X**3"
#t_str = "10*np.sin(X)+(X-4)**2-10"
#imgfp = "data/1003_plot1.png"
imgfp = sys.argv[1]
t_str = "img2dat('"+imgfp+"', len(X))"

# 입력값 범위
x_START = -5.0
x_END = 10.0

# 학습용 정답 데이터
X_train = np.linspace(x_START,x_END,num=40).reshape(-1,1)
Y_train = t(t_str,X_train)
# 시험용 정답 데이터
X_test = np.linspace(x_START,x_END,num=100).reshape(-1,1)
Y_test = t(t_str,X_test)
# 고정밀 정답 데이터 
# 그래프 상에 표시하기 위한 것
X_t = np.linspace(x_START,x_END,num=200).reshape(-1,1)
Y_t = t(t_str,X_t)

N_IN = X_train.shape[1] # 입력 레이어 노드수
N_OUT = Y_train.shape[1] # 출력 레이어 노드수
N_HD1 = 100 # 히든 레이어 노드수

# 정답표 예제수
m_train = X_train.shape[0]
m_test = X_test.shape[0]

# thetas는  th1와 th2를 하나의 일차원 어레이로 표현한 것
RSIZE = 1 # 초기 랜덤값의 범위 설정
np.random.seed(20160823) # 디버깅 및 논의를 쉽게 하기 위해서 지정
thetas = RSIZE * np.random.random_sample((N_IN+1)*N_HD1+(N_HD1+1)*N_OUT)

LR = 0.02 / m_train # Learning Rate

def th_split(thetas):

    th1 = np.reshape(thetas[:N_HD1*(1+N_IN)], (N_HD1,1+N_IN))
    th2 = np.reshape(thetas[N_HD1*(1+N_IN):], (N_OUT,1+N_HD1))
    return (th1, th2)

def feed_forward(x,thetas):

    (th1, th2) = th_split(thetas.copy())

    a1 = np.vstack(([1.0], x)) # bias 노드 추가
    z2 = np.dot(th1, a1)
    a2 = sigmoid(z2)
    a2 = np.vstack(([1.0], a2)) # bias 노드 추가
    a3 = np.dot(th2, a2)

    return (a1, z2, a2, a3)

def cost(a3,y):

    return np.sum((a3 - y) ** 2 / 2)

def back_propagation(a1, z2, a2,a3, thetas,y, LR):

    new_thetas = thetas.copy()
    (new_th1, new_th2) = th_split(new_thetas)

    # 마지막 레이어에 activation 함수 있을때
    #delta3 = (a3 - y)*d_sigmoid(a3) 
    # 없을때
    delta3 = (a3 - y)
    delta2 = np.dot(new_th2.T, delta3) * d_sigmoid(a2)

    new_th2 -= LR * np.dot(delta3, a2.T)
    new_th1 -= LR * np.dot(delta2[1:,:], a1.T)

    return np.append(new_th1.flatten(), new_th2.flatten())

# numerical gradient descent
# 계산적 기울기를 이용해서 theta 값 조정하기
def num_grad_desc(x,thetas,y,LR):

    new_thetas = thetas.copy() #새로운 thetas
    tmp_thetas = thetas.copy() #thetas 중에서 1개 theta만 변경한 것
    DELTA = 0.001

    for i in range(thetas.size):

        # 계산적 기울기 구하기
        tmp_thetas[i] = thetas[i] - DELTA
        (a1, z2, a2, a3) = feed_forward(x, tmp_thetas)
        c1 = cost(a3, y)
        tmp_thetas[i] = thetas[i] + DELTA
        (a1, z2, a2, a3) = feed_forward(x, tmp_thetas)
        c2 = cost(a3, y)
        tmp_thetas[i] = thetas[i]

        # 기울기에 따라서 thetas 조정
        new_thetas[i] -= LR*(c2 - c1)/(DELTA*2)

    return new_thetas

count =0
PLT_PAUSETIME=0.1

plt.ion()
plt.figure(figsize=(160,120))

while True: 
    
    count += 1

    DOTEST=False
    if ((count % 1000 == 0)
        or (count < 1000 and count % 100 == 0)
        or (count < 100 and count % 10 == 0)
        or (count < 10)):
        DOTEST = True

    total_cost_train = 0.0
    total_cost_test = 0.0

    print (count)

    # 학습
    for i in range(m_train):

        # 정답
        x = X_train[i:i+1,:].T
        y = Y_train[i:i+1,:].T

        # feed forward
        (a1, z2, a2, a3) = feed_forward(x, thetas)

        # cost
        c = cost(a3, y)
        total_cost_train += c

        # back propagation
        bp_thetas = back_propagation(a1, z2, a2, a3, thetas, y, LR)
        thetas = bp_thetas

        # 계산적 기울기와 비교
        if False: # BP 알고리즘 디버깅 시에만 사용
            ngd_thetas = num_grad_desc(x, thetas, y, LR)
            thetas = ngd_thetas

    # 시험 치고 그래프 업데이트
    if DOTEST:

        A3 = np.empty([m_test,1])

        for i in range(m_test):
            x = X_test[i:i+1,:].T
            y = Y_test[i:i+1,:].T
            (a1, z2, a2, a3) = feed_forward(x, thetas)
            c = cost(a3, y)
            A3[i] = a3
            total_cost_test += c

        plt.clf()

        ax = plt.subplot(111)
        ax.axis([x_START,x_END, 0,y_MAX*1.2])
        ax.text(0.5,0.9,"t(X)="+imgfp,
            fontsize=50, ha='center', transform=ax.transAxes)
        ax.text(0.5,0.8,"Iteration:"+str(count),
            fontsize=30, ha='center', transform=ax.transAxes)

        ax.plot(X_t, Y_t, 
            c="green", marker="o", markersize=5.0, linewidth=5.0)
        ax.plot(X_train, Y_train, 
            c="blue", marker="d", markersize=20.0, linestyle='None')
        ax.plot(X_test, A3, 
            c="red", marker="o", markersize=10.0, linestyle='None')

        plt.draw()
        plt.pause(PLT_PAUSETIME)

