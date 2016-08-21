import numpy as np

# 딱히 참고한 코드 없음

# 소수점 출력 설정
float_formatter = lambda x: "%+.6f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

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

# 목표 함수 t = x^2 에 따른 정답표 만들기
def t_sqr():
    X_train = np.arange(0.0,12.5,1.0).reshape(-1,1)
    Y_train = X_train*X_train
    X_test = np.arange(1.1,9.0,2.6).reshape(-1,1)
    Y_test = X_test*X_test
    return (X_train, Y_train, X_test, Y_test)

# 목표 함수 t = sin(x) 에 따른 정답표 만들기
def t_sin():
    X_train = np.arange(1.0,10,1.0).reshape(-1,1)
    Y_train = np.sin(X_train)
    X_test = np.arange(1.0,10,3.8).reshape(-1,1)
    Y_test = np.sin(X_test)
    return (X_train, Y_train, X_test, Y_test)

# 목표 함수 t = XOR(x1, x2) 문제 정답표
def t_XOR():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])
    return (X, Y, X, Y)

(X_train, Y_train, X_test, Y_test) = t_sqr()

N_IN = X_train.shape[1] # 입력 레이어 노드수
N_OUT = Y_train.shape[1] # 출력 레이어 노드수
N_HD1 = 50 # 히든 레이어 노드수

# 정답표 예제수
m_train = X_train.shape[0]
m_test = X_test.shape[0]

# thetas는  th1와 th2를 하나의 일차원 어레이로 표현한 것
RSIZE = 1 # 초기 랜덤값의 범위 설정
np.random.seed(20160822) # 디버깅 및 논의를 쉽게 하기 위해서 지정
thetas = RSIZE * np.random.random_sample((N_IN+1)*N_HD1+(N_HD1+1)*N_OUT)

LR = 0.01 / m_train # Learning Rate

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

#    delta3 = (a3 - y)*d_sigmoid(a3) # 마지막 레이어에 activation 함수 있을때
    delta3 = (a3 - y) # 없을때
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
while True: 
    
    count += 1
    total_cost_train = 0.0
    total_cost_test = 0.0

#    time.sleep (0.1)
    if count % 100 == 0:
        print_status = True;
    else:
        print_status = False;

    if print_status: 
        print ("")
        print ("iteration: ",count)
#        print ("    thetas:", thetas)
        print ("  train: x, h, y, cost")

    for i in range(m_train):

        # 정답
        x = X_train[i:i+1,:].T
        y = Y_train[i:i+1,:].T

        # feed forward
        (a1, z2, a2, a3) = feed_forward(x,thetas)
        # cost
        c = cost(a3, y)
        if print_status:
            print ("    ", x.T, a3, y, np.array([c])) 
        total_cost_train += c

        # back propagation
        bp_thetas = back_propagation(a1, z2, a2, a3, thetas,y, LR)
        thetas = bp_thetas

        # 계산적 기울기와 비교
        if False: # BP 알고리즘 디버깅 시에만 사용
            ngd_thetas = num_grad_desc(x, thetas, y, LR)
            thetas = ngd_thetas
            #print ("==bp_thetas==")
            #print (bp_thetas)
            #print ("==ngd_thetas==")
            #print (ngd_thetas)


    if print_status:
        print ("     > total cost:", "%.9f" % total_cost_train)
        print ("")
        print ("  test: x, h, y, cost")
        for i in range(m_test):
            # 정답
            x = X_test[i:i+1,:].T
            y = Y_test[i:i+1,:].T
            (a1, z2, a2, a3) = feed_forward(x, thetas)
            c = cost(a3, y)
            print ("    ", x.T, a3, y, np.array([c])) 
            total_cost_test += c
        print ("     > total cost: ", "%.9f" % total_cost_test)
        print ("")


