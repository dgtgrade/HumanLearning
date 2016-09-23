import time
import random
import math

from enum import Enum
from datetime import datetime

import numpy as np

from mnist2ndarray import *

##############################
# 프로그램 전체 설정
##############################
#
# 디버깅, 분석 그리고 논의를 쉽게 하기 위해서 랜덤 시드 지정
random.seed(20160922)
np.random.seed(20160922)

# numpy 소수점 출력 설정
float_formatter = lambda x: "%+.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

##############################
# 뉴럴 네트워크 
##############################
#
class NN:

    LRINIT = 1.0 # Initial Learning Rate
    LRDF = 0.01 # Learning Rate Decaying Factor
    DORATE = 0.0 # Dropout Rate
    MOMENTUM = 0.75 # Momentum
    LAMBDA = 0.0 # L2 Regularization Parameter

    lr = LRINIT # current learning rate

    ths = None # thetas
    nas = None # a of each nodes / without bias node
    nzs = None # z of each nodes
    dropout = None # dropout of each nodes
    doDropout = None # 테스팅 중에는 dropout 하지 않음

    # momentum
    ths_mmt = None

    ths_l = None # 레이어별 시작 위치
    ths_b = None # bias node 연결 위치 모음
    ths_d = None # dropout 된 ths 모음

    n_nodes = None
    n_nodes_l = None # 레이어별 시작 위치

    m = 0 # L2 Regularization 등에서 사용하는 전체 학습 예제수

    # n_nodes = [입력 노드수, 히든 레이어 #1 노드수, ..., 출력 노드수]
    def __init__(self,n_nodes,m):
        self.n_nodes = n_nodes
        self.nzs = np.zeros(sum(n_nodes))
        self.nas = np.zeros(sum(n_nodes))
        self.ths = np.random.uniform(-1.0,1.0,
                sum((n_nodes[0:-1]+1)*n_nodes[1:]))

        #성능을 올리기 위해서 미리 계산해 두는값들
        self.n_nodes_l = np.empty(len(n_nodes)+1).astype(np.int)
        for l in range(len(self.n_nodes_l)):
            self.n_nodes_l[l] = sum(n_nodes[0:l])

        self.ths_l = np.empty(len(n_nodes)).astype(np.int)
        self.ths_b = np.zeros(len(self.ths)).astype(np.bool)
        for l in range(len(self.ths_l)):
            self.ths_l[l] = sum((n_nodes[0:l]+1)*n_nodes[1:l+1])
            if (l<len(self.ths_l)-1):
                self.ths_b[self.ths_l[l]+\
                        np.arange(n_nodes[l+1])*(n_nodes[l]+1)] = True

        # 전체 학습 예제수
        self.m = m

        # 모멘텀
        self.reset_mmt()

        # dropout
        self.set_dropout()

    # 드랍아웃될 유닛들 선택
    # 이미 설정되어 있으면 재선택
    def set_dropout(self):
        n_nodes = self.n_nodes
        dropout=np.random.binomial([np.ones(sum(n_nodes))],
                self.DORATE)[0].astype(np.bool)
        dropout[0:n_nodes[0]]=False # no dropout units in first layer
        dropout[-n_nodes[-1]:]=False # no dropout units in last layer
        self.dropout = dropout

        # ths_d 만들기
        self.ths_d = np.zeros(len(self.ths)).astype(np.bool)
        for l in range(len(self.ths_l)):
            if (l<len(self.ths_l)-1):
                # input node가 dropout 된 경우
                for i in np.flatnonzero(self.__get_dropout(l)):
                    self.ths_d[self.ths_l[l]+\
                        # 아래의 +1은 bias 노드 때문
                        np.arange(n_nodes[l+1])*(n_nodes[l]+1)+(i+1)] = True
                # output node가 dropout 된 경우
                for i in np.flatnonzero(self.__get_dropout(l+1)):
                    self.ths_d[self.ths_l[l]+i*(n_nodes[l]+1):\
                        self.ths_l[l]+(i+1)*(n_nodes[l]+1)] = True

    def lr_decay(self):
        self.lr = self.lr * (1-self.LRDF)

    def reset_mmt(self):
        self.ths_mmt = np.zeros(self.ths.shape)

    def ff(self,x):

        n_nodes = self.n_nodes
        assert len(x) == n_nodes[0]

        self.nas[0:n_nodes[0]] = x # input node_a's

        # pl_ : of previous (left) layer
        pl_nas = np.append([1.0],self.nas[0:n_nodes[0]])
        for l in range(1,len(n_nodes)):

            thsM = self.__get_thsM(l-1)
            nzs = self.__get_nzs(l)
            nas = self.__get_nas(l)

            nzs[:] = np.dot(thsM,pl_nas)
            nas[:] = self.__sigmoid(nzs)

            # 드랍아웃 사용시에는 traing 하고 testing 할때 각각
            # 계산방법이 다르다.
            if (self.doDropout):
                dropout = self.__get_dropout(l)
                nas[:] = nas*np.invert(dropout)
            else:
                nas[:] = nas*(1.0-self.DORATE)

            pl_nas = nas
            pl_nas = np.append([1.0],pl_nas) # add bias node

    def __bp(self,y):

        ths_all = self.ths
        n_nodes = self.n_nodes
        new_ths_all = self.ths.copy()

        # nl_ : of next (right) layer
        # pl_ : of previous (left) layer
        for l in range(len(n_nodes)-1,0,-1):

            new_thsM = self.__get_thsM(l-1,new_ths_all)
            ths_dM = self.__get_thsM(l-1,self.ths_d)
            nas = self.__get_nas(l)

            if (l == len(n_nodes)-1):
                #deltas = (nas-y)*nas*(1.0-nas) # quadratic
                deltas = (nas-y) # cross-entropy
            else:
                nl_thsM = self.__get_thsM(l,ths_all)
                deltas = np.dot(nl_thsM[:,1:].T,nl_deltas)*\
                        nas*(1.0-nas)

            pl_nas = self.__get_nas(l-1)
            pl_nas = np.append([1.0],pl_nas) # add bias node

            new_thsM -= np.invert(ths_dM) *\
                self.lr*np.dot(deltas.reshape(len(deltas),-1),
                pl_nas.reshape(-1,len(pl_nas)))

            nl_deltas = deltas

        return new_ths_all

    def bp(self,x,y):
        self.ff(x)
        new_ths = self.__bp(y) - self.__d_L2()
        self.ths_prev = self.ths
        self.ths = new_ths + self.MOMENTUM*self.ths_mmt
        self.ths_mmt = self.ths - self.ths_prev

    def batch_bp(self,X,Y):

        new_ths = np.zeros(self.ths.shape)
        n_X = len(X)
        for i in range(n_X):
            x = X[i]
            y = Y[i]
            self.ff(x)
            new_ths += self.__bp(y)

        new_ths /= n_X
        new_ths -= self.__d_L2()

        self.ths_prev = self.ths
        self.ths = new_ths + self.MOMENTUM*self.ths_mmt
        self.ths_mmt = self.ths - self.ths_prev

    def __ng(self,X,Y):

        # 아직 dropout 고려하여 구현안됨
        assert self.DORATE == 0.0
        # 아직 L2 Regularization 고려하여 구현안됨
        assert LAMBDA == 0.0
        # 아직 Momentum 고려하여 구현안됨
        assert MOMENTUM == 0.0

        ths = self.ths
        new_ths = ths.copy()
        DELTA = 0.001

        n_X = len(X)

        for i in range(n_X):
            x = X[i]
            y = Y[i]
            for j in range(ths.size):

                th_org = ths[j] 

                # 계산적 기울기 구하기
                ths[j] = th_org - DELTA
                self.ff(x)
                c1 = self.cost(y)
                ths[j] = th_org + DELTA
                self.ff(x)
                c2 = self.cost(y)
                ths[j] = th_org

                # 기울기에 따라서 ths 조정
                new_ths[j] -= self.lr/n_X*(c2-c1)/(DELTA*2)

        return new_ths

    def ngd(self,x,y):
        self.ths = self.__ng(np.array([x]),np.array([y]))

    def batch_ngd(self,X,Y):
        self.ths = self.__ng(X,Y)

    def __d_L2(self):
        if (self.LAMBDA==0): return 0 # 빨라지려나?

        # 아무래도 regularization 할때도 dropout도 고려해야 맞을 듯?
        # 이에 대해 참고할 자료가 있는가?
        return self.lr*self.LAMBDA/self.m*\
            self.ths*\
            np.logical_or(not self.doDropout,np.invert(self.ths_d))*\
            np.invert(self.ths_b)

    def __L2(self):
        if (self.LAMBDA==0): return 0 # 빨라지려나?
        return self.LAMBDA/2.0/self.m*\
            sum((self.ths*\
                np.logical_or(not self.doDropout,np.invert(self.ths_d))*\
            np.invert(self.ths_b))**2)

    def cost(self,y,withL2=False):

        n_nodes = self.n_nodes
        assert len(y) == n_nodes[-1]
        oas = self.nas[-n_nodes[-1]:]   # output node_as
        #cost = np.sum((oas-y)**2/2.0)  # quadratic
        cost = -np.sum(y*np.log(oas)+(1-y)*np.log(1-oas)) # cross-entropy
        if (withL2): cost += self.__L2()

        return cost

    def batch_cost(self,X,Y):
        cost = 0.0
        n_X = len(X)
        for i in range(n_X):
            nn.ff(X[i])
            cost += nn.cost(Y[i])/n_X

        cost += self.__L2()

        return cost

    def get_output(self):
        return self.__get_nas(len(self.n_nodes)-1)

    def __get_ths(self,l,ths=None):
        ths_l = self.ths_l
        if (ths is None): ths=self.ths
        return ths[ths_l[l]:ths_l[l+1]]

    # return Matrix of thetas of layer l
    # Matrix: output neurons(rows)*input neurons(columns)
    def __get_thsM(self,l,ths=None):
        n_nodes = self.n_nodes
        if (ths is None): ths=self.ths 
        return self.__get_ths(l,ths).reshape(n_nodes[l+1],-1)

    def __get_nzs(self,l,nzs=None):
        n_nodes_l = self.n_nodes_l
        if (nzs is None): nzs=self.nzs
        return nzs[n_nodes_l[l]:n_nodes_l[l+1]]

    def __get_nas(self,l,nas=None):
        n_nodes_l = self.n_nodes_l
        if (nas is None): nas=self.nas
        return nas[n_nodes_l[l]:n_nodes_l[l+1]]

    def __get_dropout(self,l,dropout=None):
        n_nodes_l = self.n_nodes_l
        if (dropout is None): dropout=self.dropout
        return dropout[n_nodes_l[l]:n_nodes_l[l+1]]

    # 단순한 sigmoid 함수
    # z는 scalar 값 또는 ndarray
    def __sigmoid(self,z):
        Z_MAX = 100
        z = np.clip(z,-Z_MAX,Z_MAX) # overflow 에러 방지
        return 1.0/(1.0+np.exp(-z))

    # 단순한 sigmoid 미분 함수
    # a는 scalar 값 또는 ndarray
    def __d_sigmoid(self,a):
        return a*(1.0-a)

    def print_ths(self):
        n_nodes = self.n_nodes
        for l in range(0,len(n_nodes)-1):
            thsM = self.__get_thsM(l)
            print ("layer #%d<-%d"%(l+1,l))
            print (thsM)

    def print_dropout(self):
        n_nodes = self.n_nodes
        for l in range(0,len(n_nodes)):
            dropout = self.__get_dropout(l)
            print ("layer #%d"%l)
            print (dropout)

##############################
# 학습/테스트 데이터 관련 설정
##############################
#
CMAX=0xff

##############################
# 학습/테스트
##############################
#
LMode = Enum('LearnMode', 'BATCH MINI_BATCH STOCHASTIC ONE_PER_EPOCH')
DataSet = Enum('DataSet', 'MNIST XOR')

# 사용할 데이터 선택
#dset = DataSet.XOR
dset = DataSet.MNIST

if (dset == DataSet.MNIST):

    train_images_2d = mnist2ndarray("data/train-images-idx3-ubyte")/CMAX
    train_inputs = train_images_2d.reshape(len(train_images_2d),-1)
    train_labels = mnist2ndarray("data/train-labels-idx1-ubyte")

    test_images_2d = mnist2ndarray("data/t10k-images-idx3-ubyte")/CMAX
    test_inputs = test_images_2d.reshape(len(test_images_2d),-1)
    test_labels = mnist2ndarray("data/t10k-labels-idx1-ubyte")

else: # 기본은 XOR

    # XOR
    train_inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    train_labels = np.array([0,1,1,0])
    test_inputs = train_inputs
    test_labels = train_labels

# 라벨수
LABELS = np.unique(train_labels)
n_LABELS = len(LABELS)

# 학습 데이터수
assert len(train_inputs) == len(train_labels)
m = len(train_inputs)

# 테스트 데이터수
assert len(test_inputs) == len(test_labels)
m_test = len(test_inputs)

# 입력 피쳐수
n_FEATURES = train_inputs.shape[1]

# 출력 one vs all 생성
train_outputs = np.zeros([m,n_LABELS])
train_outputs[range(m),train_labels]=1.0


# 리턴: 출력 노드중 가장 큰 값의 번호
#
def classify(x):

    nn.ff(x)
    o = nn.get_output()
    l = np.argmax(o)

    return l

m = math.ceil(m) # 테스트 시에는 m을 작게
m_test = math.ceil(m_test) # 테스트 시에는 m_test을 작게

# 학습 이터레이션 설정
#lmode = LMode.BATCH
lmode = LMode.MINI_BATCH
#lmode = LMode.STOCHASTIC
#lmode = LMode.ONE_PER_EPOCH # 테스트용
MINI_BATCH_SIZE = max(2,math.ceil(m/100))

# 뉴럴 네트워크 생성
nn = NN(np.array([n_FEATURES,50,25,n_LABELS]),m)
epoch = 0

while True:

    epoch += 1
    nn.set_dropout()
    nn.lr_decay()
    nn.reset_mmt()

    # epoch 시작 출력
    if (True):
        if (True and dset == DataSet.XOR):
            time.sleep(1)
            print ("#"*32)
            print ("#%d: %s"%(epoch,datetime.now()))
            print ("#"*32)
            print ("Train started with following conditions:")
            print ("learn rate:", nn.lr)
            print ("dropout:")
            nn.print_dropout()
            print ("ths:")
            nn.print_ths()

    # 학습 중에는 dropout 실행
    nn.doDropout = True

    # m 값만 바꾸면 train 대상 전체 집합 변경 가능하다.
    if (lmode == LMode.STOCHASTIC):
        for i in np.random.permutation(m):
            i = np.random.randint(m)
            x = train_inputs[i]
            y = train_outputs[i]
            #nn.ngd(x,y)
            nn.bp(x,y)
    elif (lmode == LMode.MINI_BATCH):
        A = np.random.permutation(m)
        for i in range(0,m,MINI_BATCH_SIZE):
            I = A[i:min(m,i+MINI_BATCH_SIZE)]
            X = train_inputs[I]
            Y = train_outputs[I]
            #nn.batch_ngd(X,Y)
            nn.batch_bp(X,Y)
    elif (lmode == LMode.ONE_PER_EPOCH):
        i = np.random.randint(m)
        x = train_inputs[i]
        y = train_outputs[i]
        #nn.ngd(x,y)
        nn.bp(x,y)
    else:
        #nn.batch_ngd(train_inputs,train_outputs)
        nn.batch_bp(train_inputs[0:m],train_outputs[0:m])

    # 학습된 결과 출력
    if (True and dset == DataSet.XOR):
        print ("Train completed with following thetas:")
        print ("ths:")
        nn.print_ths()

    if (False):
        print ("#%d: %.9f"%(epoch,
            nn.batch_cost(train_inputs[0:m],train_outputs[0:m])))

    # 테스트 및 그 결과 출력
    # m_test 값만 바꾸면 test 대상 전체 집합 변경 가능하다.
    nn.doDropout = False
    if (True):
        if (dset == DataSet.MNIST):
            train_results = np.zeros(m)
            for i in range(m):
                train_results[i] = \
                    (train_labels[i]==classify(train_inputs[i]))
            train_accuracy = np.count_nonzero(train_results)/m
            test_results = np.zeros(m_test)
            for i in range(m_test):
                test_results[i] = \
                        (test_labels[i]==classify(test_inputs[i]))
            test_accuracy = np.count_nonzero(test_results)/m_test
            print ("#%d: %.3f%%, %.3f%%"%
                    (epoch,train_accuracy*100,test_accuracy*100))
        else: # XOR
            for i in range(m_test):
                print (test_inputs[i], test_labels[i], 
                        classify(test_inputs[i]))
