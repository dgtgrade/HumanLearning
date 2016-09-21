import time
import random
import math
from enum import Enum

import numpy as np

from mnist2ndarray import *

##############################
# 프로그램 전체 설정
##############################
#
#np.random.seed(20160920) # 디버깅 및 논의를 쉽게 하기 위해서 지정

##############################
# 뉴럴 네트워크 
##############################
#
class NN:

    LR = 0.5 # Learning Rate

    ths = None # thetas
    nas = None # a of each nodes / without bias node
    nzs = None # z of each nodes

    n_nodes = None

    # n_nodes = [입력 노드수, 히든 레이어 #1 노드수, ..., 출력 노드수]
    def __init__(self,n_nodes):
        self.n_nodes = n_nodes
        self.nzs = np.zeros(sum(n_nodes))
        self.nas = np.zeros(sum(n_nodes))
        self.ths = np.random.uniform(-1.0,1.0,
                sum((n_nodes[0:-1]+1)*n_nodes[1:]))

    def cost(self,y):

        n_nodes = self.n_nodes
        assert len(y) == n_nodes[-1]
        oas = self.nas[-n_nodes[-1]:]   # output node_as
        return np.sum((oas-y)**2/2.0)

    def ff(self,x):

        n_nodes = self.n_nodes
        assert len(x) == n_nodes[0]

        self.nas[0:len(x)] = x # input node_a's

        # pl_ : of previous (left) layer
        for l in range(1,len(n_nodes)):

            pl_nas = self.__get_nas(l-1)
            pl_nas = np.append([1.0],pl_nas) # add bias node

            thsM = self.__get_thsM(l)
            nzs = self.__get_nzs(l)
            nas = self.__get_nas(l)

            nzs[:] = np.dot(thsM,pl_nas)
            nas[:] = self.__sigmoid(nzs)


    def __bp(self,y):

        ths_all = self.ths
        n_nodes = self.n_nodes
        new_ths_all = self.ths.copy()

        # nl_ : of next (right) layer
        # pl_ : of previous (left) layer
        for l in range(len(n_nodes)-1,0,-1):

            new_thsM = self.__get_thsM(l,new_ths_all)
            nas = self.__get_nas(l)

            if (l == len(n_nodes)-1):
                deltas = (nas-y)*nas*(1.0-nas) 
            else:
                nl_thsM = self.__get_thsM(l+1,ths_all)
                deltas = np.dot(nl_thsM[:,1:].T,nl_deltas)*nas*(1.0-nas) 

            pl_nas = self.__get_nas(l-1)
            pl_nas = np.append([1.0],pl_nas) # add bias node

            new_thsM -= self.LR*np.dot(deltas.reshape(len(deltas),-1),
                    pl_nas.reshape(-1,len(pl_nas)))

            new_ths = self.__get_ths(l,new_ths_all)
            new_ths[:] = new_thsM.flatten()

            nl_deltas = deltas

        return new_ths_all

    def bp(self,x,y):
        self.ff(x)
        self.ths = self.__bp(y)

    def batch_bp(self,X,Y):

        new_ths = np.zeros(self.ths.shape)
        n_X = len(X)
        for i in range(n_X):
            x = X[i]
            y = Y[i]
            self.ff(x)
            new_ths += self.__bp(y)
        self.ths = new_ths/n_X

    def __ng(self,X,Y):

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
                new_ths[j] -= self.LR/n_X*(c2-c1)/(DELTA*2)

        return new_ths

    def ngd(self,x,y):
        self.ths = self.__ng(np.array([x]),np.array([y]))

    def batch_ngd(self,X,Y):
        self.ths = self.__ng(X,Y)

    def batch_cost(self,X,Y):
        cost = 0.0
        n_X = len(X)
        for i in range(n_X):
            nn.ff(X[i])
            cost += nn.cost(Y[i])/n_X
        return cost

    def get_output(self):
        return self.__get_nas(len(self.n_nodes)-1)

    def __get_ths(self,l,ths=None):
        n_nodes = self.n_nodes
        if (ths is None): ths=self.ths
        return ths[sum((n_nodes[0:l-1]+1)*n_nodes[1:l]):
            sum((n_nodes[0:l]+1)*n_nodes[1:l+1])]

    # return Matrix of thetas of layer l
    # Matrix: output neurons(rows)*input neurons(columns)
    def __get_thsM(self,l,ths=None):
        n_nodes = self.n_nodes
        if (ths is None): ths=self.ths
        return self.__get_ths(l,ths).reshape(n_nodes[l],-1)

    def __get_nzs(self,l,nzs=None):
        n_nodes = self.n_nodes
        if (nzs is None): nzs=self.nzs
        return nzs[sum(n_nodes[0:l]):sum(n_nodes[0:l+1])]

    def __get_nas(self,l,nas=None):
        n_nodes = self.n_nodes
        if (nas is None): nas=self.nas
        return nas[sum(n_nodes[0:l]):sum(n_nodes[0:l+1])]


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

##############################
# 학습/테스트 데이터 관련 설정
##############################
#
CMAX=0xff

##############################
# 학습/테스트
##############################
#
LMode = Enum('LearnMode', 'BATCH MINI_BATCH STOCHASTIC')
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

# 뉴럴 네트워크 생성
nn = NN(np.array([n_FEATURES,20,20,n_LABELS]))

# 리턴: 출력 노드중 가장 큰 값의 번호
#
def classify(x):

    nn.ff(x)
    o = nn.get_output()
    l = np.argmax(o)

    return l

# 학습 이터레이션 설정
#lmode = LMode.BATCH
lmode = LMode.MINI_BATCH
#lmode = LMode.STOCHASTIC
MINI_BATCH_SIZE = max(2,math.ceil(m/1000))

count = 0
while True:

    count += 1

    if (lmode == LMode.STOCHASTIC):
        i = np.random.randint(m)
        x = train_inputs[i]
        y = train_outputs[i]
        #nn.ngd(x,y)
        nn.bp(x,y)
    elif (lmode == LMode.MINI_BATCH):
        I = np.random.permutation(m)[0:MINI_BATCH_SIZE]
        X = train_inputs[I]
        Y = train_outputs[I]
        #nn.batch_ngd(X,Y)
        nn.batch_bp(X,Y)
    else:
        #nn.batch_ngd(train_inputs,train_outputs)
        nn.batch_bp(train_inputs,train_outputs)

    # 현재 학습 상황 출력
    if (count % 10 == 0):
        print ("#%d: %.9f"%(count,
                nn.batch_cost(train_inputs, train_outputs)))

    # 테스트 및 그 결과 출력
    if (count % 10 == 0):
        if (dset == DataSet.MNIST):
            test_results = np.zeros(test_labels.shape)
            for i in range(m_test):
                test_results[i] = (test_labels[i]==classify(test_inputs[i]))
            print ("#%d: %d/%d"%(count,
                np.count_nonzero(test_results),m_test))
        else: # XOR
            for i in range(m_test):
                print (test_inputs[i], test_labels[i], 
                        classify(test_inputs[i]))

