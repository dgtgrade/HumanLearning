# 공부를 위해서 최대한 라이브러리를 사용하지 않고 해 보기 위해서 
# 일부러 numpy 사용하지 않음
#
# 전체적으로는 딱히 참고한 코드 없음
# 부분적으로는 웹상의 수많은 코드들을 참고 하였으나 붙여넣기 한 것은 없음
#
# python 3.5 환경에서 작성 하였음
#
# 실험용으로 대충 빨리 만들어본 코드라서 깔끔하지 않음
#
# 게다가 python 사용한지 2주일 정도 밖에 안 되어서,
# python을 잘 알고 보면 이상한 코드가 많을 것으로 추정됨
# 
# Color Palette from Color Schemer Online v2
# http://www.colorschemer.com/online.html
#
# 작성자:
# Facebook: https://facebook.com/dgtgrade
# Youtube: https://youtube.com/channel/UCEdT99nAs8nalv6Mafs9RiA 
# email: dgtgrade@gmail.com

import matplotlib.pyplot as plt
import math

H = [] # 키
W = [] # 몸무게

with open("data/2_mchc.tsv") as f:
    for line in f:
        [i, height, weight] = line.split("\t")
        height = 2.54 * float(height)
        weight = 0.453592 * float(weight)
        #print ("%.2f" % height, "%.2f" % weight)
        H.append(height)
        W.append(weight)

m = len(H) # len(W)도 동일

BoxLeft = min(H)
BoxRight = max(H)
BoxBottom = min(W)
BoxTop = max(W)
BoxWidth = BoxRight-BoxLeft
BoxHeight = BoxTop-BoxBottom

f, ax = plt.subplots(figsize=(160,90))
plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

# LBox = Larger Box
LBoxLeft = BoxLeft-BoxWidth*0.2
LBoxRight = BoxRight+BoxWidth*0.2
LBoxTop = BoxTop+BoxHeight*0.2
LBoxBottom = BoxBottom-BoxHeight*0.2

ax.set(xlim=(LBoxLeft,LBoxRight),
        ylim=(LBoxBottom,LBoxTop))

ax.set_axis_bgcolor("#000000")
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

for i in range(m):
    ax.plot(H[i],W[i],linestyle="None",marker="o",markersize=15,color="#33CCFF")
    plt.pause(0.01)

ax.axhline(y=BoxBottom, linewidth=5, linestyle="--", color="#33CCFF")
plt.pause(0.5)
ax.axvline(x=BoxRight, linewidth=5, linestyle="--", color="#33CCFF")
plt.pause(0.5)
ax.axhline(y=BoxTop, linewidth=5, linestyle="--", color="#33CCFF")
plt.pause(0.5)
ax.axvline(x=BoxLeft, linewidth=5, linestyle="--", color="#33CCFF")
plt.pause(0.5)

plt.show(block=False)

BHStep = (BoxRight - BoxLeft) / 16 
BVStep = (BoxTop - BoxBottom) / 16 

GunPosX = []
GunPosY = []

# float 지원하는 range
# 기본 range와 달리 step 이 맞는 경우 end도 포함됨
def frange(start,end,step):
    assert ((start<end and step>0) 
        or (start>end and step<0)
        or (start==end and step==0))

    r = []
    x = start
    while (x >= min(start,end) and x <= max(start,end)):
        r.append(x)
        x += step

    return r

# Bottom
r = frange(BoxLeft,BoxRight,BHStep)
GunPosX.extend(r)
GunPosY.extend([BoxBottom]*len(r))
# Right
r = frange(BoxBottom,BoxTop,BVStep)
GunPosX.extend([BoxRight]*len(r))
GunPosY.extend(r)
# Top
r = frange(BoxRight,BoxLeft,-BHStep)
GunPosX.extend(r)
GunPosY.extend([BoxTop]*len(r))
# Left
r = frange(BoxTop,BoxBottom,-BVStep)
GunPosX.extend([BoxLeft]*len(r))
GunPosY.extend(r)

#plt.plot(GunPosX, GunPosY,
#        linestyle="None",marker="D",markersize=20,color="red")
#plt.show(block=False)

# 선 y=ax+b과 큰 박스의 네변이 교차 하는 점들을 반환
def cross (a,b):

    crX = []
    crY = [] 

    # Left
    x = LBoxLeft
    y = a * x + b
    if (LBoxBottom <= y and y <= LBoxTop) : 
        crX.append(x)
        crY.append(y)

    # Right
    x = LBoxRight
    y = a * x + b
    if (LBoxBottom <= y and y <= LBoxTop) : 
        crX.append(x)
        crY.append(y)

    # Top

    if (a!=0):
        y = LBoxTop
        x = (y-b)/a # y = ax + b => x = (y - b)/a
        if (LBoxLeft <= x and x <= LBoxRight) :
            crX.append(x)
            crY.append(y)

        # Bottom
        y = LBoxBottom
        x = (y-b)/a # y = ax + b => x = (y - b)/a
        if (LBoxLeft <= x and x <= LBoxRight) :
            crX.append(x)
            crY.append(y)
    else:
        # 기울기가 0이면 y=b 선으로, 박스 왼쪽선, 오른쪽선을 지난다.
        crX.append(LBoxLeft)
        crY.append(b)
        crX.append(LBoxRight)
        crY.append(b)

    return (crX, crY)


ERRMAX = 999.99
def error(a,b):
    err = 0.0
    for i in range(m):
        x = H[i]
        y = W[i]
        h = a*x + b
        err += abs(h-y)
    err = min(err / m, ERRMAX)
    return err

ThetaStep = 6 
numGuns = len(GunPosX)

err_min_all = ERRMAX # 전체 포대에서의 최소 에러
best_line_all = None

for i in range(numGuns):
    gunX = GunPosX[i]
    gunY = GunPosY[i]

    err_min = ERRMAX  # 이번 포대에서의 최소 에러
    err = err_min
    best_line = None

    for theta in range(-90,90,ThetaStep):
        a = math.tan(math.radians(theta))
        b = gunY - a * gunX #y=ax+b => b=y-ax

        crX, crY =  cross(a,b)

        err_prev = err
        err = error(a,b)

        best = False
        best_all = False

        if (err < err_min):
            best = True
            err_min = err
            if (err < err_min_all):
                best_all = True
                err_min_all = err

        if (err < err_prev):
            err_arrow = "(-)"
            err_color = "green"
        elif (err > err_prev):
            err_arrow = "(+)"
            err_color = "red"
        else:
            err_arrow = ""
            err_color = "black"

#        txt = ax.text(0.5,0.9,"MAE :"+"%6.2f" % err+" "+err_arrow,
#            family="monospace", fontsize=30, color=err_color,
#            ha='center', transform=ax.transAxes)


        gundot, = ax.plot(gunX, gunY, 
                marker="*",markersize="40",color="#FFCC33",alpha=1.0)

        line, = ax.plot(crX, crY, 
                linestyle="--", linewidth="10",
                marker="None",markersize="30",color="#FFCC33",alpha=0.8)
        plt.show(block=False)
        plt.pause(0.01)

        gundot.remove()
        line.remove()
#        txt.remove()

        if (best):
            if (best_line != None):
                best_line.remove()
            best_line, = ax.plot(crX, crY, 
                    linestyle="--", linewidth="10",
                    marker="None",color="#FF6633",alpha=0.8)

        if (best_all):
            if (best_line_all != None):
                best_line_all.remove()
            best_line_all, = ax.plot(crX, crY, 
                    linestyle="-", linewidth="15",
                    marker="None",color="#FF3366",alpha=0.8)


    if (best_line != None):
        best_line.remove()

plt.pause(1)
plt.show(block=True)

