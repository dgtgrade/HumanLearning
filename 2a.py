import matplotlib.pyplot as plt

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

BoxLeft = min(H)
BoxRight = max(H)
BoxBottom = min(W)
BoxTop = max(W)

plt.plot(H,W,linestyle="None",marker="o",markersize=10,color="blue")
plt.axhline(y=BoxTop, linewidth=5, color="green")
plt.axhline(y=BoxBottom, linewidth=5, color="green")
plt.axvline(x=BoxLeft, linewidth=5, color="green")
plt.axvline(x=BoxRight, linewidth=5, color="green")
plt.show(block=False)

BHStep = (BoxRight - BoxLeft) / 10
BVStep = (BoxTop - BoxBottom) / 10

GunPosX = []
GunPosY = []

# Bottom
r = range(int(BoxLeft),int(BoxRight)+1,int(BHStep))
GunPosX.extend(r)
GunPosY.extend([BoxBottom]*len(r))
# Right
r = range(int(BoxBottom),int(BoxTop)+1,int(BVStep))
GunPosX.extend([BoxRight]*len(r))
GunPosY.extend(r)
# Top
r = range(int(BoxRight),int(BoxLeft)-1,-int(BHStep))
GunPosX.extend(r)
GunPosY.extend([BoxTop]*len(r))
# Left
r = range(int(BoxTop),int(BoxBottom)-1,-int(BVStep))
GunPosX.extend([BoxLeft]*len(r))
GunPosY.extend(r)

plt.plot(GunPosX, GunPosY,
        linestyle="None",marker="D",markersize=20,color="red")
plt.show()


