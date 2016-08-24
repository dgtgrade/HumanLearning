from skimage import io, color
import numpy as np
import sys

imgfp = sys.argv[1]

img = color.rgb2gray(io.imread(imgfp))
numY, numX = np.shape(img)

# 어느 정도 어두운 값만 필터링한 점의 집합
dots = np.where(img < 0.2)

# f0[i][0]: X 좌표가 i인 모든점의 y 좌표값 합
# f0[i][1]: X 좌표가 i인 모든점의 개수
# f[i]: X 좌표가 i인 모든점의 평균 y 좌표값
# f: 최종 함수
f0 = np.zeros((numX,2))
f = np.zeros(numX)

for i in range(len(dots[0])):
    ypos = dots[0][i]
    xpos = dots[1][i]

    f0[xpos][0] += ypos
    f0[xpos][1] += 1
    f[xpos] = f0[xpos][0]/f0[xpos][1]

    # y 값을 [0,1]로 스케일링 해 줌
    # 이미지에서는 아래쪽이 y 좌표가 크고,
    # 함수그래프에서는 위쪽이 y 좌표가 크므로,
    # y 값을 뒤집어 주어야 함
    f[xpos] = 1 - f[xpos]/numY


# 소수점 출력 설정
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

print (f)


