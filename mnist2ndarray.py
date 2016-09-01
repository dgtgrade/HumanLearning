import numpy as np
import sys
import struct

# mnist 파일은 idx 포맷을 사용함
# 상세한 내용은 다음 문서를 참고
# http://yann.lecun.com/exdb/mnist/
#
def mnist2ndarray(path):
    with open(path,"rb") as bf:

        # magic number
        # D: dimension
        (ignore, dtype, D) = struct.unpack("h2b", bf.read(4))
        assert ignore == 0
        assert dtype == 0x08

        # dimension별 사이즈
        XShape = struct.unpack(">"+"i"*D, bf.read(4*D))

        # 결과 X
        X1D = np.fromfile(bf, np.uint8)
        X = np.reshape(X1D, XShape)

    return X

if __name__ == "__main__":
    X = mnist2ndarray(sys.argv[1])
    print (X)
    print (X.shape)

