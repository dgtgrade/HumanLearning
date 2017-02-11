x1 = 1
x2 = 1

def activate(z):
    return 1 if z > 0 else 0

w3 = [0, 1, 1]
w4 = [-1, 1, 1]
wo = [-2, 3, -2]

z3 = 1 * w3[0] + x1 * w3[1] + x2 * w3[2]
x3 = activate(z3)
z4 = 1 * w4[0] + x1 * w4[1] + x2 * w4[2]
x4 = activate(z4)

zo = 1 * wo[0] + x3 * wo[1] + x4 * wo[2]
xo = activate(zo)

out = xo

print(out)
