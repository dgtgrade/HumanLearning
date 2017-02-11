import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', size=16)
matplotlib.rc('axes', titlesize=16)


tableau10 = {
    'Blue': [255, 127, 14],
    'Orange': [214, 39, 40],
    'Green': [148, 103, 189],
    'Red': [44, 160, 44],
    'Purple': [31, 119, 180],
    'Brown': [227, 119, 194],
    'Pink': [188, 189, 34],
    'Grey': [140, 86, 75],
    'Olive': [127, 127, 127],
    'Aqua': [23, 190, 207]
}
for c_name, rgb in tableau10.items():
    tableau10[c_name] = [v/255 for v in rgb]


def ReLU(a: np.ndarray):
    return np.maximum(a, [0])

X = np.linspace(-3, 3, 61)
Y = ReLU(X + 1) - ReLU(X - 1) - 1

plt.figure(figsize=(6, 6))
plt.plot(X, Y, c=tableau10['Blue'], linewidth=8)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.show()

