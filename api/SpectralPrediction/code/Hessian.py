"""
Author: Hao Ma
Last update: 2022-01-23
Mail: oaham@xmu.edu.cn
Please refer 吴国祯, 拉曼谱学:峰强中的信息.第3版. 科学出版社: 2014. for a better understanding.
Usage: For understanding the calculation process of vibrations, and using water as the example.
"""
import numpy
import math
pi = 3.141592653589793
hessian = []
amu = 1.66053878E-27
b2m = 0.529177249E-10
au2 = 4.35974434E-18
# water
mass_matrix = numpy.array([[16,0,0,0,0,0,0,0,0],[0,16,0,0,0,0,0,0,0],[0,0,16,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,0,1]])
print(mass_matrix)
force_matrix = numpy.array([
[0.528081,0,0,0,0,0,0,0,0],
[0,-0.250589E-3,0,0,0,0,0,0,0],
[-0.111067, 0, 0.606743,0,0,0,0,0,0],
[-0.438922E-01,0,-0.516194E-01,0.457876E-01,0,0,0,0,0],
[ 0.0, 0.125294E-3, 0,0, -0.167271E-03,0,0,0,0],
[0.676936E-02,0,-0.523520, 0.315143E-02,0,0.539065,0,0,0],
[-0.484189,0,0.162687,-0.189549E-02,0,-0.992079E-02, 0.486084,0,0],
[0,0.125294E-03,0,0,  0.419763E-04,0,0, -0.167271E-03,0],
[0.104298, 0, -0.832231E-01, 0.484679E-01 , 0, -0.155458E-01, -0.152766, 0, 0.987689E-01]])
force_matrix_E = numpy.array([
[0.528081,0,0,0,0,0,0,0,0],
[0, -0.250589E-3,0,0,0,0,0,0,0],
[0,  0,  0.606743,0,0,0,0,0,0],
[0,  0,  0, 0.457876E-01,0,0,0,0,0],
[0, 0,  0,  0,  -0.167271E-03, 0,0,0,0],
[0,  0, 0,  0, 0, 0.539065E+00,0,0,0],
[0,  0,  0,  0,  0, 0, 0.486084E+00, 0,0 ],
[0, 0, 0,  0, 0, 0, 0,  -0.167271E-03,0],
[0, 0, 0,  0,  0, 0,  0, 0,  0.987689E-01]])
#力常数矩阵
m = force_matrix.T + force_matrix - force_matrix_E
kmat = numpy.zeros((9,9))
# print(kmat)
for i in range(0,9):
    for j in range(0,9):
        kmat[i,j] = m[i,j] / ((mass_matrix[i,i] * mass_matrix[j,j])**0.5)
# print(kmat)
new_mat= numpy.linalg.eigh(kmat)
print((new_mat[0]))
for l in range(0,9):
    if new_mat[0][l] > 0:
        v = (new_mat[0][l]*au2/(b2m*b2m*amu))**0.5/(2*pi*3.0E10)
        print(v)
    else:
        v = -1 * (abs(new_mat[0][l]) * au2 / (b2m * b2m * amu)) ** 0.5 / (2 * pi * 3.0E10)
        print(v)
