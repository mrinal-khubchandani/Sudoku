import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from predictDigit import *
from preprocessImg import *
from extractDigit import *
from solveSudoku import *

filepath = sys.argv[1]
img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = preprocess(img)
X,Y = getBlocks(img)

sudoku = []
classify(X[0])
for i in range(9):
    temp = []
    sudoku.append(temp)

su = 0
for i in range(9):
    for j in range(9):
        if Y[i*9 + j] == -1:
            sudoku[i].append(-1)
            continue
        su = su + 1
        dig = classify(X[i*9+j])
        sudoku[i].append(dig)

print(su)
for i in range(9):
    s = " ".join(str(sudoku[i]))
    print(s)

getSudoku(sudoku)
f = open('output.txt','w')

for i in range(9):
    s = " ".join(str(sudoku[i]))
    print(s)

for i in range(9):
    for j in range(9):
        f.write(str(sudoku[i][j]) + ' ')
    
    f.write('\n')

f.close()