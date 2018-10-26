import NeuralNetwork as NN
import pandas
import numpy as np
import random

df = pandas.read_csv('Data/Credit Card Fraud Detection/creditcard.csv')
#for line in df:
    #print(line)
#print(df["name"])

Values = list(df.ix[0])
inputValues = Values[:-1]
outputValues = Values[-1]
#print(outputValues)

#TestNN = NN.Neural_Network(30, [40, 40, 100, 50, 40, 30, 30, 20, 10, 10, 5, 4, 2], 1)
#TestNN.ForwardPass(inputValues)
#TestNN = NN.Neural_Network(2, [2, 1], 1)
TestNN = NN.Neural_Network(2, [2], 1)
weights = list([np.array([[0.45, 0.78], [-0.12, 0.13]]), np.array([[1.5], [-2.3]])])
#weights = list([np.array([0.45, -0.12, 0.78, 0.13]), np.array([1.5, -2.3])])

TestNN.LoadWeights(weights)
#print(TestNN.Predict([1, 0]))
TestNN.Train([[1, 0]], [[1]])
#print(TestNN.Predict([1, 0]))


XSet = []
YSet = []
for i in range(4):
    a = random.randint(0, 1)
    b = random.randint(0, 1)
    c = 0
    if(a != b):
        c = 1
    XSet.append([a, b])
    YSet.append([c])
TestNN.Train(XSet, YSet)
print(TestNN.Predict([1, 0]))
print(TestNN.Predict([0, 1]))
print(TestNN.Predict([0, 0]))
print(TestNN.Predict([1, 1]))
