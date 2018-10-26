import numpy as np
import pandas

def sigmoid(Neurons):
    #print("Neurons", Neurons)
    e = 2.71828183
    Neurons = Neurons * -1
    for i in range(len(Neurons)):
        x = (1. / (1. + e**(Neurons[i])))
        Neurons[i] = x
    return Neurons

def ConvertToVector(array):
    if(type(array) != "<class 'numpy.ndarray'>"):
        array = np.array([array])
    #print("array", array)
    while(array.ndim > 1):
        array = array[0]
    return array

def appendBias(Neurons):
    if(type(Neurons) == type([])):
        Neurons.append(1)
        return Neurons
    #print("Neurons", Neurons.shape)
    shape = Neurons.shape[0] + 1
    NewArray = np.zeros(shape)
    for i in range(len(Neurons)):
        NewArray[i] = Neurons[i]
    NewArray[len(Neurons)] = 1
    return NewArray

def DeleteBias(Neurons):
    if(type(Neurons) == type([])):
        NewArray = Neurons[:-1]
        return NewArray
    shape = Neurons.shape[0] - 1
    NewArray = np.zeros(shape)
    for i in range(shape):
        NewArray[i] = Neurons[i]
    return NewArray


class Neural_Network:
    def __init__(self, input_number, hidden_layers, output_number):
        self.LearningRate = 0.7
        self.Moment = 0.2
        self.input_number = input_number
        self.hidden_layers = hidden_layers
        self.output_number = output_number
        number_of_weights = len(hidden_layers) + 1
        self.layers = [input_number] + hidden_layers
        self.layers.append(output_number)
        self.weights = list()
        self.deltas = list()
        for i in range(number_of_weights):
            self.deltas.append(np.zeros((self.layers[i] + 1, self.layers[i + 1])))
            self.weights.append(np.random.random((self.layers[i] + 1, self.layers[i + 1])))
            #print(i, ": ", self.weights[i])
        #print(self.weights)

    def SetLearningRate(self, LearningRate):
        self.LearningRate = LearningRate

    def SetMoment(self, Moment):
        self.Moment = Moment

    def LoadWeights(self, weights):
        self.weights = weights

    def Predict(self, inputValues):
        self.NeuronsValues = []
        inputValues = np.array(inputValues)
        inputValues = appendBias(inputValues)
        self.NeuronsValues.append(inputValues)
        inputValues = [inputValues]
        #print("inputValues", inputValues)
        #print("self.weights[0]", self.weights[0])
        CurrentNeurons = np.dot(inputValues, self.weights[0])
        #print("CurrentNeurons", CurrentNeurons)
        CurrentNeurons = sigmoid(CurrentNeurons)
        #print("sigmoid(CurrentNeurons)", CurrentNeurons)
        CurrentNeurons = ConvertToVector(CurrentNeurons)
        CurrentNeurons = appendBias(CurrentNeurons)
        self.NeuronsValues.append(CurrentNeurons)
        for i in range(1, len(self.weights)):
            CurrentNeurons = np.dot([CurrentNeurons], self.weights[i])
            CurrentNeurons = ConvertToVector(CurrentNeurons)
            CurrentNeurons = sigmoid(CurrentNeurons)
            if(i != len(self.weights) - 1):
                CurrentNeurons = appendBias(CurrentNeurons)
            self.NeuronsValues.append(CurrentNeurons)
        #print(self.NeuronsValues)
        return CurrentNeurons

    def DerivetiveOfErrorInRespectToOutput(self, Outputs, Target):
        CurrentGrad = np.zeros(Outputs.shape)
        for j in range(len(CurrentGrad)):
            CurrentGrad[j] = 2 * (Outputs[j] - Target[j])
        return CurrentGrad

    def DerivetiveOfSigmoid(self, Result):
        Grad = np.zeros(Result.shape)
        for j in range(len(Grad)):
            Grad[j] = Result[j] * (1 - Result[j])
        return Grad

    def DerivetiveOfSigmoidMultipliedByCurGrad(self, CurrentGrad, Result):
        SigmoidDerivetives = self.DerivetiveOfSigmoid(Result)
        for j in range(len(CurrentGrad)):
            CurrentGrad[j] = CurrentGrad[j] * SigmoidDerivetives[j]
        return CurrentGrad

    def Train(self, trainX, trainY):
        for i in range(len(trainX)):
            X = trainX[i]
            Y = trainY[i]
            X = ConvertToVector(X)
            Y = ConvertToVector(Y)
            CurrentGrad = self.DerivetiveOfErrorInRespectToOutput(self.Predict(X), Y)
            #print("self.NeuronsValues", self.NeuronsValues)
            #print("CurrentGradFun", CurrentGrad)
            for h in range(len(self.NeuronsValues) - 1, 0, -1):
                CurrentY = self.NeuronsValues[h]
                CurrentX = self.NeuronsValues[h - 1]
                #print("CurrentY", CurrentY)
                #print("CurrentGradFun2", CurrentGrad)
                CurrentGrad = self.DerivetiveOfSigmoidMultipliedByCurGrad(CurrentGrad, CurrentY)
                #print("self.weights[h - 1]", self.weights[h - 1])
                CurrentWeightGrad = np.zeros(self.weights[h - 1].shape)

                for j in range(len(CurrentWeightGrad)):
                    for k in range(len(CurrentWeightGrad[j])):
                        #print("CurrentWeightGrad", CurrentWeightGrad)
                        #print("CurrentGradFun5", CurrentGrad)
                        #print("CurrentX", CurrentX)
                        CurrentWeightGrad[j][k] = CurrentGrad[k] * CurrentX[j]
                        self.deltas[h - 1][j][k] =  -1 * CurrentWeightGrad[j][k] * self.LearningRate + self.deltas[h - 1][j][k] * self.Moment
                            #print("CurrentWeightGrad[j][k] = CurrentGrad[j] * CurrentX[k]: ", CurrentWeightGrad[j][k], CurrentGrad[k], CurrentX[j])
                            #print("j:", j, "k:", k)
                    #print("Fun3")
                #print("Finland: ", CurrentWeightGrad)
                #print("CurrentGrad = np.dot(CurrentGrad, self.weights[h - 1]): ", CurrentGrad, self.weights[h - 1])
                #print("CurrentGradFun3", CurrentGrad)
                #if(h < len(self.NeuronsValues) - 1):
                #CurrentGrad = DeleteBias(CurrentGrad)
                #print("CurrentGradFun4", CurrentGrad)
                CurrentGrad = (np.dot([CurrentGrad], self.weights[h - 1].transpose()))
                CurrentGrad = ConvertToVector(CurrentGrad)
                #print("CurrentGradFun4", CurrentGrad)
                CurrentGrad = DeleteBias(CurrentGrad)
                #print("CurrentGradFun3", CurrentGrad)
                for j in range(len(self.weights[h - 1])):
                    for k in range(len(self.weights[h - 1][j])):
                        self.weights[h - 1][j][k] = self.weights[h - 1][j][k] + self.deltas[h - 1][j][k]
                #print("self.weights[h - 1]: ", self.weights[h - 1])


            #print(self.weights)



#Neural_Network(2, [5, 3, 5], 2)
