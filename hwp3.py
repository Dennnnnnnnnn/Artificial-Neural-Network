import numpy as np
import matplotlib.pyplot as plt
import os
np.random.seed(42)

def SingleNeuron(X, W):
    WeightedSum = W[0]
    for i in range(len(X)):
        WeightedSum += X[i] * W[i + 1]
    return Sigmoid(WeightedSum)

def Sigmoid(x): #neuron's excitation function
    return 1/(1 + np.exp(-x)) #analytical expression

def SingleLayerNN(X, W): # a single layer NN (there could be any numbers of neurons in it)
    rows = len(W) #The weight's matrix has the dimensions those of the NN architecture( i. e., the number of neurons)
    st = len(W[0]) #and the number of input signals into a single neuron, (not used)  the dimensions of the W matrix are rows x st
    Y = []; # array to store the responses of all neurons
    for i in range(rows): # "running" through all neurons
        Y.append(SingleNeuron(X, W[i])) #each time an individual neuron is excited, each excitation is then stored in array Y
    return Y #returning all responses of all neurons (stored in a single array Y)

def BackPropagationTrainSet(XSet, W, NNTansSet): #the training of the single layer network, NNTansSet - true answer, used as "teacher"
    iterMax = 200    # predefined max iteration number
    eps = 1e-6 # desired accuracy in the training
    iterNr = 1 #(=epoch)
    derr = [] #the array of the partial derivatives of the cost function (gradient)
    step = 1 #step for updating the weights in negative direction of the gradient
    step_multiplier = 0.8 # multiplier for a step
    TrainSetSize = len(XSet) #the number of training values (how many values the network will have to "learn"
    TrainingSetCurrentInd = 0 #the current index of the "training" value
    X = XSet[TrainingSetCurrentInd] #the "training" values are stored in the array
    NNTans = NNTansSet[TrainingSetCurrentInd] #the true response of the network, i.e. "the teacher"
    X1 = [1] + X #constructing the array of input signals to perform the multplication with corresponding weigths ( 9 signals, but additional fictional signal to match the bias w0)
    for i in range(len(NNTans)): #initialization of the initial values for the derivatives
        derr.append(1e10) #since numpy is not used, assigning is performed for each value
    Continue = 1 # stop condition for the training
    MSE = [0]
    counterEpochMSE = 0 # aux variable
    thresholdMSE = 1e-2 # %1 - treshold for MSE progress along epochs
    MSEprogress = 20 # if 5 epochs in a row the MSE does not change more then thresholdMSE (1%) in respect to previous epochs MSE value, the training can be considered finished and terminated
    while Continue: # the training starts, all neurons are being trained at the same time, until one specific neuron learns how to active, while others learn to not activate, it is managable to control the learning duration for a individual neuron, but this is only possible and relevant in a SINGLE layer NN
        Continue = 0
        for TrainingSetCurrentInd in range(TrainSetSize): #the training scenarios are iterated through, during each specific neuron learns its pattern ("prescribed sample - the digit")
            X = XSet[TrainingSetCurrentInd]
            NNTans = NNTansSet[TrainingSetCurrentInd]
            X1 = [1] + X #input signals' array is adjusted for bias to conveniently perform dot product
            NNFans = SingleLayerNN(X, W) 
            #print(NNFans) #display of intermediate actuall responses
            SumErr = 0
            for i in range(len(NNTans)): #i-th neuron learns to activate or not activate depending on "teacher"
                SumErr += (NNTans[i] - NNFans[i])**2
                derr[i] = (NNTans[i] - NNFans[i]) * NNFans[i] * (1-NNFans[i]) #partial derivatives of the cost function for each neurons are calculated
                for j in range(len(W[0])): #i-th neuron's cost functon's partial derivative is caculated in respect to j-th weight
                    W[i][j] = W[i][j] + derr[i] * X1[j] * step #each weight is updated in the gradient's negative direction of the cost function
                    ABSderr =  max([abs(el) for el in derr]) #el - element: #the stopping condition will consist of the maximum absolute value derivative (when the max derivative is relatively small, the function can be terminated)
            ContinueLocal = (ABSderr > eps) and (iterNr < iterMax) #general condition for termination: max derivative or exceeded max iteration number
            Continue = Continue or ContinueLocal #refreshing "while"" condition
        MSE.append(SumErr / len(NNTans))
        if MSE[iterNr] > MSE[iterNr - 1]:
            step = step * step_multiplier # if current MSE value increases compared to previous epoch's MSE, the step is reduced
        if abs((MSE[iterNr] - MSE[iterNr - 1])/MSE[iterNr]) < thresholdMSE:
            #print("control ",MSE[iterNr], MSE[iterNr - 1])
            counterEpochMSE += 1
        else:
            counterEpochMSE = 0
        if counterEpochMSE == MSEprogress:
            Continue = False
        iterNr +=1 #an iteration (=epoch) has finished (during one iteration all (27 + 3) weights are changed only once in the negative direction of the gradient)
    plt.plot([el for el in range(len(MSE) - 1)], MSE[1:])
    plt.title('MSE vs epoch')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.show(block = True)
    return W #returning the weights of the trained network
LayerSizeArray = [100+1, 10]
W = []
for i in range(LayerSizeArray[1]):
    W.append([])
    for j in range(LayerSizeArray[0]):
        random_weight = 0
        while random_weight == 0:
            random_weight = np.random.uniform(0, 1)
        W[i].append(random_weight)

path = os.path.dirname(__file__)
trainpath = os.path.join(path, "matrices")
testpath = os.path.join(path, "test_cases")
xs = []
for num in range(0, 10):
    for count in range(1, 4):
        trfpath = os.path.join(trainpath, f"train{num}_{count}.csv")
        with open(trfpath, "r", encoding="utf-8") as filename:
            text = np.loadtxt(filename, delimiter=";", dtype=int)
            xs.append(text)

XSet = []
values = []
for i in range(0, 30):
    for j in range(0, 10):
        for k in range(0,10):
            values.append(int(xs[i][j][k]))
    XSet.append(values)
    values = []

NNTansSet = []
for i in range(0,10):
    teacher_vector = [0] * 10
    teacher_vector[i] = 1
    for _ in range(0,3):
        NNTansSet.append(teacher_vector)

W = BackPropagationTrainSet(XSet, W, NNTansSet)

for digit in range(0, 10):
    ttpath = os.path.join(testpath, f"{digit}_t.csv")
    with open(ttpath, "r") as Xfile:
        Xtext = np.loadtxt(Xfile, delimiter=";", dtype=int)
    TestSample = []
    for row in range(0, 10):
        for col in range(0, 10):
            TestSample.append(Xtext[row][col])

    Predictions = SingleLayerNN(TestSample, W)
    print(f"Testing for unseen digit {digit}:")
    for i, prediction in enumerate(Predictions):
        if i == digit:
            print(f"\033[44mDigit {i}: {prediction:.3f}\033[0m")            
        else:
            print(f"Digit {i}: {prediction:.3f}")