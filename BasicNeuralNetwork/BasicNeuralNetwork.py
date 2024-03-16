import random

learningRate = 0.5
totalIterations = 500

outputLayer = 0

targetOut_AND = [0, 0, 0, 1]
targetOut_OR = [0, 1, 1, 1]
targetOut_NAND = [1, 1, 1, 0]
targetOut_NOR = [1, 0, 0, 0]
targetOut_XOR = [0, 1, 1, 0]

def createNewMatrix(matrix): #Creates a blank matrix based on the given matrix size.
    newMatrix = [[]]*len(matrix)
    for i in range(0, len(matrix)):
        newMatrix[i] = [0]*len(matrix[0])
    return newMatrix

def matrixMultiplication(matrixA, matrixB): #Matrix Multiplication
    xA = len(matrixA)
    if(type(matrixA[0]) == type([])):
        yA = len(matrixA[0])
    else:
        yA = 1
    xB = len(matrixB)
    if(type(matrixB[0]) == type([])):
        yB = len(matrixB[0])
        resultingMatrix = [[0]*yB]*xA
    else:
        yB = 1
        resultingMatrix = [0]*xA
        
    if(yA == 1):
        resultingMatrix = [0]*yB
        for aX in range(0, xA):
            for bY in range(0, yB):
                resultingMatrix[bY] += matrixA[aX] * matrixB[aX][bY]

    else:
        resultingMatrix = [[]]*xA
        
        if(yB == 1):
            resultingMatrix = [0]*xA
        else:
            for i in range(0, xA):
                resultingMatrix[i] = [0]*yB
                
        for aX in range(0, xA):
            for aY in range(0, yA):
                if(yB == 1):
                    resultingMatrix[aX] += matrixA[aX][aY] * matrixB[aY]
                else:
                    for bY in range(0, yB):
                        resultingMatrix[aX][bY] += matrixA[aX][aY] * matrixB[aY][bY]
                    
    return resultingMatrix
    
def dotProduct(matrixA, matrixB): #Dot Product of two matrices
    resultingProduct = 0
    for x in range(0, len(matrixA)):
        resultingProduct += matrixA[x] * matrixB[x]
    
    return resultingProduct

def scalar(number, array): #Scalar Multiplication
    newArray = []
    if(type(array[0]) == type([])):
        newArray = createNewMatrix(array)

        for x in range(0, len(array)):
            for y in range(0, len(array[0])):
                newArray[x][y] = number * array[x][y]
    else:
        for x in range(0, len(array)):
            newArray[x] = array[x] * number
        
    return newArray
def matrixAddToElements(number, array): #All emenents become number + element
    newArray = []
    if(type(array[0]) == type([])):
        newArray = createNewMatrix(array)

        for x in range(0, len(array)):
            for y in range(0, len(array[0])):
                newArray[x][y] = number + array[x][y]
    else:
        for x in range(0, len(array)):
            newArray[x] = array[x] + number
        
    return newArray
def matrixNumberPowerElement(number, matrix): #All elements become number ^ element
    newMatrix = createNewMatrix(matrix)
    for x in range(0, len(matrix)):
        for y in range(0, len(matrix[0])):
            newMatrix[x][y] = number ** matrix[x][y]
            
    return newMatrix
def matrixInverseElement(matrix): #All elements become 1 / element
    newMatrix = createNewMatrix(matrix)
    for x in range(0, len(matrix)):
        for y in range(0, len(matrix[0])):
            newMatrix[x][y] = 1 / matrix[x][y]
            
    return newMatrix

def matrixSigmoid(matrix): #Take sigmoid of entire Matrix.
    return matrixInverseElement(matrixAddToElements(1, matrixNumberPowerElement(2.718281828459045, scalar(-1, matrix))))
def sigmoid(array): #Take sigmoid of number/array
    if(type(array) == type([])):
        if(type(array[0]) == type([])):
            newArray = [0]*len(array)
            for x in range(0, len(array)):
                newArray[x] = sigmoid(array[x])
        else:
            newArray = [0]*len(array)
            for x in range(0, len(array)):
                newArray[x] = 1.0 / (1.0 + (2.718281828459045 ** -array[x]))
    else:
        newArray = 1.0 / (1.0 + (2.718281828459045 ** -array))
           
    return newArray  

#Method that creates the neural network responsible for operations.
def neuralNetwork(targetOutput):
    inputLayer = [[0, 0], [0, 1], [1, 0], [1, 1]]
    global outputLayer
    global totalIterations
    global learningRate
    
    inputLayerWeights = [ # 3x2 grid -> 2 Inputs Layers, 2 Hidden Layers
        [random.random() - .5, random.random() - .5],
        [random.random() - .5, random.random() - .5],
        [random.random() - .5, random.random() - .5]]
    hiddenLayerWeights = [random.random() - .5, random.random() - .5, random.random() - .5]
    
    size = len(targetOutput)
    
    inputLayerWithBias = inputLayer
    for includeBias in range(0, len(inputLayer)):
        inputLayerWithBias[includeBias].insert(0,1)
        
    hiddenLayerWithBias = [0, 0, 0]
    deltaHidden = [0, 0, 0]
    
    for i in range(0, totalIterations):
        for j in range(0, size):
            hiddenLayerActivation = matrixMultiplication(inputLayerWithBias[j], inputLayerWeights)
            hiddenLayer = sigmoid(hiddenLayerActivation)
            
            #Bias
            hiddenLayerWithBias = [1, hiddenLayer[0], hiddenLayer[1]]
            outputLayer = sigmoid(dotProduct(hiddenLayerWithBias, hiddenLayerWeights))
            
            #Calculate error: 
            deltaOutput = targetOutput[j] - outputLayer
            
            #Update Layers
            for x in range(0, len(inputLayerWeights)):
                deltaHidden[x] =  (deltaOutput * hiddenLayerWeights[x]) * (hiddenLayerWithBias[x] * (1 - hiddenLayerWithBias[x])) #Calculate Error
                hiddenLayerWeights[x] = hiddenLayerWeights[x] + (learningRate * (deltaOutput * hiddenLayerWithBias[x])) #Update Hidden Weights (Gradient Descent)
                for y in range(0, len(inputLayerWeights[0])): #Update Input Weights
                    inputLayerWeights[x][y] = inputLayerWeights[x][y] + (learningRate * deltaHidden[y + 1] * inputLayerWithBias[j][x])

    #Training is complete. Show results:
    hiddenLayer = matrixSigmoid(matrixMultiplication(inputLayerWithBias, inputLayerWeights))
    hiddenLayerWithBias = hiddenLayer
    for x in range(0, len(hiddenLayer)):
        hiddenLayerWithBias[x].insert(0,1)
    outputLayer = sigmoid(matrixMultiplication(hiddenLayerWithBias, hiddenLayerWeights))
    totalCost = 0
    for i in range(0, 4): 
        totalCost += (1/2) * ((outputLayer[i] - targetOutput[i]) ** 2)
    cost = totalCost / 4 
    
    hiddenLayer = sigmoid(matrixMultiplication(inputLayerWithBias, inputLayerWeights))
        
    hiddenLayerWithBias = hiddenLayer
    for includeBias in range(0, len(hiddenLayer)):
        hiddenLayerWithBias[includeBias].insert(0,1)
        
    actualOutput = sigmoid(matrixMultiplication(hiddenLayerWithBias, hiddenLayerWeights))
    
    print("Total cost: " + str(cost))
    print("Target:\t\t" + str(targetOutput))
    showResult = [0]*len(actualOutput)
    for z in range(0, len(actualOutput)): #Activation
        if(actualOutput[z] > 0.5):
            showResult[z] = 1
        else:
            showResult[z] = 0     
    print("End Result:\t" + str(showResult) + "\n")
    
    
#Creates the neural networks based on targets
neuralNetwork(targetOut_AND)
neuralNetwork(targetOut_OR)
neuralNetwork(targetOut_NAND)
neuralNetwork(targetOut_NOR)
neuralNetwork(targetOut_XOR)

