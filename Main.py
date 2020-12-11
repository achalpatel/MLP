from MLPv1 import *
import numpy as np
import pandas as pd
import re

def read_file(filepath) -> list:
    dataset = []        
    with open(filepath) as fp:
        for line in fp:                              
            compiler = re.compile("\d+")
            dataList = compiler.findall(line)                        
            row = list(map(int, dataList[1:]))            
            dataset.append(row)
    return dataset

def createDataFrame(dataset : list):
    df = pd.DataFrame(dataset, columns=['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','label'])    
    return df

dataset = read_file("dataset.txt")
validation_set = read_file("validation_set.txt")
df = createDataFrame(dataset)
validationDf = createDataFrame(validation_set)
numberOfInputNodes = 10
numberOfHiddenNodes = 11
numberOfOutputNodes = 8
numberOfHiddenLayers = 1
learningRate = 0.01
g = Graph(learningRate, numberOfInputNodes, numberOfHiddenNodes, numberOfOutputNodes, numberOfHiddenLayers)

print("Input Nodes : ", len(g.inputLayer.nodes))
print("Output Nodes : ", len(g.outputLayer.nodes))
print("Hidden Nodes : ", len(g.hiddenLayerList[0].nodes))

g.readDf(df)
g.validationDf = validationDf
# Print Initial Weights
print("Initial Weights:")
# g.printEdgeWeights()

# Train MLP
epochs = 15
g.runANN(epochs)

# print Finial weights
print("Final Weights : ")
# g.printEdgeWeights()

# Run prediction on Training set
print("Train Set Prediction-------------------------------------")
g.dfTest(g.trainDf)

# Run Prediction on Testing set
print("Test Set Prediction-------------------------------------")
g.dfTest(g.testDf)

# Run Prediction on Validation set
print("Validation Set Prediction-------------------------------------")
g.dfTest(g.validationDf)