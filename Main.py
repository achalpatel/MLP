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
df = createDataFrame(dataset)
numberOfInputNodes = 10
numberOfHiddenNodes = 10
numberOfOutputNodes = 8
numberOfHiddenLayers = 1
learningRate = 0.01
g = Graph(learningRate)
g.createHiddenLayers(numberOfHiddenLayers)
g.createNodes(numberOfInputNodes, numberOfHiddenNodes, numberOfOutputNodes)

print("Input Nodes : ", len(g.inputLayer.nodes))
print("Output Nodes : ", len(g.outputLayer.nodes))
print("Hidden Nodes : ", len(g.hiddenLayerList[0].nodes))

g.connectInputToHidden()
g.connectHiddenToOutput()
g.calculateInitialWeights()
g.readDf(df)
g.runANN()
# Run prediction on Training set
print("Train Set Prediction-------------------------------------")
g.dfTest(g.trainDf)

# Run Prediction on Testing set
print("Test Set Prediction-------------------------------------")
g.dfTest(g.testDf)
values, count = np.unique(g.trainDf['label'], return_counts=True)
print(values, count)