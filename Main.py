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
            # print(row)
            dataset.append(row)
    return dataset

def createDataFrame(dataset : list):
    df = pd.DataFrame(dataset, columns=['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','label'])    
    return df


dataset = read_file("dataset.txt")
df = createDataFrame(dataset)
numberOfInputNodes = 10
numberOfOutputNodes = 8
g = Graph()
g.createHiddenLayer()
g.createMultipleInputNodes(numberOfInputNodes)
g.createMultipleHiddenNodes(g.hiddenLayerList[0], 10)
g.createMultipleOutputNodes(numberOfOutputNodes)

print("Graph total nodes : ",len(g.nodeList))
print("Input Nodes : ", len(g.inputLayer.nodes))
print("Output Nodes : ", len(g.outputLayer.nodes))
print("Hidden Nodes : ", len(g.hiddenLayerList[0].nodes))

g.connectInputToHidden()
g.connectHiddenToOutput()
g.calculateInitialWeights()
g.readDf(df)
g.singlePass()

for node in g.inputLayer.nodes:    
    print("input node value:",node.value)

print("------------------------------------------------")
for node in g.hiddenLayerList[0].nodes:    
    print("hidden node value:",node.value)

print("------------------------------------------------")
for node in g.outputLayer.nodes:    
    print("output node value:",node.value)

g.updateHiddenToOutputWeights()
g.updateInputToHiddenWeights()
g.singlePass()
print("------------------------------------------------")
for node in g.inputLayer.nodes:    
    print("input node value:",node.value)

print("------------------------------------------------")
for node in g.hiddenLayerList[0].nodes:    
    print("hidden node value:",node.value)

print("------------------------------------------------")
for node in g.outputLayer.nodes:    
    print("output node value:",node.value)
