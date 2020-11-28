from MLPv1 import *
import numpy as np
import pandas as pd
import re

def read_file(filepath) -> list:
    dataset = []        
    with open(filepath) as fp:
        for line in fp:                              
            rowLine = {}
            compiler = re.compile("\d+")
            dataList = compiler.findall(line)            
            rowLine['id'] = dataList[0]
            rowLine['attributes'] = dataList[1:-1]
            rowLine['label'] = dataList[-1]
            dataset.append(rowLine)
    return dataset

dataset = read_file("dataset.txt")
numberOfInputNodes = len(dataset[0]['attributes'])
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
g.singlePass(dataset[0])

# for node in g.inputLayer.nodes:
#     print("------------------------------------------------")
#     node.printData()

# for node in g.hiddenLayerList[0].nodes:
#     print("------------------------------------------------")
#     node.printData()