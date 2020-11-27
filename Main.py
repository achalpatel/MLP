from MLPv1 import *
import numpy as np
import pandas as pd
import re

g = Graph()
g.createHiddenLayer()
g.createMultipleInputNodes(2)
g.createMultipleHiddenNodes(g.hiddenLayerList[0], 4)
g.createMultipleOutputNodes(1)

print("Graph total nodes : ",len(g.nodeList))
print("Input Nodes : ", len(g.inputLayer.nodes))
print("Output Nodes : ", len(g.outputLayer.nodes))
print("Hidden Nodes : ", len(g.hiddenLayerList[0].nodes))

g.connectInputToHidden()
g.connectHiddenToOutput()
for node in g.nodeList:
    print("------------------------------------------------")
    print(node)
    node.printData()


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
            print(rowLine)
            dataset.append(rowLine)
    return dataset

dataset = read_file("dataset.txt")
# print(dataset)