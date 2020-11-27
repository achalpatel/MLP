from MLPv1 import *
import numpy as np
import pandas as pd
from io import StringIO 

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


def read_file(filepath):
    with open(filepath) as fp:
        for line in fp:
            print(line.strip())                   
    #    StringData = StringIO(line)

# read_file("dataset.txt")