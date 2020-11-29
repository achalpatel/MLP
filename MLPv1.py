#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random
import math

from pandas.core.frame import DataFrame


class Node:
    def __init__(self):
        self.inEdgeList = []
        self.outEdgeList = []
        self.value = None
    
    def addInEdge(self, edge):
        self.inEdgeList.append(edge)
    
    def addOutEdge(self, edge):
        self.outEdgeList.append(edge)

        
class InputNode(Node):
    pass

class OutputNode(Node):
    pass

class HiddenNode(Node):
    pass

class Edge:
    def __init__(self, fromNode, toNode):
        self.fromNode = fromNode
        self.toNode = toNode
        self.weight = None


class Layer:
    def __init__(self):
        self.nodes = []
    
    def addNode(self, node):
        self.nodes.append(node)
    def removeNode(self, node):
        self.nodes.remove(node)

class InputLayer(Layer):
    pass

class OutputLayer(Layer):
    pass

class HiddenLayer(Layer):
    pass

class Utility:
    def logistic(node: Node) -> float:
        sumValue = 0.0
        for edge in node.inEdgeList:            
            sumValue += edge.weight * edge.fromNode.value
        ans = 1/ (1 + math.exp(-sumValue))
        return ans

        

class Graph:
    def __init__(self):
        self.inputLayer = InputLayer()
        self.outputLayer = OutputLayer()
        self.hiddenLayerList = []
        self.nodeList = []
        self.edgeList = []
        self.df = None
        self.maxAttribList = []
        self.minAttribList = []
        self.targetList = []
    
    def createHiddenLayer(self):
        layer = HiddenLayer()        
        self.hiddenLayerList.append(layer)    
    
    def createInputNode(self):
        node = InputNode()
        self.inputLayer.addNode(node)
        self.nodeList.append(node)
        return node
    
    def createOutputNode(self):
        node = OutputNode()
        self.outputLayer.addNode(node)
        self.nodeList.append(node)
        return node
    
    def createHiddenNode(self, hiddenLayer: HiddenLayer):
        node = HiddenNode()
        hiddenLayer.addNode(node)
        self.nodeList.append(node)
        return node

    def createMultipleInputNodes(self, count : int):        
        for i in range(count):
            self.createInputNode()            

    def createMultipleOutputNodes(self, count : int):        
        for i in range(count):
            self.createOutputNode()            

    def createMultipleHiddenNodes(self, hiddenLayer : HiddenLayer, count : int):        
        for i in range(count):
            self.createHiddenNode(hiddenLayer)

    def connectInputToHidden(self):
        for fromNode in self.inputLayer.nodes:
            for toNode in self.hiddenLayerList[0].nodes:
                edge = Edge(fromNode, toNode)
                self.edgeList.append(edge)
                fromNode.addOutEdge(edge)
                toNode.addInEdge(edge)

    def connectHiddenToOutput(self):
        for fromNode in self.hiddenLayerList[0].nodes:
            for toNode in self.outputLayer.nodes:
                edge = Edge(fromNode, toNode)
                self.edgeList.append(edge)
                fromNode.addOutEdge(edge)
                toNode.addInEdge(edge)
            
    def calculateInitialWeights(self):
        for edge in self.edgeList:
            edge.weight = random.uniform(0, 1/10)
    
    def printEdgeData(self):
        for edge in self.edgeList:
            edge.printData()

    def readDf(self, dataframe : DataFrame):
        self.df = dataframe
        # Read Max, Min value of Each column and store in a List                
        for i in range(self.df.shape[1]-1):
            self.maxAttribList.append(pd.DataFrame.max(self.df.iloc[:,[i]]))
            self.minAttribList.append(pd.DataFrame.min(self.df.iloc[:,[i]]))

    def inputLayerFeed(self, row : DataFrame):        
        x = 0
        for node in self.inputLayer.nodes:
            normalized = float((row[x]-self.minAttribList[x])/(self.maxAttribList[x] - self.minAttribList[x]))
            node.value = normalized
            x+=1
        self.targetList = [0] * len(self.outputLayer.nodes)
        self.targetList[row[-1]-1] = 1        
    
    def softMax(self):
        denominator = self.softMaxDenominator()
        for node in self.outputLayer.nodes:
            node.value = math.exp(node.value)/denominator
            
    def softMaxDenominator(self) -> float:
        denominator = 0.0
        for node in self.outputLayer.nodes:
            denominator += math.exp(node.value)
        return denominator

    def mse(self):
        sumVal = 0.0
        m = len(self.outputLayer.nodes)
        for i in range(m):        
            sumVal += (self.targetList[i] - self.outputLayer.nodes[i].value) ** 2
        ans = sumVal/m
        # print("mse:",ans)
        return ans

    def singlePass(self):
        self.inputLayerFeed(self.df.iloc[0])
        for layer in self.hiddenLayerList:
            for node in layer.nodes:
                node.value = Utility.logistic(node)
        
        for node in self.outputLayer.nodes:
            node.value = Utility.logistic(node)
        self.softMax()
        self.mse()
        
        