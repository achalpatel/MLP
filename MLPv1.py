#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from random import *
import math


class Node:
    def __init__(self):
        self.inEdgeList = []
        self.outEdgeList = []
        self.value = None
    
    def addInEdge(self, edge):
        self.inEdgeList.append(edge)
    
    def addOutEdge(self, edge):
        self.outEdgeList.append(edge)

    def printData(self):
        print("value : ",self.value)
        # for edge in self.inEdgeList:
            # print("In - edge : ",edge)
        # for edge in self.outEdgeList:
            # print("Out - edge : ",edge)
        
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
    
    def printData(self):
        print("From Node : ", self.fromNode)
        print("To Node : ", self.toNode)
        print("Weight : ", self.weight)

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
            edge.weight = random()
    
    def printEdgeData(self):
        for edge in self.edgeList:
            edge.printData()

    def inputLayerFeed(self, values:list):
        for i in range(len(values)):
            self.inputLayer.nodes[i].value = int(values[i])

    def singlePass(self, rowLine: dict):
        attributes = rowLine['attributes']
        self.inputLayerFeed(attributes)
        for layer in self.hiddenLayerList:
            for node in layer.nodes:
                node.value = Utility.logistic(node)
                print(node.printData())
        