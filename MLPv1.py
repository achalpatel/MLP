#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from enum import Enum

class Node:
    def __init__(self):
        self.inEdgeList = []
        self.outEdgeList = []
    
    def addInEdge(self, edge):
        self.inEdgeList.append(edge)
    
    def addOutEdge(self, edge):
        self.outEdgeList.append(edge)
    

class InputNode(Node):
    pass

class OutputNode(Node):
    def __init__(self):
        self.value = None


class HiddenNode(Node):
    def __init__(self):
        self.value = None


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

class Graph:
    def __init__(self):
        self.inputLayer = None
        self.outputLayer = None
        self.hiddenLayerList = []
        self.nodeList = []
    
    def createInputLayer(self):
        layer = InputLayer()
        self.inputLayer = layer
    
    def createOutputLayer(self):
        layer = OutputLayer()
        self.outputLayer = layer
    
    def createHiddenLayer(self):
        layer = HiddenLayer()        
        self.hiddenLayerList.append(layer)
    
    def createInputNode(self):
        node = InputNode()
        self.nodeList.append(node)
        return node
    
    def createOutputNode(self):
        node = OutputNode()
        self.nodeList.append(node)
        return node
    
    def createHiddenNode(self):
        node = HiddenNode()
        self.nodeList.append(node)
        return node
