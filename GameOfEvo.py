from logging.handlers import BaseRotatingHandler
import os

from numpy import true_divide
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random as r
from re import L
import math as m
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from ast import literal_eval as make_tuple
import cv2
import numpy as np
from screeninfo import get_monitors

class Node:
    def __init__(self, index, typ, on = False, value = 0):
        self.index = index
        self.typ = typ
        self.on = on;
        self.value = value
        self.bias = value
    
    def compute(self, brain, inputs):
            match self.typ:
                case "inp":
                    if self.index < len(inputs): self.value = inputs[self.index]
                    else: self.value = 0
                case _:
                    if self.isOn():
                        for (male, female, properties) in brain.edges.data():
                            if female == self and male.isOn(): self.value += properties["weight"] * male.value
    
    def noConnections(self, brain):
        for (male,female) in brain.edges():
            if self == male or self == female:
                return False
        return True
    
    def isOn(self):
        return self.on

    def turnOn(self):
        self.on = True

    def turnOff(self):
        self.on = True
    
    def notDupe(self, brain):
        for node in brain:
            if self.index == node.index and self.typ == node.typ:
                return False
        return True
    
    def toReadable(self):
        if self.typ == "inp":
            return inputt(self.index)
        if self.typ == "mid":
            return "m" + str(self.index)
        if self.typ == "out":
            return outputt(self.index)
    
    def toReadableValue(self, value):
        if self.typ == "inp":
            return inputt(self.index) + ": " + str(value)
        if self.typ == "mid":
            return "m" + str(self.index) + ": " + str(value)
        if self.typ == "out":
            return outputt(self.index) + ": " + str(value)
    
    def typeindex(self):
        match self.typ:
            case "inp":
                return 0
            case "mid":
                return 1
            case "out":
                return 2
            
                    
def nodeMax(index_values):
    (maxI, maxV) = index_values[0]
    for (index, value) in index_values:
        if value > maxV: (maxI, maxV) = (index, value)
    return maxI

def onlyPositions(i):
    match i:
        case 0:
            return "1X"
        case 1:
            return "1Y"
        case 2:
            return "2X"
        case 3:
            return "2Y"
        case 4:
            return "3X"
        case 5:
            return "3Y"

def inputt(i):
    match i:
        case 0:
            return "T"
        case 1:
            return "X"
        case 2:
            return "Y"
        case 3:
            return "LM"
        case 4:
            return "BX"
        case 5:
            return "BY"
        case 6:
            return "1X"
        case 7:
            return "1Y"
        case 8:
            return "2X"
        case 9:
            return "2Y"
        case 10:
            return "3X"
        case 11:
            return "3Y"
        case 12:
            return "1"
        case 13:
            return "2"
        case 14:
            return "3"
        case 15:
            return "C"

def outputt(i):
    match i:
        case 0:
            return "<"
        case 1:
            return "<v"
        case 2:
            return "v"
        case 3:
            return "v>"
        case 4:
            return ">"
        case 5:
            return "^>"
        case 6:
            return "^"
        case 7:
            return "<^"
        case 8:
            return "~"
        case 9:
            return "K"
        case 10:
            return "O"

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

class Brain:
    def __init__(self, inputs, middle, outputs, bias_max = 10, neuroactivity = 66, connectivity = 10, weight_max = 10, obstacle = False):
        self.inputs = inputs
        self.middle = middle
        self.outputs = outputs
        self.bias_max = bias_max
        self.neuroactivity = neuroactivity
        self.connectivity = connectivity
        self.weight_max = weight_max
        self.obstacle = obstacle
        if not obstacle:
            brain = nx.DiGraph()
            brain.add_nodes_from([Node(i, "inp") for i in range(inputs)])
            brain.add_nodes_from([Node(i, "mid") for i in range(middle)]) #, value = r.randrange(bias_max)
            brain.add_nodes_from([Node(i, "out") for i in range(outputs)]) #, value = r.randrange(bias_max)

            #Activate nodes
            for node in brain:
                if r.randrange(100) + 1 < neuroactivity: node.turnOn()
            
            #Connect nodes
            for male in brain:
                if male.isOn() and male.typ != "out":
                    for female in brain:
                        if r.randrange(100) + 1 < connectivity * 2 and female.typ != "inp" and female.isOn(): brain.add_edge(male, female, weight = r.randrange(weight_max)) 
            self.brain = brain   
        else: self.brain = None
    
    def think(self, inputs):
        if self.obstacle: return 10
        for node in self.brain:
            node.compute(self.brain, inputs)
        outputs = []
        for node in self.brain:
            if node.typ == "out" and node.isOn(): outputs.append((node.index, node.value))
        conclusion = nodeMax(outputs) if len(outputs) > 0 else 10
        for node in self.brain:
            node.value = node.bias
        return conclusion
    
    def thoughtsOn(self, inputs):
        for node in self.brain:
            node.compute(self.brain, inputs)
        self.drawThoughts()
        for node in self.brain:
            node.value = node.bias

    def add(self, other):
        for node in other.brain:
            if node.notDupe(self.brain): self.brain.add_node(node)
        for edge in other.brain.edges.data():
            self.brain.add_edges_from([edge])
    
    def copy(self):
        new = Brain(self.inputs, self.middle, self.outputs, self.bias_max, self.neuroactivity, self.connectivity, self.weight_max)
        new.brain = self.brain.copy()
        return new

    def pureOffspring(self, world):
        babyBrain = self.copy()

        for node in babyBrain.brain:
            if r.randrange(100) + 2 ** world.gen < self.neuroactivity and not node.isOn(): node.turnOn()
        
        for node in babyBrain.brain:
            if r.randrange(100) + 2 ** world.gen < self.neuroactivity and node.isOn() and node.noConnections(babyBrain.brain): node.turnOff()

        for male in babyBrain.brain:
            if male.isOn() and male.typ != "out":
                for female in babyBrain.brain:
                    if r.randrange(100) + 2 ** world.gen <= self.connectivity and female.typ != "inp" and ((male, female) not in babyBrain.brain.edges) and female.isOn(): babyBrain.brain.add_edge(male, female, weight = r.randrange(self.weight_max)) 

        for (male, female) in babyBrain.brain.edges:
            if self.brain.has_edge(male,female) and r.randrange(100) + 2 ** world.gen <= self.connectivity / 2: self.brain.remove_edge(male, female)

        for (male, female, properties) in babyBrain.brain.edges.data():
            if r.randrange(100) + 2 ** world.gen < 50: properties["weight"] = properties["weight"] + r.randrange(-int(self.weight_max/4), int(self.weight_max/4))
        return babyBrain

    def mixedOffspring(self, other, world):
        babyBrain = self.copy()
        babyBrain.add(other)

        for node in babyBrain.brain:
            if r.randrange(100) + 2 ** world.gen < self.neuroactivity and not node.isOn(): node.turnOn()
        
        for node in babyBrain.brain:
            if r.randrange(100) + 2 ** world.gen < self.neuroactivity and node.isOn() and node.noConnections(babyBrain.brain): node.turnOff()
        
        for male in babyBrain.brain:
            if male.isOn() and male.typ != "out":
                for female in babyBrain.brain:
                    if r.randrange(100) + 2 ** world.gen <= self.connectivity and female.typ != "inp" and ((male, female) not in babyBrain.brain.edges) and female.isOn(): babyBrain.brain.add_edge(male, female, weight = r.randrange(self.weight_max)) 

        for (male, female) in babyBrain.brain.edges:
            if self.brain.has_edge(male,female) and r.randrange(100) + 2 ** world.gen <= self.connectivity / 2: self.brain.remove_edge(male, female)

        for (male, female, properties) in babyBrain.brain.edges.data():
            if r.randrange(100) + 2 ** world.gen < 50: properties["weight"] = properties["weight"] + r.randrange(-int(self.weight_max/4), int(self.weight_max/4))
        return babyBrain

    def draw(self):
        readableBrain = nx.DiGraph()

        colorMap = []
        for node in self.brain:
            if node.isOn():
                new = node.toReadable()
                colorMap.append(node.bias)
                match node.typ:
                    case "inp":
                        index = node.index
                    case "mid":
                        index = node.index + int((self.inputs - self.middle)/2)
                    case "out":
                        index = node.index + int((self.inputs - self.outputs)/2)
                pos = (node.typeindex(), index)
                #print(node.toReadable() + " : " + str(pos))
                readableBrain.add_node(new, pos=pos)
        while (len(colorMap) > len(readableBrain)): colorMap.remove(colorMap[-1])

        edgeColors = []
        for (male, female, properties) in self.brain.edges.data():
            if male.isOn() and female.isOn():
                edgeColors.append(properties["weight"])
                readableBrain.add_edge(male.toReadable(),female.toReadable(), weight = int(properties["weight"]))

        weights = nx.get_edge_attributes(readableBrain, "weight")
        pos = nx.get_node_attributes(readableBrain, 'pos')
        nx.draw(readableBrain, pos = pos, with_labels = True, node_color = colorMap, edge_color = edgeColors, edge_cmap = plt.cm.Greys, cmap = plt.cm.Blues)
        #nx.draw_networkx_edge_labels(readableBrain, pos, edge_labels=weights)
        plt.show()
    
    def confidence(self, node):
        if node.typ == "inp": return int(node.value * 100)
        res = 0;
        for other in self.brain:
            if other.isOn() and other.typ == node.typ:
                
                res += other.value
        if res == 0: 
            if node.value != 0: return 100
            else: return 0
        return int((node.value / res) * 100)

    def drawThoughts(self):
        readableBrain = nx.DiGraph()

        colorMap = []
        for node in self.brain:
            if node.isOn():
                colorMap.append(self.confidence(node)/100)
                new = node.toReadableValue(self.confidence(node))
                match node.typ:
                    case "inp":
                        index = node.index
                    case "mid":
                        index = node.index + int((self.inputs - self.middle)/2)
                    case "out":
                        index = node.index + int((self.inputs - self.outputs)/2)
                pos = (node.typeindex(), index)
                #print(node.toReadable() + " : " + str(pos))
                readableBrain.add_node(new, pos=pos)
        while (len(colorMap) > len(readableBrain)): colorMap.remove(colorMap[-1])
        #print("nodes: " + str(len(readableBrain)))
        #print("colors: " + str(len(colorMap)))

        edgeColors = []
        for (male, female, properties) in self.brain.edges.data():
            if male.isOn() and female.isOn():
                edgeColors.append(properties["weight"])
                readableBrain.add_edge(male.toReadableValue(self.confidence(male)),female.toReadableValue(self.confidence(female)), weight = int(properties["weight"]))

        weights = nx.get_edge_attributes(readableBrain, "weight")
        pos = nx.get_node_attributes(readableBrain, 'pos')
        nx.draw(readableBrain, pos = pos, with_labels = True, node_color = colorMap, edge_color = edgeColors, edge_cmap = plt.cm.Greys, cmap = plt.cm.Reds)
        #nx.draw_networkx_edge_labels(readableBrain, pos, edge_labels=weights)
        plt.pause(0.005)
        plt.clf()
        plt.show(block = False)

    def printBrain(self):
        for (male, female, properties) in self.brain.edges.data():
            if(male.isOn() and female.isOn()):
                print(male.typ + " #" + str(male.index) + " --" + str(properties["weight"]) + "--> " + female.typ + " #" + str(female.index), end = " ")
    
    def play(self, video, dude):
        if(dude == None):
            print("Not a Dude")
            return
        i = 0
        for (frame, step) in zip(video, dude.inputlist):
            for inputs in step:
                #print(dude.inputlist[i])
                self.thoughtsOn(inputs)
                
                #plt.clear()
                i += 1
            (x, y) = (get_monitors()[0].width - frame.shape[1] - 5, 25)
            cv2.imshow("gen: " + str(gen), frame)
            cv2.moveWindow("gen: " + str(gen), x, y)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        plt.close()

class obstacle:
    def __init__(self, xstart, xend, ystart, yend):
        self.xstart = xstart
        self.xend = xend
        self.ystart = ystart
        self.yend = yend
    
    def create(self, world):
        for i in range(self.xend - self.xstart):
            x = i + self.xstart
            for j in range(self.yend - self.ystart):
                y = j + self.ystart
                block = minidude(world, isObstacle = True)
                block.pos = (x, y)
                block.species = 0
                block.strength = 0
                world.population.append(block)
                #print("block placed")

#def randBrain():
#    Brain = keras.models.Sequential(name = "Brain")
#    Brain.add(keras.layers.Input(shape = (1,)))
#    Brain.add(keras.layers.Dense(6, input_shape = (15,), kernel_initializer = 'random_normal', bias_initializer = 'random_normal'))
#    Brain.add(keras.layers.Dense(11, activation = "softmax", kernel_initializer = 'random_normal', bias_initializer = 'random_normal'))
#    Brain.compile(
#    optimizer=keras.optimizers.RMSprop(),  # Optimizer
#    # Loss function to minimize
#    loss=keras.losses.SparseCategoricalCrossentropy(),
#    # List of metrics to monitor
#    metrics=[keras.metrics.SparseCategoricalAccuracy()],
#
#    return Brain      

def directionChar(i):
    match i:
        case 0:
            return "<"
        case 1:
            return "/"
        case 2:
            return "V"
        case 3:
            return "\\"
        case 4:
            return ">"
        case 5:
            return "/"
        case 6:
            return "^"
        case 7:
            return "\\"
        case 8:
            return "~"
        case 9:
            return "X"
        case 10:
            return "O"

def Direction(i):
    match i:
        case 0:
            return "left"
        case 1:
            return "up-left"
        case 2:
            return "up"
        case 3:
            return "up-right"
        case 4:
            return "right"
        case 5:
            return "down-right"
        case 6:
            return "down"
        case 7:
            return "down-left"
        case 8:
            return "Sex"
        case 9:
            return "Kill"
        case 10:
            return "Stay"

def randPos(world):
    return (world.spawnx + r.randrange(world.spawnlength - world.spawnx), world.spawny + r.randrange(world.spawnheight - world.spawny))

def sigmoid(x):
    sig = 1 / (1 + m.exp(-x))
    return sig

def int2color(i):
    return ((i >> 16)& 0x000000FF, (i >> 8)& 0x000000FF, i & 0x000000FF)

def int2bw(i, length):
    return (i * (255 / length), i * (255 / length), i * (255 / length))

def colorshift(i):
    match r.randrange(3):
        case 0:
            return i + 1
        case 1: 
            return i + (2**8)
        case 2: 
            return i + (2**18)

def colorblend(i, j):
    (ir, ig, ib) = int2color(i)
    (jr, jg, jb) = int2color(j)
    return (int((ir + jr)/2) << 16) | (int((ig + jg)/2) << 8) | int((ib + jb)/2)

class minidude:
    def __init__(self, world, isObstacle = False):
        self.Brain = Brain(16, 8, 11, obstacle = isObstacle)
        self.world = world
        self.pos = randPos(world)
        self.birthpos = self.pos
        self.lastmove = 0
        self.time = 0
        self.species = 0
        while(self.species == 0):
            self.species = r.randrange(2**32)
        self.babies = 0
        self.actions = 0
        self.action_limit = world.action_limit
        self.kills = 0
        self.strength = r.randrange(100)
        self.wantedChildren = r.randrange(world.birthrate) + 1
        self.isObstacle = isObstacle
        self.inputlist = []
    
    def pos(self):
        return self.pos

    def xpos(self):
        (xpos, _) = self.pos
        return xpos

    def ypos(self):
        (_, ypos) = self.pos
        return ypos

    def canMove(self, world, newPos):
        for dude in world.population:
            if not(dude is None) and dude.pos == newPos:
                return False
        return self.validPos(newPos)
        #for dude in world.population:
        #    if not(dude is None) and dude.pos == newPos:
        #       if dude.species == self.species: return False
        #       world.population.remove(dude)
        #(x, y) = newPos;
        #return (0 <= x and x < world.length and 0 <= y and y < world.height)
    
    
    def threeClosest(self, world):
        differences = []
        for otherdude in world.population:
            if not (otherdude is None) and otherdude != self: differences.append(m.sqrt((otherdude.xpos() - self.xpos()) ** 2 + (otherdude.ypos() - self.ypos()) ** 2))
            else: differences.append(world.length*world.height)

        sorte = differences.copy()
        sorte.sort()

        Closest = []
        for i in range(min(3, len(world.population))):
            Closest.append(world.population[differences.index(sorte[i])])
        
        return Closest
    
    def isNear(self, dude):
        if (abs(self.xpos() - dude.xpos()) == 1) and (abs(self.ypos() - dude.ypos()) == 1): return True
        if self.xpos() == dude.xpos() and (abs(self.ypos() - dude.ypos()) == 1): return True
        if (abs(self.xpos() - dude.xpos()) == 1) and self.ypos() == dude.ypos(): return True

    def isClose(self):
        for dude in self.world.population:
            if self.isNear(dude): return True
        return False

    def validPos(self, pos):
        (x, y) = pos
        return (0 <= x and x < self.world.length) and (0 <= y and y < self.world.height)

    def available_position(self, dude):
        spots = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        availabe = []
        for spot in spots:
            newspot = tuple(map(sum, zip((self.xpos(), self.ypos()), spot)))
            if(self.validPos(newspot) and (not(newspot == dude.pos or newspot == self.pos))): return newspot
        newspot = dude.pos
        return newspot

    def baby(self, world):
        baby = minidude(world)
        baby.Brain = self.Brain.pureOffspring(world)
        baby.pos = randPos(world)
        baby.species = colorshift(self.species)
        baby.wantedChildren = self.wantedChildren
        return baby
    
    def fornicate(self, dude, world):
        baby = minidude(world)
        baby.Brain = self.Brain.mixedOffspring(dude.Brain, world)
        baby.species = colorblend(self.species, dude.species)
        baby.pos = dude.pos
        baby.wantedChildren = self.wantedChildren
        self.world.births += 1
        return baby

    def act(self, world):
        if self.isObstacle: return
        if len(self.inputlist) == 0: self.inputlist.append([])
        self.actions += 1
        num_actions = 11
        Closest = self.threeClosest(world)

        Races = []
        for dude in Closest:
            Races.append(self.species == dude.species)
        for i in range(max(3 - len(Races), 0)):
            Races.append(0)

        Positions = []
        for dude in Closest:
            Positions.append(abs(dude.xpos() - self.xpos()) / world.length) 
            Positions.append(abs(dude.ypos() - self.ypos()) / world.height)
        for i in range(max(3 - len(Positions), 0)):
            Positions.append(0)
            Positions.append(0)

        (bx, by) = self.birthpos
        stats = [self.time / world.lifespan, self.xpos() / world.length, self.ypos() / world.height, self.lastmove / num_actions, bx / world.length, by / world.height]
        ins =  stats + Positions + Races + [int(self.isClose())] #
        #print(self.inputlist)
        if len(self.inputlist) == 1:
            self.inputlist[0].append(ins)
        else: 
            self.inputlist[-1].append(ins)
        #ins = tf.constant([self.time])
        #print(confidence)
        direction = self.Brain.think(ins)
        #self.Brain.printBrain()
        #print(chr((dude.species % 94) + 33) + ": direction: " + Direction(direction))
        match direction:
            case 0:
                (x, y) = self.pos
                newPos = tuple(map(sum, zip((x, y), (-1, 0))))
                newXPos = tuple(map(sum, zip((x, y), (-1, 0))))
                newYPos = tuple(map(sum, zip((x, y), (0, 0))))
                if(self.canMove(world, newPos)): self.pos = newPos
                elif(self.canMove(world, newXPos)): self.pos = newXPos
                elif(self.canMove(world, newYPos)): self.pos = newYPos
                elif self.actions < self.action_limit: self.act(world)
            case 1:
                (x, y) = self.pos
                newPos = tuple(map(sum, zip((x, y), (-1, 1))))
                newXPos = tuple(map(sum, zip((x, y), (-1, 0))))
                newYPos = tuple(map(sum, zip((x, y), (0, 1))))
                #newPos = tuple(map(sum, zip((x, y), (-1, 0))))
                #newXPos = tuple(map(sum, zip((x, y), (-1, 0))))
                #newYPos = tuple(map(sum, zip((x, y), (0, 0))))
                if(self.canMove(world, newPos)): self.pos = newPos
                elif(self.canMove(world, newXPos)): self.pos = newXPos
                elif(self.canMove(world, newYPos)): self.pos = newYPos
                elif self.actions < self.action_limit: self.act(world)
            case 2:
                (x, y) = self.pos
                newPos = tuple(map(sum, zip((x, y), (0, 1))))
                newXPos = tuple(map(sum, zip((x, y), (0, 0))))
                newYPos = tuple(map(sum, zip((x, y), (0, 1))))
                if(self.canMove(world, newPos)): self.pos = newPos
                elif(self.canMove(world, newXPos)): self.pos = newXPos
                elif(self.canMove(world, newYPos)): self.pos = newYPos
                elif self.actions < self.action_limit: self.act(world)
            case 3:
                (x, y) = self.pos
                newPos = tuple(map(sum, zip((x, y), (1, 1))))
                newXPos = tuple(map(sum, zip((x, y), (1, 0))))
                newYPos = tuple(map(sum, zip((x, y), (0, 1))))
                #newPos = tuple(map(sum, zip((x, y), (0, 1))))
                #newXPos = tuple(map(sum, zip((x, y), (0, 0))))
                #newYPos = tuple(map(sum, zip((x, y), (0, 1))))
                if(self.canMove(world, newPos)): self.pos = newPos
                elif(self.canMove(world, newXPos)): self.pos = newXPos
                elif(self.canMove(world, newYPos)): self.pos = newYPos
                elif self.actions < self.action_limit: self.act(world)
            case 4:
                (x, y) = self.pos
                newPos = tuple(map(sum, zip((x, y), (1, 0))))
                newXPos = tuple(map(sum, zip((x, y), (1, 0))))
                newYPos = tuple(map(sum, zip((x, y), (0, 0))))
                if(self.canMove(world, newPos)): self.pos = newPos
                elif(self.canMove(world, newXPos)): self.pos = newXPos
                elif(self.canMove(world, newYPos)): self.pos = newYPos
                elif self.actions < self.action_limit: self.act(world)
            case 5:
                (x, y) = self.pos
                newPos = tuple(map(sum, zip((x, y), (1, -1))))
                newXPos = tuple(map(sum, zip((x, y), (1, 0))))
                newYPos = tuple(map(sum, zip((x, y), (0, -1))))
                #newPos = tuple(map(sum, zip((x, y), (1, 0))))
                #newXPos = tuple(map(sum, zip((x, y), (1, 0))))
                #newYPos = tuple(map(sum, zip((x, y), (0, 0))))
                if(self.canMove(world, newPos)): self.pos = newPos
                elif(self.canMove(world, newXPos)): self.pos = newXPos
                elif(self.canMove(world, newYPos)): self.pos = newYPos
                elif self.actions < self.action_limit: self.act(world)
            case 6:
                (x, y) = self.pos
                newPos = tuple(map(sum, zip((x, y), (0, -1))))
                newXPos = tuple(map(sum, zip((x, y), (0, 0))))
                newYPos = tuple(map(sum, zip((x, y), (0, -1))))
                if(self.canMove(world, newPos)): self.pos = newPos
                elif(self.canMove(world, newXPos)): self.pos = newXPos
                elif(self.canMove(world, newYPos)): self.pos = newYPos
                elif self.actions < self.action_limit: self.act(world)
            case 7:
                (x, y) = self.pos
                newPos = tuple(map(sum, zip((x, y), (-1, -1))))
                newXPos = tuple(map(sum, zip((x, y), (-1, 0))))
                newYPos = tuple(map(sum, zip((x, y), (0, -1))))
                #newPos = tuple(map(sum, zip((x, y), (0, -1))))
                #newXPos = tuple(map(sum, zip((x, y), (0, 0))))
                #newYPos = tuple(map(sum, zip((x, y), (0, -1))))
                if(self.canMove(world, newPos)): self.pos = newPos
                elif(self.canMove(world, newXPos)): self.pos = newXPos
                elif(self.canMove(world, newYPos)): self.pos = newYPos
                elif self.actions < self.action_limit: self.act(world)
            case 8:
                copy = world.population.copy()
                for dude in copy:
                    if self.isNear(dude) and world.births < world.birth_limit and dude.babies < world.baby_limit and not dude.Brain.obstacle: 
                        world.population.append(self.fornicate(dude, world))
                        dude.babies += 1
            case 9:
                for dude in world.population.copy():
                    #if self.isNear(dude) and dude.strength >= self.strength: print("not strong enough")
                    if self.isNear(dude) and dude.strength < self.strength: 
                        #print("kill")
                        world.population.remove(dude)
                        self.world.kills += 1
                        self.kills += 1
            case 10:
                if self.actions < self.action_limit: self.act(world)
        self.lastmove = direction


def populate(init_pop, world, obxstart, obxend, obystart, obyend):
    obstacle(obxstart, obxend, obystart, obyend).create(world)
    #populate world with random minidudes
    for i in range(init_pop):
        world.population.append(minidude(world))
        
    #eliminate minidudes with matching positions
    for dude in world.population.copy():
        if not dude.isObstacle:
            for otherdude in world.population:
                if otherdude.pos == dude.pos and dude != otherdude:
                    if not otherdude.isObstacle:
                        world.population.remove(otherdude)
                    elif otherdude.isObstacle and (dude in world.population):
                        #print(str(otherdude.pos) + " : " + str(dude.pos))
                        world.population.remove(dude)

class world:
    def __init__(self, init_pop, length, height, elim, birth_limit, birthrate, spawnx, spawny, spawnlength, spawnheight, 
                killLimit, baby_limit, action_limit, representation, obxstart, obxend, obystart, obyend, lifespan):
        self.spawnx = spawnx
        self.spawny = spawny
        self.length = length
        self.height = height
        self.elim = elim
        self.births = 0
        self.birth_limit = birth_limit
        self.kills = 0
        self.killLimit = killLimit
        self.birthrate = birthrate
        self.spawnlength = spawnlength
        self.spawnheight = spawnheight
        self.baby_limit = baby_limit
        self.action_limit = action_limit
        self.representation = representation
        self.init_pop = init_pop
        self.gen = 0
        self.step = 0
        self.obxstart = obxstart
        self.obxend = obxend
        self.obystart = obystart
        self.obyend = obyend
        self.population = []
        populate(init_pop, self, obxstart, obxend, obystart, obyend)
        self.lifespan = lifespan
        

    def atPos(self, position):
        for dude in self.population:
            if dude.pos == position:
                return dude
        return None

    def nextState(self):
        for dude in self.population:
            step = []
            dude.inputlist.append(step)
            dude.actions = 0 
        for dude in self.population:
            if not (dude is None): 
                dude.act(self)
                dude.time += 1
                
        self.step += 1
    
    def nextGen(self):
        #eliminate dudes that violate rules
        self.elim(self.population, self)
        if len(self.population) == 0: 
            print("Extinction")
            populate(self.init_pop, self, self.obxstart, self.obxend, self.obystart, self.obyend)
            return

        #repopulate world with kids of survivors
        children = []
        for survivor in self.population:
            if not survivor.Brain.obstacle:
                for i in range(survivor.wantedChildren):
                    children.append(survivor.baby(self))

        self.population = []
        obstacle(self.obxstart, self.obxend, self.obystart, self.obyend).create(self)
        self.population += children

        #eliminate minidudes with matching positions
        for dude in self.population:
            for otherdude in self.population:
                if (otherdude != dude) and (otherdude.pos == dude.pos):
                    self.population.remove(otherdude)
        self.births = 0;
        self.kills = 0;
        self.gen += 1
        self.step = 0
    
    def frame(self):
        img = np.zeros((self.height + 1,self.length + 1,3), np.uint8)
        img[:,:] = (255,255,255)
        for dude in world.population:
            img[dude.ypos()][dude.xpos()] = int2color(dude.species) #chr((dude.species % 94) + 33) if world.representation else directionChar(dude.lastmove)
        for i in range(self.length):
            img[self.height][i] = int2bw(i, self.length)
        for i in range(self.height + 1):
            img[i][self.length] = int2bw(i, self.height)
        return cv2.resize(img, (self.length * 20, self.height * 20), interpolation = cv2.INTER_AREA)
    
def print_world(world):
        for dude in world.population:
            vis_world[dude.ypos()][dude.xpos()] = chr((dude.species % 94) + 33) if world.representation else directionChar(dude.lastmove)
        print(" ", end = " ")
        for i in range(world.length): print(str(i % 10), end = " ")
        print(str(world.gen), end ="")
        i = 0
        for row in vis_world:
            print()
            print(str(i % 10), end = " ")
            for pos in row:
                print(str(pos), end = " ")
            print("|", end = "")
            i += 1
        print()
        print(" ", end = "")
        for i in range(2*world.length): print("-", end = "")
        print(str(world.step), end ="")

def rules(dude, population, world):
    #for other in population:
    #    if dude.isNear(other) and other.species != dude.species: return False
    if dude.ypos() > 3: return False
    dude.wantedChildren += int(world.length - dude.xpos() / world.length/4) - 1
    return True

def elim(population, world):
    doppel = population.copy()
    for dude in doppel:
        if (not rules(dude, doppel, world) and (dude in population)) or dude.Brain.obstacle: 
            #print(dude.pos)
            population.remove(dude)
    #print(str(len(population)) + " Dudes left")

def printBrains(world):
    for dude in world.population:
        dude.Brain.printBrain()
        print()

def play(video):
        for frame in video:
            (x, y) = (get_monitors()[0].width - frame.shape[1] - 5, 25)
            cv2.imshow("gen: " + str(gen), frame)
            cv2.moveWindow("gen: " + str(gen), x, y)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
    
def loading(year, lifespan): 
    os.system('cls')
    percent = int(10 * (year / lifespan))
    for i in range(percent):
        print("-", end="")
    for i in range(10 - percent):
        print(" ", end="")
    print(str(int(100 * (year / lifespan))) + "%")

if __name__ == "__main__":
    init_pop = 200
    lifespan = 50
    birthLimit = 40
    KillLimit = 20
    maxChildren = 10
    BabyLimit = 10
    action_limit = 10
    (length, height) = (30,30)
    (spawnx, spawnlength, spawny, spawnheight) = (0, 30, 20, 30)
    (obxstart, obxend, obystart, obyend) = (0, 25, 10, 11)
    representation = True
    world = world(init_pop, length, height, elim, birthLimit, maxChildren, spawnx, spawny, spawnlength, spawnheight, 
                    KillLimit, BabyLimit, action_limit, representation, obxstart, obxend, obystart, obyend, lifespan)
    

    generations = int(input("Generations: "))
    steps = int(input("Steps: "))
    video = []
    allGens = []
    step = 0
    gen = 0
    while generations == 0 or gen < generations:
        prev = None
        #printBrains(world)
        #print_world(world)
        #print()
        #cv2.destroyWindow()
        cv2.destroyAllWindows()    
        for year in range(lifespan):
           # if prev != None: play(prev)
            video.append(world.frame())
            allGens.append(world.frame())
            (x, y) = (get_monitors()[0].width - world.frame().shape[1] - 5, 25)
            cv2.imshow("gen: " + str(gen), world.frame())
            cv2.moveWindow("gen: " + str(gen), x, y)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
            world.nextState()
            loading(year, lifespan)
            #cv2.destroyWindow()
            #printBrains(world)
            #print_world(world)
            #print()
        prev = video
        
        cv2. destroyAllWindows() 
        loading(lifespan, lifespan) 
        play(video)
        video = []
        step += 1
        gen += 1

        if not steps == 0 and step >= steps:
            answer = input(">")
            while (answer != "continue"):
                match answer:
                    case "quit":
                        generations = -1
                        break
                    case "inspect":
                        dude = None
                        while(dude == None):
                            print("Position:", end=" ")
                            position = make_tuple(input(""))
                            print(position)
                            dude = world.atPos(position)
                            if dude == None: 
                                print("No dude at that position")
                            else:
                                dude.Brain.draw()
                                #dude.printBrain()
                                print()

                    case "position":
                        print("index:", end =" ")
                        index = input("")
                        print(str(world.population[int(index)].pos))

                    case "stats":
                        population = len(world.population)
                        print("population: " + str(population))
                    
                    case "representation":
                        world.representation = not world.representation

                    case "step":
                        print("Steps:", end =" ")
                        steps = int(input(""))

                    case "years":
                        print("Lifespan:", end =" ")
                        lifespan = int(input(""))

                    case "new map":
                        print("World Dimensions:", end=" ")
                        (world.length, world.height) = make_tuple(input(""))
                        print("Spawn Dimensions:", end=" ")
                        (world.spawnx, world.spawnlength, world.spawny, world.spawnheight) = make_tuple(input(""))
                        print("Obstacle Dimensions:", end=" ")
                        (world.obxstart, world.obxend, world.obystart, world.obyend) = make_tuple(input(""))
                    
                    case "new":                
                        world.population = []
                        obstacle(world.obxstart, world.obxend, world.obystart, world.obyend).create(world)
                        populate(world.init_pop, world, world.obxstart, world.obxend, world.obystart, world.obyend)

                    case "replay":
                        play(prev)
                    
                    case "play":
                        dude = None
                        while(dude == None):
                            print("Position:", end=" ")
                            position = make_tuple(input(""))
                            print(position)
                            dude = world.atPos(position)
                            if dude == None: 
                                print("No dude at that position")
                            else:
                                cv2.destroyAllWindows()  
                                dude.Brain.play(prev, dude)
                    
                    case "play all":
                        cv2. destroyAllWindows() 
                        play(allGens)

                answer = input(">")
            step = 0
        world.nextGen()
    cv2. destroyAllWindows() 
    play(allGens)