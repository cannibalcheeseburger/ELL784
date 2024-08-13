import matplotlib.pyplot as plt
from random import randint
class Perceptron:
    def __init__(self,weights,bias):
        self.weights = weights
        self.bias = bias
    
    def output(self,inputs):
        net = 0
        for i in range(len(inputs)):
            net = net + self.weights[i]*inputs[i]
        return net+self.bias

    def classify(self,points):
        classes = {0:[],1:[]}
        for point in points:
            print("For Points:",point)
            if self.output(point)<0:
                print("Class 0")
                classes[0].append(point)
            else:
                print("Class 1")
                classes[1].append(point)
        return classes

def plot(output):
    plt.scatter([point[0] for point in output[0]],[point[1] for point in output[0]])
    plt.scatter([point[0] for point in output[1]],[point[1] for point in output[1]])
    plt.legend(["Class 0","Class 1"],ncol=2,loc = "lower right")
    plt.show()
    
def generate_points(dimension,num):
    points = []
    for i in range(num):
        point = []
        point = [(randint(0,100)/100) for _ in range(dimension)]
        points.append(point)
    return points   
        
preceptron = Perceptron(weights=[1,1],bias=-1.5)
#points = [[0,0],[0,1],[1,0],[1,1]]
points = generate_points(2,10)
output = preceptron.classify(points)
plot(output)
