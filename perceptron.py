import numpy as np
from matplotlib import pyplot

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=0, learning_rate=0.2):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, label):
        prediction = self.predict(training_inputs)
        self.weights[1] += self.learning_rate * (label - prediction) * training_inputs[0]
        self.weights[2] += self.learning_rate * (label - prediction) * training_inputs[1]
        self.weights[0] += self.learning_rate * (label - prediction)


p=Perceptron(2)

file = open('data.txt', 'r')
data = []
training_inputs=[]
labels=[]
for line in file:
    new_data=(list(map(float, line.split(','))))
    training_input=[new_data[0],new_data[1]]
    label=new_data[2]
    training_inputs.append(training_input)
    labels.append(label)

#acc_in_each_iteration=[]

for j in range(2000):
  for i in range(0,200):
      p.train(training_inputs[i],labels[i])

  counter=0
  for i in range(0,200):
      activation_result=p.predict(training_inputs[i])
      if(activation_result==labels[i]):
        counter+=1
      
  #acc_in_each_iteration.append(counter)
  if j%500==0:
    print j
  pyplot.scatter(j,counter/2)
    #acc_in_each_iteration.append([j,counter/2])
pyplot.show()

for i in range(0,200):
      activation_result=p.predict(training_inputs[i])
      if(activation_result==labels[i]):
        counter+=1
        pyplot.scatter(training_inputs[i][0],training_inputs[i][1],c="blue")
      else:
        pyplot.scatter(training_inputs[i][0],training_inputs[i][1],c="green")

#x1,x2,y1,y2
x1=-200
x2=0
weight_list=p.weights
y1=(-weight_list[0]-x1*weight_list[1])/weight_list[2]
y2=(-weight_list[0]-x2*weight_list[1])/weight_list[2]
pyplot.plot([x1,x2],[y1,y2],c="red")
  
pyplot.show()

