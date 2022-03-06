from .layer import *

class Network(object):
    def __init__(self, bias=False):  
        self.fc1 = FullyConnected(4096, 1024, add_bias=bias) 
        self.act1 = ReLu()
        self.fc2 = FullyConnected(1024, 256, add_bias=bias)
        self.act2 = ReLu()
        self.fc3 = FullyConnected(256, 64, add_bias=bias)
        self.act3 = ReLu()
        self.fc4 = FullyConnected(64, 13, add_bias=bias)
        self.loss = SoftmaxWithloss()

    def forward(self, input, target):
        h1 = self.fc1.forward(input)
        a1 = self.act1.forward(h1)
        h2 = self.fc2.forward(a1)
        a2 = self.act2.forward(h2)
        h3 = self.fc3.forward(a2)
        a3 = self.act3.forward(h3)
        h4 = self.fc4.forward(a3)
        pred, loss = self.loss.forward(h4, target)
        return pred, loss

    def backward(self, input):
        ## by yourself .Finish your own NN framework
        dz4 = self.loss.backward()
        dA4 = self.fc4.backward(dz4, self.fc3.output)
        dz3 = np.multiply(dA4, self.act3.backward())
        dA3 = self.fc3.backward(dz3, self.fc2.output)
        dz2 = np.multiply(dA3, self.act2.backward())
        dA2 = self.fc2.backward(dz2, self.fc1.output)
        dz1 = np.multiply(dA2, self.act1.backward())
        _ = self.fc1.backward(dz1, input)

    def update(self, lr):
        self.fc1.weight -= lr*self.fc1.weight_grad
        self.fc1.bias -= lr*self.fc1.bias_grad
        self.fc2.weight -= lr*self.fc2.weight_grad
        self.fc2.bias -= lr*self.fc2.bias_grad
        self.fc3.weight -= lr*self.fc3.weight_grad
        self.fc3.bias -= lr*self.fc3.bias_grad
        self.fc4.weight -= lr*self.fc4.weight_grad
        self.fc4.bias -= lr*self.fc4.bias_grad
    
    def update_momentum(self, lr, alpha):
        self.fc1.weight = self.fc1.weight*alpha - lr*self.fc1.weight_grad
        self.fc1.bias = self.fc1.bias*alpha - lr*self.fc1.bias_grad
        self.fc2.weight = self.fc2.weight*alpha - lr*self.fc2.weight_grad
        self.fc2.bias = self.fc2.bias*alpha - lr*self.fc2.bias_grad
        self.fc3.weight = self.fc3.weight*alpha - lr*self.fc3.weight_grad
        self.fc3.bias = self.fc3.bias*alpha - lr*self.fc3.bias_grad
        self.fc4.weight = self.fc4.weight*alpha - lr*self.fc4.weight_grad
        self.fc4.bias = self.fc4.bias*alpha - lr*self.fc4.bias_grad
