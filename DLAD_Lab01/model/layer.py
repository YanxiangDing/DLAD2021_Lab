import numpy as np

class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *inputs):
        r"""Define the forward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *output_grad):
        r"""Define the backward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
## by yourself .Finish your own NN framework
class FullyConnected(_Layer):
    def __init__(self, in_features, out_features, add_bias):
        self.weight = np.random.randn(out_features, in_features) * 0.01
        if add_bias:
            self.bias = np.ones((out_features, 1), dtype=np.float32) * 0.1
        else:
            self.bias = np.zeros((out_features, 1), dtype=np.float32)

        self.weight_grad = np.zeros((out_features, in_features), dtype=np.float32)
        self.bias_grad = np.zeros((out_features, 1), dtype=np.float32)
        self.output = np.zeros((out_features, 1), dtype=np.float32)

    def forward(self, inputs):
        self.samples = inputs.shape[1]
        self.output = np.dot(self.weight, inputs) + self.bias
        return self.output

    def backward(self, dZ, A):
        self.weight_grad = np.dot(dZ, A.T)/self.samples
        self.bias_grad = np.sum(dZ, axis=1, keepdims=True)/self.samples
        input_grad = np.dot(self.weight.T, dZ)
        return input_grad

## by yourself .Finish your own NN framework
class ReLu(_Layer):
    def __init__(self):
        pass

    def forward(self, inputs):
        self.output = inputs
        self.output[self.output < 0] = 0
        return self.output

    def backward(self):
        act_grad = self.output
        act_grad[act_grad > 0] = 1
        return act_grad

class SoftmaxWithloss(_Layer):
    def __init__(self):
        pass

    def forward(self, inputs, target):
        exps = np.exp(inputs - np.amax(inputs, axis=0))
        self.predict = exps/np.sum(exps, axis=0)
        self.target = target
        your_loss = np.sum(-np.multiply(target, np.log(self.predict)))/inputs.shape[1]
        return self.predict, your_loss

    def backward(self):
        softmax_grad = self.predict - self.target
        return softmax_grad
    
    