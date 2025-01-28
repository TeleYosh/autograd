from autograd.Value_engine import Value
import numpy as np

class Neuron:
  def __init__(self, n_input):
    self.w = [Value(np.random.uniform(-1,1)) for _ in range(n_input)]
    self.b = Value(np.random.uniform(-1,1), label='b')

  def __call__(self, x):
    out = sum((wi*xi for wi, xi in zip(self.w,x)), self.b)
    return out.tanh()

  def parameters(self):
    return self.w

class Layer:
  def __init__(self, n_input, n_out):
    self.neurons = [Neuron(n_input) for _ in range(n_out)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs)==1 else outs

  def parameters(self):
    params = []
    for neuron in self.neurons:
      params.extend(neuron.parameters())
    return params

class MLP:
  def __init__(self, n_input, n_outs):
    self.layers = []
    self.layers.append(Layer(n_input, n_outs[0]))
    for i in range(len(n_outs)-1):
      l = Layer(n_outs[i], n_outs[i+1])
      self.layers.append(l)

  def __call__(self, x):
    z = x
    for layer in self.layers:
      z = layer(z)
    return z

  def parameters(self):
    params = []
    for layer in self.layers:
      layer_param = layer.parameters()
      params.extend(layer.parameters())
    return params