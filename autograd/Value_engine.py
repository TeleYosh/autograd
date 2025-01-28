import numpy as np

class Value:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data}, label={self.label})"

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, _children=(self, other), _op='+')

    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward

    return out

  def __radd__(self, other):
    return self + other

  def __neg__(self):
    return self * (-1)

  def __sub__(self, other):
    return self + -other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, _children=(self, other), _op='*')
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data**other, _children=(self,), _op=f'**{other}')

    def _backward():
      self.grad += other*self.data**(other-1) * out.grad
    out._backward = _backward

    return out

  def __truediv__(self, other):
    return self * other**(-1)

  def tanh(self):
    t = (np.exp(2*self.data)-1) / (np.exp(2*self.data)+1)
    out = Value(t, _children=(self,), _op='tanh')
    def _backward():
      self.grad += (1-t**2) * out.grad
    out._backward = _backward
    return out

  def __rmul__(self, other):
    return self * other

  def exp(self):
    e = np.exp(self.data)
    out = Value(e, _children=(self,), _op='exp')
    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward
    return out

  def backward(self):
    topo = []
    visited = set()
    def topo_sort(node):
      if node not in visited:
        visited.add(node)
        for child in node._prev:
          topo_sort(child)
        topo.append(node)

    topo_sort(self)
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
