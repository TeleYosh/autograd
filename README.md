An implementation of an automatic differentiation algorithm. For now, it only works on scalars.

- **`Value_engine`**: contains the autograd algorithm.
- **`MLP`** implements a small neural network (MLP) to test backpropagation.
- **`Viz`** allows for vizualisation of the operations' graph.

The autograd engine is used to train an MLP on the moon dataset in **`demo`**.
