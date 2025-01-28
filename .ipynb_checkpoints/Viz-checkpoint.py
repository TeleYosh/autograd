from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(node):
      if node not in nodes:
        nodes.add(node)
        for child in node._prev:
          # nodes.add(child) # bad idea
          edges.add((child, node))
          build(child)
    build(root)
    return nodes, edges

def draw_graph(root):
  dot = Digraph(format='svg', graph_attr={'rankdir':'LR'})
  nodes, edges = trace(root)

  for n in nodes:
    uid = str(id(n))
    dot.node(name = uid, label = f"{n.label}|data {n.data: .4f}|grad {n.grad: .4f}", shape='record')
    if n._op:
      dot.node(name = uid + n._op, label=n._op)
      dot.edge(uid + n._op, uid)
  for n1, n2 in edges:
    uid1, uid2 = str(id(n1)), str(id(n2))
    dot.edge(uid1, uid2+n2._op)

  return dot