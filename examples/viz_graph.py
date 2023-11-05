from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            if v.ctx:
                for child in v.ctx.prev:
                    edges.add((child, v))
                    build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = f"shape {str(n.shape)} | grad {n.grad}", shape='record')
        if n.ctx:
            dot.node(name=str(id(n)) + str(n.ctx), label=str(n.ctx))
            dot.edge(str(id(n)) + str(n.ctx), str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + str(n2.ctx))
    
    return dot

