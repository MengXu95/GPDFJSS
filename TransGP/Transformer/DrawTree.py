from graphviz import Digraph
import uuid


def draw_tree(tree, filename="tree_output"):
    dot = Digraph(comment="Syntax Tree")

    def add_node(dot, node, parent=None):
        node_id = str(uuid.uuid4())  # Unique ID for each node

        if isinstance(node, tuple):
            label = node[0]
            dot.node(node_id, label)
            if parent:
                dot.edge(parent, node_id)
            for child in node[1:]:
                add_node(dot, child, node_id)
        else:
            # Leaf (terminal)
            dot.node(node_id, node)
            if parent:
                dot.edge(parent, node_id)

    try:
        add_node(dot, tree)
        dot.render(filename, format='png', cleanup=True)
        print(f"Tree rendered and saved to {filename}.png")
    except Exception as e:
        print(f"Error drawing tree: {e}")


# Example usage
tree = ('subtract',
        ('minimum',
         ('multiply', 'OWT', 'PT'),
         ('add',
          ('add',
           ('maximum', 'WKR', 'WIQ'),
           ('maximum', 'WIQ', 'MWT')),
          'WIQ')),
        ('protected_div',
         ('add',
          ('maximum',
           ('subtract', 'TIS', 'PT'),
           ('subtract',
            ('maximum', 'PT', 'TIS'),
            'WIQ')),
          ('subtract',
           ('add', 'TIS', 'MWT'),
           ('maximum', 'PT', 'TIS'))),
         ('maximum',
          ('maximum',
           ('minimum', 'SLACK', 'WKR'),
           ('maximum', 'WKR', ('minimum', 'SLACK', 'WKR'))),
          ('minimum',
           ('protected_div', 'TIS', 'TIS'),
           ('multiply', 'OWT', 'PT')))))

# draw_tree(tree)
