from flask import Flask, render_template, request, jsonify
import json
from dataclasses import dataclass


app = Flask(__name__)

@dataclass
class Node:
    id: int
    """Unique identifier for the node."""

    name: str
    """Full name of the person. Can be empty string."""

    name_images: list[str]
    """List of image paths, where each image is a character in the person's name."""

    biography: str
    """Biographical text about the person."""

    generation: int
    """Which generation this person belongs to in the family tree. 1 for the root, -1 if not known."""

    children: list[int]
    """List of child node IDs."""

C = Node('C', 'Node C', [], "Stuff", 2, [])
C2 = Node('C', 'Node C', [], "Stuff", 2, [])
B = Node('B', 'Node B', [], "Stuff", 2, [C, C2])
B2 = Node('B', 'Node B', [], "Stuff", 2, [])
A2 = Node('A2', 'Node A2', ["/static/dog.png"], "Stuff", 2, [ B2])
A = Node('A', 'Node A', ["/static/cat.png"], "Stuff", 1, [A2, B])

nodes = []
edges = []

def compute_graph_data(node):
    assert len(node.name_images) <= 1
    data = {"id": node.id, "label": node.name}
    if len(node.name_images) > 0: 
        data["image"] = node.name_images[0]
    nodes.append({"data": data})

    for child in node.children:
        compute_graph_data(child)
        edges.append({"data": {"source": node.id, "target": child.id}})

compute_graph_data(A)

graph_data = {
    "nodes": nodes,
    "edges": edges
}
print(graph_data)

nodes = json.dumps(nodes)
edges = json.dumps(edges)

html = f"""
<!DOCTYPE html>
<html>
<head>
  <title>Zeng Family</title>
  <script src="https://unpkg.com/cytoscape@3.19.0/dist/cytoscape.min.js"></script>
  <style>
    #cy {{
      width: 100%;
      height: 900px;
      border: 1px solid #ccc;
    }}
  </style>
</head>
<body>
  <div id="cy"></div>


<script>
  var nodes = `{nodes}`;
  var nodes = JSON.parse(nodes);
  console.log(nodes);
  var edges = `{edges}`;
  var edges = JSON.parse(edges);
  console.log(edges);
  cytoscape({{
  container: document.getElementById('cy'),
  elements: [...nodes, ...edges],
  style: [
      {{
      selector: 'node',
      style: {{
          'background-color': '#0074D9',
          'background-fit': 'cover',
          'background-image': 'data(image)',     // Will use image if available
          'label': 'data(label)',                // Will show label if image not there
          'color': '#fff',
          'text-valign': 'bottom',
          'text-halign': 'center',
          'text-outline-color': '#000',
          'text-outline-width': 2,
          'width': 60,
          'height': 60
      }}
      }},
      {{
      selector: 'edge',
      style: {{
          'width': 2,
          'line-color': '#ccc',
          'target-arrow-color': '#ccc',
          'target-arrow-shape': 'triangle'
      }}
      }}
  ],
  layout: {{
      name: 'breadthfirst',     // hierarchical layout
      directed: true,
      padding: 20,
      spacingFactor: 1.5,
  }}
  }});
</script>

</body>
</html>
"""

f = open('./graph_visualizer/templates/index.html', 'w')
f.write(html)
f.close()