from flask import Flask, Response
import graphviz

app = Flask(__name__)

def create_family_tree():
    # Create a directed graph
    dot = graphviz.Digraph(format="svg")
    dot.attr(rankdir="TB", fontsize="10")

    # Example: node with only text
    dot.node("A", label="Grandparent")

    # Example: node with only image
    #dot.node("B", image="C:\\Users\\orien\\OneDrive\\Documents\\Programming\\zeng-dynasty\\graph_visualizer\\static\\cat.png", label="", shape="box")
    dot.node("B", image="", label="", shape="box")

    # Example: node with image and text
    dot.node("C", image="http://127.0.0.1:5000/static/cat.png", label="Parent", labelloc="b", shape="box")

    # Edges
    dot.edge("A", "B")
    dot.edge("A", "C")

    return dot


@app.route("/")
def render_tree():
    dot = create_family_tree()
    svg_data = dot.pipe(format="svg")
    return Response(svg_data, mimetype="image/svg+xml")


if __name__ == "__main__":
    app.run(debug=True)
