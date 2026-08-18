from graphviz import Digraph

def render_family_tree(family_relations, output_file="family_tree"):
    """
    Render a family tree using Graphviz.

    Parameters
    ----------
    family_relations : list of tuples
        Each tuple is (parent, child).
    output_file : str
        The filename for the rendered output (without extension).
    """
    dot = Digraph(comment="Family Tree", format="png")
    dot.attr(rankdir="TB")  # Top to bottom

    # Add edges for parent → child relationships
    for parent, child in family_relations:
        dot.edge(parent, child)

    # Render to file
    dot.render(output_file, view=True)

# Example usage
if __name__ == "__main__":
    relations = [
        ("Grandpa Joe", "Dad"),
        ("Grandma Mary", "Dad"),
        ("Dad", "Me"),
        ("Mom", "Me"),
        ("Dad", "Sister"),
    ]

    render_family_tree(relations, "family_tree")
