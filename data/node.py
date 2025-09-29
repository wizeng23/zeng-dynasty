
from dataclasses import dataclass

@dataclass
class Node:
    id: int
    """Unique identifier for the node."""

    name: str
    """Full name of the person. Can be empty string."""

    name_images: list[str]
    """List of image paths, where each image is a character in the person's name.
    
    This is a fallback field for when the name is not known.
    """

    generation: int
    """Which generation this person belongs to in the family tree. 1 for the root, -1 if not known."""

    parent: int
    """ID of the parent node. Root node has parent -1."""

    children: list[int]
    """List of child node IDs."""

    biography: str
    """Biographical text about the person."""

    notes: str
    """Notes about the person. For now, this is handwritten notes in the book by the person's name."""


