from itertools import combinations
import collections

from PIL import Image
import dataclasses
from typing import List, Optional, Tuple
import numpy as np

import cv2


###############################################################################
# Helpers
###############################################################################


def get_image(filepath):
    """Get the image from the filepath and return a binary array."""
    image = Image.open(filepath)
    data = np.asarray(image)
    # filters out red color, and turns into binary array
    if len(data.shape) == 3:
        return (data[:, :, 0] > 150).astype(np.uint8)
    else:
        return (data > 150).astype(np.uint8)


def save_image(a, filepath):
    """Save the image to the filepath."""
    img = Image.fromarray(a * 255, mode="L")
    img.save(filepath)


def show_image(a):
    """Given a binary grid, display the image."""
    a = a * 255
    # a = 255 - a
    out = Image.fromarray(np.uint8(a))
    out.show()


###############################################################################
# Create pages
###############################################################################


def remove_small_islands(orig_a, max_size=10):
    """Remove specks from the image."""
    a = orig_a.copy()  # avoid modifying original
    rows, cols = a.shape
    visited = np.zeros_like(a, dtype=bool)

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for r in range(rows):
        for c in range(cols):
            if a[r, c] == 0 and not visited[r, c]:
                # Start BFS
                queue = collections.deque([(r, c)])
                visited[r, c] = True
                coords = [(r, c)]

                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and a[nr, nc] == 0
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                            coords.append((nr, nc))

                # After BFS, check island size
                if len(coords) <= max_size:
                    for rr, cc in coords:
                        a[rr, cc] = 1

    return a


def get_corners(a, start_i, start_j):
    """Returns the coordinates of the corners of the page (tl, tr, br, bl)."""
    rows, cols = a.shape
    queue = collections.deque()
    queue.append((start_i, start_j))

    visited = np.zeros([rows, cols])

    tl, tr, br, bl = (
        (start_i, start_j),
        (start_i, start_j),
        (start_i, start_j),
        (start_i, start_j),
    )

    while queue:
        i, j = queue.popleft()
        if i < 0 or i >= rows or j < 0 or j >= cols:
            continue
        if visited[i][j]:
            continue
        if a[i][j]:
            continue
        visited[i][j] = 1

        if -i - j > -tl[0] - tl[1]:
            tl = i, j
        if -i + j > -tr[0] + tr[1]:
            tr = i, j
        if i - j > bl[0] - bl[1]:
            bl = i, j
        if i + j > br[0] + br[1]:
            br = i, j

        queue.append((i + 1, j))
        queue.append((i - 1, j))
        queue.append((i, j + 1))
        queue.append((i, j - 1))

    return tl, tr, br, bl


def normalize_page(a, page_corners, width=1300, height=1950):
    tl, tr, br, bl = page_corners
    tlp = [tl[1], tl[0]]
    trp = [tr[1], tr[0]]
    brp = [br[1], br[0]]
    blp = [bl[1], bl[0]]
    src_pts = np.float32([tlp, trp, brp, blp])
    dst_pts = np.float32(
        [
            [0, 0],  # top-left
            [width - 1, 0],  # top-right
            [width - 1, height - 1],  # bottom-right
            [0, height - 1],  # bottom-left
        ]
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(a, M, (width, height))


###############################################################################
# Create graphs
###############################################################################


def trim_borders(a):
    """Trim the borders of the page."""
    rows, cols = a.shape
    a = a[50 : rows - 50, :]
    col_present = np.sum(1 - a, axis=0)
    if np.sum(col_present[:150]) > np.sum(col_present[-150:]):
        return a[:, 150 : cols - 50]
    else:
        return a[:, 50 : cols - 150]


def shrink_page(a):
    """Shrinks the page to the smallest possible rectangle that contains the tree."""
    rows, cols = a.shape

    def find_shrink_start(arr):
        mid = int(len(arr) / 2)
        search_range = arr[mid - 150 : mid + 150]
        max_idx = np.argmax(search_range)
        return mid - 150 + max_idx

    # Cut left/right
    col_present = np.sum(1 - a, axis=0)
    min_col = find_shrink_start(col_present)
    max_col = min_col
    while min_col > 0 and col_present[min_col]:
        min_col -= 1
    while max_col < cols - 1 and col_present[max_col]:
        max_col += 1
    min_col = max(min_col - 10, 0)
    max_col = min(max_col + 10, cols - 1)
    # print("Min and max cols")
    # print(min_col, max_col)

    a = a[:, min_col : max_col + 1]

    # Cut bottom
    row_present = np.sum(1 - a, axis=1)
    max_row = row_present.shape[0] - 1
    while max_row > 0 and not row_present[max_row]:
        max_row -= 1
    max_row = min(max_row + 10, rows - 1)
    # print("Max rows")
    # print(max_row)
    a = a[:max_row, :]

    return a


def is_tree_start_page(a):
    col_sums = np.sum(1 - a, axis=0)
    end = col_sums.shape[0] - 1
    while end > 0 and col_sums[end] == 0:
        end -= 1
    start = end
    while start > 0 and col_sums[start] > 0:
        start -= 1
    # print("col Start and end")
    # print(start, end)
    if not (50 < end - start < 100):
        return False

    row_sums = np.sum(1 - a[:, start:end], axis=1)
    end = row_sums.shape[0] - 1
    while end > 0 and row_sums[end] == 0:
        end -= 1
    start = 0
    while start < end and row_sums[start] == 0:
        start += 1
    # print("row Start and end")
    # print(start, end)
    return (0 < start < 200) and (250 < end < 650) and (200 < end - start < 600)


def remove_adjacent(numbers, threshold=30):
    """
    Remove adjacent numbers from a set, keeping only the first of each sequence.
    For example: {4, 6, 7, 8} becomes {4, 7}

    Args:
      numbers: Set of integers

    Returns:
      Set with adjacent numbers removed
    """
    if not numbers:
        return []

    sorted_nums = sorted(numbers)
    result = [sorted_nums[0]]

    for i in range(1, len(sorted_nums)):
        # If current number is not adjacent to previous, keep it
        if sorted_nums[i] - sorted_nums[i - 1] > threshold:
            result.append(sorted_nums[i])

    return result


def find_best_orphans(left: list[int], right: list[int]) -> tuple[list[int], str]:
    """Print out location of orphans and which side they are on."""
    flip = False
    if len(right) > len(left):
        flip = True
        left, right = right, left
    num_orphans = len(left) - len(right)
    best_orphans = []
    best_orphan_score = None

    possible_orphan_groups = list(combinations(range(len(left)), num_orphans))
    for orphans in possible_orphan_groups:
        new_left = [left[i] for i in range(len(left)) if i not in orphans]
        score = sum([abs(x - y) for x, y in zip(new_left, right)])
        if best_orphan_score is None or score < best_orphan_score:
            best_orphan_score = score
            best_orphans = orphans
    return best_orphans, "left" if not flip else "right"


# TODO: Account for case where orphan could be top edge. Set padding after finding orphan
# Find orphan, remove it, then add padding to minimize line distance
# When finding orphan, normalize distances somehow?
# TODO: Retrim graph
# TODO: Add more border padding
def merge_graphs(g1, g2):
    """Connect the two graphs, g1 on the left."""
    g1_edge = np.sum(1 - g1[:, -5:], axis=1)
    left_x = remove_adjacent([x for x in range(len(g1_edge)) if g1_edge[x] > 0])
    g2_edge = np.sum(1 - g2[:, :5], axis=1)
    right_x = remove_adjacent([x for x in range(len(g2_edge)) if g2_edge[x] > 0])
    print("Left x, right x")
    print(left_x, right_x)

    # Try matching at top
    if left_x[0] < right_x[0]:
        # Add rows to top of g1 to align with g2
        padding = right_x[0] - left_x[0]
        left_x = [x + padding for x in left_x]
        g1 = np.vstack([np.ones((padding, g1.shape[1])).astype(np.uint8), g1])
    elif left_x[0] > right_x[0]:
        # Add rows to top of g2 to align with g1
        padding = left_x[0] - right_x[0]
        right_x = [x + padding for x in right_x]
        g2 = np.vstack([np.ones((padding, g2.shape[1])).astype(np.uint8), g2])

    # Make heights equal by padding bottom of shorter array with 1's
    if g1.shape[0] < g2.shape[0]:
        padding = g2.shape[0] - g1.shape[0]
        g1 = np.vstack([g1, np.ones((padding, g1.shape[1])).astype(np.uint8)])
    elif g2.shape[0] < g1.shape[0]:
        padding = g1.shape[0] - g2.shape[0]
        g2 = np.vstack([g2, np.ones((padding, g2.shape[1])).astype(np.uint8)])

    # Horizontally concatenate the arrays
    orphan_idxs, side = find_best_orphans(left_x, right_x)
    print("Adjusted left x, right x")
    print(left_x, right_x)
    if side == "left":
        orphans = [left_x[i] for i in orphan_idxs]
        left_x = [left_x[i] for i in range(len(left_x)) if i not in orphan_idxs]
    else:
        orphans = [right_x[i] for i in orphan_idxs]
        right_x = [right_x[i] for i in range(len(right_x)) if i not in orphan_idxs]
    if len(orphan_idxs) > 0:
        print(f"!!!!!!!!!!!!!!!!!!Orphans {orphans} on {side} side")
    for left, right in zip(left_x, right_x):
        low, high = min(left, right), max(left, right)
        g1[low : high + 1, -1:] = 0
        g2[low : high + 1, :1] = 0
    return np.hstack([g1, g2])


###############################################################################
# Parse graphs
###############################################################################


def find_lines(image, threshold=70):
    """
    Find connected components in a binary image that span more than 70 pixels
    in either width or height.

    Args:
      image: 2D list where 0 = black pixel, 1 = white pixel

    Returns:
      List of sets, where each set contains (row, col) tuples of pixels
      in a large connected component
    """
    if len(image) == 0 or len(image[0]) == 0:
        return []

    image = image.copy()
    rows = len(image)
    cols = len(image[0])
    results = []

    # Iterate through entire array
    for i in range(rows):
        for j in range(cols):
            # If we find a black pixel (0)
            if image[i][j] == 0:
                # BFS to find connected component
                component = set()
                queue = [(i, j)]
                image[i][j] = 1  # Mark as visited

                min_x = max_x = i
                min_y = max_y = j

                while queue:
                    x, y = queue.pop(0)
                    component.add((x, y))

                    # Update bounds
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)

                    # Check four direct directions
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy

                        # Check bounds and if pixel is black
                        if 0 <= nx < rows and 0 <= ny < cols and image[nx][ny] == 0:
                            image[nx][ny] = 1  # Mark as visited
                            queue.append((nx, ny))

                # Check if component is large enough
                if max_x - min_x > threshold or max_y - min_y > threshold:
                    results.append(component)

    return results


def find_line_ends(points, threshold=50):
    """
    Find the endpoints of a line represented by a set of points.

    Args:
      points: Set of (x, y) tuples representing a connected component

    Returns:
      Tuple of two sets: (left_end_ys, right_end_ys)
      Each set contains y-coordinates of endpoints, with adjacent values removed
    """
    if not points:
        return (set(), set())

    # Find minimum x coordinate
    min_x = min(x for x, y in points)

    # Get y coordinates at the top
    top_ys = {y for x, y in points if x == min_x}

    # Remove adjacent numbers
    top_ys_filtered = remove_adjacent(top_ys)

    # Find maximum x coordinate
    max_x = max(x for x, y in points)

    # Get y coordinates near the bottom
    bottom_ys = {y for x, y in points if abs(x - max_x) <= threshold}

    # Remove adjacent numbers
    bottom_ys_filtered = remove_adjacent(bottom_ys)

    top_points = [(min_x, y) for y in sorted(top_ys_filtered)]
    bottom_points = [(max_x, y) for y in sorted(bottom_ys_filtered)]
    return (top_points, bottom_points)


@dataclasses.dataclass
class Node:
    id: Optional[int] = None
    top: Optional[Tuple[int, int]] = None
    bot: Optional[Tuple[int, int]] = None
    children: List["Node"] = dataclasses.field(default_factory=list)

    def __str__(self):
        return f"Node(id={self.id}, top={self.top}, bot={self.bot})"


def sort_nodes(nodes: List[Node]) -> List[Node]:
    """Sort nodes, top to bottom, left to right"""

    def compare_key(node):
        return (node.top[0], node.top[1])

    # First, sort by both top[0] and top[1] as a baseline
    nodes_sorted = sorted(nodes, key=compare_key)

    # Now apply the custom logic
    result = []
    i = 0
    while i < len(nodes_sorted):
        # Collect nodes that should be grouped together
        group = [nodes_sorted[i]]
        j = i + 1

        while j < len(nodes_sorted):
            # Check if top[0] values are within 60 of each other
            if abs(nodes_sorted[j].top[0] - nodes_sorted[i].top[0]) < 60:
                group.append(nodes_sorted[j])
                j += 1
            else:
                break

        # Sort the group by top[1]
        group.sort(key=lambda node: node.top[1])
        result.extend(group)
        i = j

    return result


def infer_ends(nodes, a):
    """Infer the top and bottom ends of nodes."""
    for n in nodes:
        if n.top is None:
            top = max(0, n.bot[0] - 200)
            y = n.bot[1]
            min_y = max(0, y - 30)
            max_y = min(a.shape[1] - 1, y + 30)
            while top < n.bot[0] and 0 not in a[top][min_y:max_y]:
                top += 1
            top = max(0, top - 10)
            n.top = (top, y)
        if n.bot is None:
            bot = min(a.shape[0] - 1, n.top[0] + 200)
            print("start bot", bot)
            y = n.top[1]
            min_y = max(0, y - 30)
            max_y = min(a.shape[1] - 1, y + 30)
            while bot > n.top[0] and 0 not in a[bot][min_y:max_y]:
                bot -= 1
            bot = min(bot + 10, a.shape[0] - 1)
            n.bot = (bot, y)


def verify_nodes(nodes):
    """Verify the nodes are valid."""
    for n in nodes:
        if not (60 < n.bot[0] - n.top[0] < 250):
            print("node looks sus:", str(n))


def print_trees(nodes):
  def print_tree(n, indent):
    print("  " * indent + str(n))
    for c in n.children:
      print_tree(c, indent + 1)

  children = []
  for n in nodes:
    for c in n.children:
      children.append(c)
  for n in nodes:
    if n not in children:
      print_tree(n, 0)


def get_name_image(node, a):
    """Get the image of the name of a node."""
    y = node.top[1]
    name = a[node.top[0]+5:node.bot[0]-5, max(0, y-40):min(a.shape[1], y+40)]

    col_present = np.sum(1 - name, axis=0)
    min_y = 0
    while min_y < len(col_present) and not col_present[min_y]:
        min_y += 1
    min_y = max(min_y - 10, 0)
    max_y = len(col_present) - 1
    while max_y > 0 and not col_present[max_y]:
        max_y -= 1
    max_y = min(max_y + 10, len(col_present) - 1)
    name = name[:, min_y:max_y]
    print("min_y", min_y, "max_y", max_y)

    row_present = np.sum(1 - name, axis=1)
    min_x = 0
    while min_x < len(row_present) and not row_present[min_x]:
        min_x += 1
    min_x = max(min_x - 10, 0)
    max_x = len(row_present) - 1
    while max_x > 0 and not row_present[max_x]:
        max_x -= 1
    max_x = min(max_x + 10, len(row_present) - 1)
    name = name[min_x:max_x, :]
    print("min_x", min_x, "max_x", max_x)
    return name
