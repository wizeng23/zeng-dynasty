import collections

from PIL import Image
import numpy as np

DIR = '/Users/williamzeng/Library/Mobile Documents/com~apple~CloudDocs/Documents/misc/projects/family_tree/book1'
PATH = '{}/family-tree-{}.png'
# PATH = '/Users/williamzeng/Downloads/11.png'

def get_tilt(corners):
  sums = [sum(p) for p in corners]
  min_sum = min(sums)
  max_sum = max(sums)
  top_left = [p for p in corners if sum(p) = min_sum]
  bottom_right = [p for p in corners if sum(p) = max_sum]

  i_sorted = sorted(corner, lambda p: p[0])
  if sum(i_sorted[0]) == min_sum:
    top_right = i_sorted[1]
  else:
    top_right = i_sorted[0]
  if sum(i_sorted[2]) == max_sum:
    bottom_left = i_sorted[3]
  else:
    bottom_left = i_sorted[2]
  print(f'Top left: {}')

def get_page_old(grid, start_i, start_j):
  print(f'get_page_old {start_i} {start_j}')
  rows, cols = grid.shape
  queue = collections.deque()
  queue.append((start_i, start_j))

  visited = np.zeros([rows, cols])

  min_i, max_i = (start_i, None), (start_i, None)
  min_j, max_j = (None, start_j), (None, start_j)

  while queue:
    i, j = queue.popleft()
    if i < 0 or i >= rows or j < 0 or j >= cols:
      continue
    if visited[i][j]:
      continue
    if not grid[i][j]:
      continue
    visited[i][j] = 1

    if i < min_i[0]:
      min_i = i, j
    elif i > max_i[0]:
      max_i = i, j
    if j < min_j[1]:
      min_j = i, j
    elif j > max_j[1]:
      max_j = i, j

    queue.append((i + 1, j))
    queue.append((i - 1, j))
    queue.append((i, j + 1))
    queue.append((i, j - 1))

  print('Page corners')
  print(min_i, max_i, min_j, max_j)
  # Cut off page boundaries
  page = grid[min_i[0]+50:max_i[0]-50, min_j[1]+150:max_j[1]-150]
  return page

def get_left_page_old(grid):
  rows, cols = grid.shape
  print(rows, cols)

  i = int(rows / 2)
  j = 0

  while grid[i][j] == 0:
    j += 1

  print(j)

  # visited = np.zeros([rows, cols])
  page = get_page(grid, i, j)
  print(page.shape)
  print(grid[i][j])
  for i in range(200):
    print(list(page[i][:50]))

def get_page(grid, start_i, start_j):
  rows, cols = grid.shape
  queue = collections.deque()
  queue.append((start_i, start_j))

  min_i, max_i = start_i, start_i
  min_j, max_j = start_j, start_j
  visited = np.zeros([rows, cols])

  while queue:
    i, j = queue.popleft()
    if i < 0 or i >= rows or j < 0 or j >= cols:
      continue
    if grid[i][j] or visited[i][j]:
      continue
    visited[i][j] = 1

    if i < min_i:
      min_i = i
    elif i > max_i:
      max_i = i
    if j < min_j:
      min_j = j
    elif j > max_j:
      max_j = j

    queue.append((i + 1, j))
    queue.append((i - 1, j))
    queue.append((i, j + 1))
    queue.append((i, j - 1))

  edge = 50
  min_i += edge
  min_j += edge
  max_i -= edge
  max_j -= edge

  print(min_i, max_i, min_j, max_j)
  return grid[min_i:max_i, min_j:max_j]

def shrink_page(grid):
  rows, cols = grid.shape

  # Cut left/right
  col_present = np.sum(grid, axis=0)
  print(col_present.shape)

  min_col = int(cols / 2)
  max_col = min_col
  while col_present[min_col]:
    min_col -= 1
  while col_present[max_col]:
    max_col += 1
  min_col -= 10
  max_col += 10
  print('Min and max cols')
  print(min_col, max_col)

  grid = grid[:, min_col:max_col + 1]

  # Cut top/bottom
  row_present = np.sum(grid, axis=1)
  print(row_present.shape)
  min_row = 0
  # for i, c in enumerate(list(row_present)):
  #   print(i, c)
  while not row_present[min_row]:
    min_row += 1
  min_row -= 10

  max_row = row_present.shape[0] - 1
  # for i, c in enumerate(list(row_present)):
  #   print(i, c)
  while not row_present[max_row]:
    max_row -= 1
  max_row += 10
  print('Min and max rows')
  print(min_row, max_row)
  grid = grid[min_row:max_row, :]

  return grid

def get_final_page(filepath):
  image = Image.open(filepath)
  # print(image.mode) # RGBA
  data = np.asarray(image)
  # print(data.shape)
  #Image.fromarray(data[:25,-25:]).show()
  # print(data[:25,-25:][:,:,0])
  # red = data[:,:,0]
  # bw = (red < 150).astype(int)
  # for i in range(len(bw)):
  #   print(list(bw[i][:50]))
  # print(list(bw[:100,:].astype(int)))
  # image.show()
  # print(data.shape)
  grid = (data[:,:,0] < 150).astype(int)
  rows, cols = grid.shape
  # print(rows, cols)

  i = int(rows / 2)
  # j = int(cols / 4)
  j = int(3 * cols / 4)
  while grid[i][j]:
    j += 1
  page = get_page(grid, i, j)
  # page = grid[178:2031, 543:1672]  # left page

  print('Page shape')
  print(page.shape)
  page = shrink_page(page)
  print('Page shape after shrinking')
  print(page.shape)
  # for curi in range(200):
  #   print(list(page[curi][-50:]))

  page = page * 255
  page = 255 - page
  out = Image.fromarray(np.uint8(page))
  out.show()
  for num in range(4, 13):
    filepath = PATH.format(DIR, num)

def show_image(grid):
  grid = grid * 255
  grid = 255 - grid
  out = Image.fromarray(np.uint8(grid))
  out.show()

def main():
  # for num in range(4, 13):
  filepath = PATH.format(DIR, '05')
  image = Image.open(filepath)
  # print(image.mode) # RGBA
  data = np.asarray(image)
  # print(data.shape)
  #Image.fromarray(data[:25,-25:]).show()
  # print(data[:25,-25:][:,:,0])
  # red = data[:,:,0]
  # bw = (red < 150).astype(int)
  # for i in range(len(bw)):
  #   print(list(bw[i][:50]))
  # print(list(bw[:100,:].astype(int)))
  # image.show()
  print(data.shape)
  grid = (data[:,:,0] < 150).astype(int)
  rows, cols = grid.shape
  print(rows, cols)

  i = int(rows / 2)
  j = 0
  # j = int(cols / 4)
  # j = int(3 * cols / 4)
  while not grid[i][j]:
    j += 1
  page = get_page_old(grid, i, j)
  # page = get_page(grid, i, j)
  # page = grid[178:2031, 543:1672]  # left page

  print('Page shape')
  print(page.shape)
  page = shrink_page(page)
  print('Page shape after shrinking')
  print(page.shape)
  # for curi in range(200):
  #   print(list(page[curi][-50:]))

  # for num in range(4, 13):
  #   filepath = PATH.format(DIR, num)
  show_image(page)


if __name__ == "__main__":
  main()