from PIL import Image
import numpy as np

PATH = '/Users/williamzeng/Library/Mobile Documents/com~apple~CloudDocs/Documents/misc/projects/family_tree/pages/tree/11'

image = Image.open(PATH)
data = np.asarray(image)
image.show()