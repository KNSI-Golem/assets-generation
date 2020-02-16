import os
from pathlib import Path
from PIL import Image

for outfit_path in Path('bin').glob('**/*.gif'):
    outfit = Image.open(outfit_path)
    width, height = outfit.size
    if width != 32*4 or height != 48*4:
        os.remove(outfit_path)
