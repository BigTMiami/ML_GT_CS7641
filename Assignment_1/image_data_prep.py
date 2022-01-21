from PIL import Image
import numpy as np

master_data = []

ma = np.empty((0, 30000)).astype(int)


test_image = "data/images/fruits-360_dataset/fruits-360/Training/Apple Braeburn/0_100.jpg"
im = Image.open(test_image)
ima = np.asarray(im).astype(int)


ma = np.append(ma, [ima.flatten()], axis=0)

ma.shape


ma[0][12332]
