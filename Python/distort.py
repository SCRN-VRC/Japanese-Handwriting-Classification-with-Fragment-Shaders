# Distort the ETL dataset for training

from PIL import Image
import numpy as np
import pickle
import random
from tqdm import tqdm

DATA_DIR = './data/'

gen_out = np.load(DATA_DIR + 'gen_out2.npy')
fb = open(DATA_DIR + 'gen_labels2.pkl','rb')
gen_labels = pickle.load(fb)
fb.close()

w = 64
h = 64

for i in tqdm(range(len(gen_out))):
    image = Image.fromarray(gen_out[i], 'L')
    
    # warp
    dx = w * random.uniform(0.0, 0.5)
    dy = h * random.uniform(0.0, 0.5)
    x1 = int(random.uniform(-dx, dx))
    y1 = int(random.uniform(-dy, dy))
    x2 = int(random.uniform(-dx, dx))
    y2 = int(random.uniform(-dy, dy))
    w2 = w + abs(x1) + abs(x2)
    h2 = h + abs(y1) + abs(y2)
    data = (
        x1, y1,
        -x1, h2 - y2,
        w2 + x2, h2 + y2,
        w2 - x2, -y1,
    )

    # keep the dimensions same
    image = image.transform((w, h), Image.QUAD, data, fillcolor=(255))
    rand_rot = random.uniform(-15.0, 15.0)
    image = image.rotate(rand_rot, fillcolor=(255))

    gen_out[i] = np.array(image)
    
# Save data
gen_out = np.array(gen_out)
np.save(DATA_DIR + 'gen_out2.npy', gen_out)
with open(DATA_DIR + 'gen_labels2.pkl', 'wb') as fp:   #Pickling
    pickle.dump(gen_labels, fp)
    fp.close()