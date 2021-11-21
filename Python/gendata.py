from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
import random
import pickle

TF_DATA_DIR = 'D:/Storage/Python/transformers/data/'
DATA_DIR = './data/'
FONTS_DIR = Path('./fonts/')
ITERS = 60

# Characters to generate
characters = []
with open(TF_DATA_DIR + 'jp_seq2text.tsv', encoding='utf-8') as f:
    for line in f:
        char, _ = line.strip('\n').split('\t')
        characters.append(char)

gen_out = []
gen_labels = []
fonts = list(map(str, list(FONTS_DIR.glob('*'))))
sizes = list(range(35, 52))

for i in tqdm(range(ITERS)):
    
    random.shuffle(characters)
    for char in characters:
        rand_bg = random.randint(200, 255)
        image = Image.new('L', (64, 64), (rand_bg))
        draw = ImageDraw.Draw(image)
        
        # random font
        rand_font = random.randint(0, len(fonts) - 1)
        rand_size = random.sample(sizes, 1)[0]
        font = ImageFont.truetype(fonts[rand_font], rand_size)
        
        # offset
        rand_x = 0
        rand_y = 0
        rand_col = random.randint(0, 30)
        draw.text((rand_x, rand_y), '学', (rand_col), font=font)
        w, h = draw.textsize('学', font=font)
        
        # warp
        dx = w * random.uniform(0.0, 0.4)
        dy = h * random.uniform(0.0, 0.4)
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
        image = image.transform((w, h), Image.QUAD, data, fillcolor=(rand_bg))
        bg = Image.new('L', (64, 64), (rand_bg))
        bg.paste(image, ((64 - w) // 2, (64 - h) // 2))
        image = bg
        
        # rotation
        rand_rot = random.uniform(-15.0, 15.0)
        image = image.rotate(rand_rot, fillcolor=(rand_bg))
        
        # blur
        rand_blur = random.randint(0, 1)
        image = image.filter(ImageFilter.GaussianBlur(rand_blur))
        #image.show()
        # check if font generated anything
        np_img = np.array(image)
        average = np.average(np_img, axis=(0,1))
        image.show()
        if ((rand_bg - average) > 5):
            gen_out.append(np_img)
            gen_labels.append(char)

# # Save data
# gen_out = np.array(gen_out)
# np.save(DATA_DIR + 'gen_out.npy', gen_out)
# with open(DATA_DIR + 'gen_labels.pkl', 'wb') as fp:   #Pickling
#     pickle.dump(gen_labels, fp)
#     fp.close()