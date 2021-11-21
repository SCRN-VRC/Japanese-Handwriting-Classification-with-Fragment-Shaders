# Find all characters in the ETL dataset that the translator uses

from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import pickle
import random

TF_DATA_DIR = 'D:/Storage/Python/transformers/data/'
ETL_DATA_DIR = Path('./data/etlcdb-image-extractor/etl_data/images/')
DATA_DIR = './data/'

# Characters
num2Char = {}
char2Num = {}
with open(TF_DATA_DIR + 'jp_seq2text.tsv', encoding='utf-8') as f:
    for line in f:
        char, i = line.strip('\n').split('\t')
        char2Num[char] = i
        num2Char[i] = char
    f.close()

# outputs
gen_out = []
gen_labels = []

dirs = list(map(str, list(ETL_DATA_DIR.iterdir())))
#dirs = dirs[len(dirs)-1:]
for img_path in dirs:
    print('Checking %s...' % (img_path))
    char_dir = Path(img_path)
    char_paths = list(map(str, list(char_dir.iterdir())))
    # look in every folder
    for char_path in char_paths:
        check_file = char_path + '/.char.txt'
        try:
            fp = open(check_file, 'r', encoding='utf-8')
            read_char = fp.readline().strip('\s')
            # if char is used by the translator, save picture
            if (not char2Num.get(read_char) == None):
                image_dir = Path(char_path)
                # everything but the .txt file
                images = list(map(str, list(image_dir.glob('*'))))[1:]
                for image_path in images:
                    image = Image.open(image_path)
                    image = ImageOps.grayscale(image)
                    # crop this particular set
                    if 'ETL9G' in char_path:
                        image = image.crop((20, 20, 108, 107))
                    image = image.resize((64, 64))
                    np_img = np.array(image)
                    gen_out.append(np_img)
                    gen_labels.append(read_char)
                print('Added %d images...' % (len(images)))
            fp.close()
        except FileNotFoundError:
            continue

shufList = list(zip(gen_out, gen_labels))
random.shuffle(shufList)
gen_out, gen_labels = zip(*shufList)

# Save data
gen_out = np.array(gen_out)
np.save(DATA_DIR + 'gen_out2.npy', gen_out)
with open(DATA_DIR + 'gen_labels2.pkl', 'wb') as fp:   #Pickling
    pickle.dump(gen_labels, fp)
    fp.close()