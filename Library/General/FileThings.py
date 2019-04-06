import os
import json
import numpy as np
from os.path import join
from PIL import Image


### PATHS ###

def get_filepaths(folder, ext=''):
    """ return filepaths of all files in folder """
    return [join(folder, file) for file in os.listdir(folder) if ext in file]


### TEXT FILE ###

def read_text_file(filepath):
    """ return lines from text file """
    with open(filepath, 'r') as file:
        return file.read().split('\n')

def read_file_csv(filepath, split_value=','):
    """ return lines from text file split by specified value """
    return [row.split(split_value) for row in read_text_file(filepath)]

def read_file_split_line_break(filepath):
    """ return list of lists from text file split by blank lines """
    lines = [''] + read_text_file(filepath)
    idxs = [i for i, l in enumerate(lines) if l == ''] + [len(lines)]
    data = [lines[idx + 1:idxs[i + 1]] for i, idx in enumerate(idxs[:-1])]
    return data


### IMAGE FILE ###

def load_image(file):
    """ returns normalized data for image file """
    data = np.array(Image.open(file)) / 255
    return data[:, :, :3] if data.shape[2] > 3 else data

def load_images(filepaths):
    """ load images from filepaths as array """
    return np.array([load_image(f) for f in filepaths])

def save_image_to_file(data, filepath, print_me=False):
    """ convert array to image and save to file """
    data = data * 255 if data.max() <= 1.0 else data
    image = Image.fromarray(data.astype('uint8'))
    image.save(filepath)
    if print_me:
        print('Saved img to {}'.format(os.path.basename(filepath)))


### JSON FILE ###

def load_json(filepath):
    """ read json from file """
    with open(filepath, 'r') as file:  
        return json.load(file)

def write_json(filepath, data, indent=3):
    """ write json to file """
    with open(filepath, 'w') as file:  
        json.dump(data, file, indent=indent)

def base_json():
    """ """
    data = {'network': 
                {'auto': {},
                 'class': {},
                 'reg': {},
                 'embed': {}
                 },
            'learning': {}
            }
    return data


### OLD ###

def remove_things():
    """ ??? """
    base_path = os.path.join(paths.base_path, 'data', 'image', 'hearthstone')
    filepaths = [os.path.join(base_path, f) for f in os.listdir(base_path)]
    idxs = [int(f.split('.')[0].split('_')[-1]) for f in os.listdir(base_path)]
    print('Num files: {}'.format(len(filepaths)))
    # loop through first 10
    rems = []
    ds = list(dt.load_image(filepaths[0]))
    for i in range(1, len(idxs[1:1000])):
        ds.append(list(dt.load_image(filepaths[i])))
        if len(ds) > 2:
            ds = ds[1:]
        mean = np.abs(np.mean(ds[1] - ds[0]))
        if mean < 1e-4:
            print(i)
            rems.append(i)
    return rems
    
def rename_idxs():
    """ """
    base_path = os.path.join(paths.base_path, 'data', 'image', 'hearthstone')
    filepaths = [os.path.join(base_path, f) for f in os.listdir(base_path)]
    idxs = [f.split('.')[0].split('_')[-1] for f in os.listdir(base_path)]
    for i, idx in enumerate(idxs):
        if int(idx) < 1000:
            old_path = filepaths[i]
            new_path = filepaths[i].replace(idx, '0' * (4 - len(idx)) + idx)
            #os.rename(old_path, new_path)


