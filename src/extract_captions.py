from pycocotools.coco import COCO
import os
from contextlib import redirect_stdout
import argparse
import pickle

from regex import E

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', choices=['coco', 'bird'], type=str, required=True)
args = parser.parse_args()
dataset = args.dataset

if dataset == 'coco':
    dir = '../../coco'
    for type in ['train2014', 'val2014']:
        annFile = '{}/annotations/captions_{}.json'.format(dir, type)
        with redirect_stdout(open(os.devnull, 'w')):
            # initialize COCO api for instance annotations
            coco = COCO(annFile)
        captions = []
        images = []
        for img_id, img in coco.imgs.items():
            caps_id = coco.getAnnIds(img_id)
            caps = coco.loadAnns(caps_id)
            for cap in caps:
                cap = cap['caption']
                cap = [x.strip() for x in cap.split(',')]
                cap = ' , '.join(cap)
                cap = cap.replace('.', '')
                cap = cap.replace('\'', ' \'')
                cap = cap.replace('\n', '')
                cap += '\n'
                captions.append(cap)
                images.append(str(img_id) + '\n')
        with open(f'../data/captions/coco/captions_{type}.raw', 'w') as f:
            f.writelines(captions)
        with open(f'../data/captions/coco/images_{type}.txt', 'w') as f:
            f.writelines(images)
elif dataset == 'bird':
    for type in ['train', 'test']:
        path_to_file_names = f'../../birds/{type}/filenames.pickle'
        with open(path_to_file_names, 'rb') as f:
            file_names = pickle.load(f)
        captions = []
        images = []
        for file_name in file_names:
            file_name = file_name.rstrip()
            path_to_captions = f'../../birds/text/{file_name}.txt'
            with open(path_to_captions, 'r') as f:
                caps = f.readlines()
                for cap in caps:
                    # when caption has only one sentence
                    if cap.count('.') == 1:
                        cap = [x.strip() for x in cap.split(',')]
                        cap = ' , '.join(cap)
                        cap = cap.replace('.', '')
                        cap = cap.replace('\'', ' \'')
                        cap = cap.replace('\n', '')
                        cap += '\n'
                        captions.append(cap)
                        images.append(file_name + '\n')
                    # when caption has multiple sentences
                    else:
                        cap = [x.strip() for x in cap.rstrip().rstrip('.').split('.')]
                        cap = ' , '.join(cap)
                        cap = [x.strip() for x in cap.split(',')]
                        cap = ' , '.join(cap)
                        cap = cap.replace('.', '')
                        cap = cap.replace('\'', ' \'')
                        cap = cap.replace('\n', '')
                        cap += '\n'
                        captions.append(cap)
                        images.append(file_name + '\n')
        with open(f'../data/captions/bird/captions_{type}.raw', 'w') as f:
            f.writelines(captions)
        with open(f'../data/captions/bird/images_{type}.txt', 'w') as f:
            f.writelines(images)
