from pycocotools.coco import COCO
import os
from contextlib import redirect_stdout

dataDir = '../../coco'
dataType = 'train2017'
annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
with redirect_stdout(open(os.devnull, 'w')):
    # initialize COCO api for instance annotations
    coco = COCO(annFile)

cap_idx_list = []
for img_id, img in coco.imgs.items():
    caps_id = coco.getAnnIds(img_id)
    caps = coco.loadAnns(caps_id)
    num_cap = 0
    for cap in caps:
        cap = cap['caption']
        cap = cap.replace(',', ' ,')
        cap = cap.replace('.', '')
        cap = cap.replace('\n', '')
        print(cap)
        cap_idx_list.append(str(img_id) + '.' + str(num_cap) + '\n')
        num_cap += 1
with open('cap_idx.txt', 'w') as f:
    f.writelines(cap_idx_list)
