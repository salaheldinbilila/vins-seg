# Helper functions for segmentation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2 as cv
from keras_segmentation.data_utils.data_loader import get_image_array

# Generated color map from the csv file
color_map = {0: {'name': 'unlabeled', 'rgb': [0, 0, 0]},
1: {'name': 'paved-area', 'rgb': [128, 64, 128]},
2: {'name': 'dirt', 'rgb': [130, 76, 0]},
3: {'name': 'grass', 'rgb': [0, 102, 0]},
4: {'name': 'gravel', 'rgb': [112, 103, 87]},
5: {'name': 'water', 'rgb': [28, 42, 168]},
6: {'name': 'rocks', 'rgb': [48, 41, 30]},
7: {'name': 'pool', 'rgb': [0, 50, 89]},
8: {'name': 'vegetation', 'rgb': [107, 142, 35]},
9: {'name': 'roof', 'rgb': [70, 70, 70]},
10: {'name': 'wall', 'rgb': [102, 102, 156]},
11: {'name': 'window', 'rgb': [254, 228, 12]},
12: {'name': 'door', 'rgb': [254, 148, 12]},
13: {'name': 'fence', 'rgb': [190, 153, 153]},
14: {'name': 'fence-pole', 'rgb': [153, 153, 153]},
15: {'name': 'person', 'rgb': [255, 22, 96]},
16: {'name': 'dog', 'rgb': [102, 51, 0]},
17: {'name': 'car', 'rgb': [9, 143, 150]},
18: {'name': 'bicycle', 'rgb': [119, 11, 32]},
19: {'name': 'tree', 'rgb': [51, 51, 0]},
20: {'name': 'bald-tree', 'rgb': [190, 250, 190]},
21: {'name': 'ar-marker', 'rgb': [112, 150, 146]},
22: {'name': 'obstacle', 'rgb': [2, 135, 115]}}

# Get the colored mask from the segmented mask
def mask_to_color(mask,map_dict):
    rgb_mask = np.zeros(mask.shape + (3,),dtype=np.uint8)
    for key in np.unique(mask):
        rgb_mask[mask==key] = map_dict[key]['rgb']
    return rgb_mask

# Get the legend handles required for plotting
def create_legend_handles(mask,map_dict):
    handles = []
    for key in np.unique(mask):
        color_val = [x/255.0 for x in map_dict[key]['rgb']]
        name = map_dict[key]['name']
        h = mpatches.Patch(color=tuple(color_val), label=name)
        handles.append(h)
    return handles

# Predict function
def seg_predict(model,inp):
    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes
    x = get_image_array(inp, input_width, input_height,ordering=None)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
    pr = cv.resize(pr, (inp.shape[1], inp.shape[0]), interpolation=cv.INTER_NEAREST)
    return pr

# Visualize rgb mask
def visualize(inp,pr,color_map):
    h = create_legend_handles(pr,color_map)
    pr = mask_to_color(pr,color_map)
    inp = cv.cvtColor(inp,cv.COLOR_BGR2RGB)
    fig, arr = plt.subplots(1,3,figsize=(20,15))
    arr[0].imshow(inp)
    arr[0].set_title('Original Image')
    arr[1].imshow(inp)
    arr[1].imshow(pr, alpha=0.5)
    arr[1].set_title('Segmented Image')
    arr[2].imshow(pr)
    arr[2].set_title('Mask')
    arr[2].legend(handles=h, bbox_to_anchor=(1, 0.5), loc='center left')
    plt.savefig('out.png')
