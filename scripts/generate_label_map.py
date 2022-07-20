import glob
import os

imagenet_val_dir = 'datasets/imagenet/val'
val_label_map_path = 'assets/imagenet_val_label_map.txt'

classes = [
    d for d in os.listdir(imagenet_val_dir)
    if os.path.isdir(os.path.join(imagenet_val_dir, d)) and not d.startswith('.')
]
assert len(classes) == 1000
classes.sort()
class_to_idx = {classes[i]: i for i in range(1000)}

img_list = glob.glob(os.path.join(imagenet_val_dir, '*/*.JPEG'))
with open(val_label_map_path, 'w') as f:
    for img_path in img_list:
        label_name = img_path.split('/')[-2]
        img_name = os.path.splitext(img_path.split('/')[-1])[0]
        f.write(f'{img_name} {label_name} {class_to_idx[label_name]}\n')
