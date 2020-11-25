import load_data

import os
import h5py
import mmcv
import argparse
import os.path as osp

# coco dataset format
'''
{
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}


image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}

annotation = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}

categories = [{
    "id": int,
    "name": str,
    "supercategory": str,
}]
'''


def convert_svhn_to_coco_train(ann_file, out_file, image_prefix):
    data_infos = h5py.File(ann_file, "r")

    annotations = []
    images = []
    obj_count = 0
    for idx in range(len(data_infos['/digitStruct/name'])):
        img_name = load_data.get_name(idx, data_infos)
        anno = load_data.get_bbox(idx, data_infos)

        filename = img_name
        img_path = osp.join(image_prefix, img_name)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        for e in range(len(anno['label'])):
            label = int(anno['label'][e])
            left = anno['left'][e]
            top = anno['top'][e]
            width = anno['width'][e]
            height = anno['height'][e]

            if label == 10:
                label = 0
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=label,
                bbox=[left, top, width, height],
                area=width * height,
                segmentation=[],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': 0, 'name': '10'},
                    {'id': 1, 'name': '1'},
                    {'id': 2, 'name': '2'},
                    {'id': 3, 'name': '3'},
                    {'id': 4, 'name': '4'},
                    {'id': 5, 'name': '5'},
                    {'id': 6, 'name': '6'},
                    {'id': 7, 'name': '7'},
                    {'id': 8, 'name': '8'},
                    {'id': 9, 'name': '9'}])
    print(coco_format_json)
    mmcv.dump(coco_format_json, out_file)


def convert_svhn_to_coco_test(out_file, image_prefix):

    import re

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def Sort(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    def sorted_alphanumeric(data):
        return sorted(data, key=Sort)

    sorted_file = sorted_alphanumeric(os.listdir(image_prefix))

    annotations = []
    images = []
    obj_count = 0
    for file in sorted_file:
        if file[-4:] != ".png":
            continue

        idx = file[:-4]
        img_name = file

        filename = img_name
        img_path = osp.join(image_prefix, img_name)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        for e in range(1):
            label = int(0)
            left, top, width, height = 0, 0, 0, 0

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=label,
                bbox=[left, top, width, height],
                area=width * height,
                segmentation=[],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': 0, 'name': '10'},
                    {'id': 1, 'name': '1'},
                    {'id': 2, 'name': '2'},
                    {'id': 3, 'name': '3'},
                    {'id': 4, 'name': '4'},
                    {'id': 5, 'name': '5'},
                    {'id': 6, 'name': '6'},
                    {'id': 7, 'name': '7'},
                    {'id': 8, 'name': '8'},
                    {'id': 9, 'name': '9'}])
    print("DONE")
    mmcv.dump(coco_format_json, out_file)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m',
                        type=str,
                        default='train',
                        help='mode of json')
    args = parser.parse_args()
    if args.mode == 'train':
        convert_svhn_to_coco_train('./dataset/train/digitStruct.mat',
                                   'svhn_coco.json',
                                   './dataset/train')
    elif args.mode == 'test':
        convert_svhn_to_coco_test('svhn_coco_test.json', './dataset/test')
