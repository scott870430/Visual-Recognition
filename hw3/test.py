import load_data
import argparse
import utils
import function
from config import get_cfg_defaults

import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Batch_size = 8
    # '''
    version = args.version
    print(version)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    trainable = cfg.TRAIN.TRAINABLE
    class_number = cfg.TRAIN.CLASS_NUMBER
    hidden_layer = cfg.TRAIN.HIDDEN_LAYER
    isnorm = cfg.TRAIN.ISNORM

    model = function.get_instance_segmentation_model(class_number,
                                                     trainable,
                                                     hidden_layer)
    load_model_path = os.path.join(args.model_path,
                                   version,
                                   'model',
                                   version + str(args.epoch))
    model.load_state_dict(torch.load(load_model_path,
                          map_location=lambda storage, loc: storage))
    model.to(device)

    # '''
    print("test dataset!")
    if isnorm:
        test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                                            ])
    else:
        test_transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            ])
    test_dataset = load_data.TPVDataset_test(test_transform, args.test)
    test_loader = Data.DataLoader(test_dataset,
                                  batch_size=8,
                                  shuffle=False,
                                  num_workers=8,
                                  collate_fn=utils.collate_fn)

    prediction = []
    with torch.no_grad():
        model.eval()
        for batch, (images, idx) in enumerate(test_loader, 1):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            n_instances = len(outputs)
            for i, p in enumerate(outputs):
                for label, mask, score in zip(p['labels'],
                                              p['masks'],
                                              p['scores']):
                    pred = {}
                    pred['image_id'] = idx[i]
                    pred['category_id'] = int(label.cpu())
                    np.set_printoptions(threshold=np.inf)
                    mask = np.where(mask.cpu().numpy() >
                                    args.mask_threshold,
                                    1,
                                    0).astype(np.uint8)[0, :, :]
                    pred['segmentation'] = utils.binary_mask_to_rle(mask)
                    pred['score'] = float(score.cpu())
                    prediction.append(pred)

    prediction_path = './prediction'
    if not os.path.isdir(prediction_path):
        os.makedirs(prediction_path)

    import json
    with open('./prediction/' + '0856165_' +
              args.output_version + '.json', "w") as f:
        json.dump(prediction, f)

    with open('./prediction/' + 'match.txt', "a") as f:
        print(args.output_version,
              args.version,
              args.epoch,
              "           -",
              args.mask_threshold,
              file=f)

    print("END")
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-ids', '-t',
                        type=str,
                        default='7',
                        help='use which gpu')

    parser.add_argument('--config', '-c',
                        type=str,
                        default='config.yaml',
                        help='using config'
                        )
    parser.add_argument('--test',
                        default=False,
                        type=bool,
                        help='is test?'
                        )
    parser.add_argument('--version',
                        type=str,
                        help='model version'
                        )
    parser.add_argument('--epoch',
                        type=str,
                        help='epoch'
                        )
    parser.add_argument('--model-path',
                        type=str,
                        help='model path'
                        )
    parser.add_argument('--output-version',
                        type=str,
                        help='output version'
                        )
    parser.add_argument('--version-name',
                        default='test',
                        type=str,
                        help='version name'
                        )
    parser.add_argument('--mask-threshold',
                        type=float,
                        help='mask threshold'
                        )
    args = parser.parse_args()
    print(args)
    main(args)
