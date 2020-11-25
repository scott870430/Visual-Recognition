import json
import argparse


def new_dict():
    temp = {}
    temp['bbox'], temp['score'], temp['label'] = [], [], []

    return temp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t',
                        type=str,
                        default='yolov3.bbox.json',
                        help='file name of yolov3 prediction json')

    parser.add_argument('--version', '-v',
                        type=str,
                        default='test',
                        help='output file name of json')

    args = parser.parse_args()
    save_prediction_path = './prediction'
    if not os.path.isdir(save_prediction_path):
        os.makedirs(save_prediction_path)
    file_name = os.path.join(save_prediction_path, args.target)
    
    f = open(file_name, "r")
    json_file = json.load(f)

    prediction = []
    d = new_dict()
    img_index = 0
    for e in json_file:
        if str(e['image_id']) != str(img_index):
            prediction.append(d)
            img_index += 1
            d = new_dict()
            if str(e['image_id']) != str(img_index):
                print(img_index)
                prediction.append(d)
                img_index += 1
                d = new_dict()

        if str(e['image_id']) == str(img_index):

            box = [round(t) for t in e['bbox']]
            d['bbox'].append([box[1],
                              box[0],
                              box[1] + box[3],
                              box[0] + box[2]])
            if e['category_id'] == 0:
                e['category_id'] = 10
            d['label'].append(e['category_id'])
            d['score'].append(e['score'])

    prediction.append(d)

    version = args.version
    with open('./prediction/' + str(version) + '.json', "w") as f:
            json.dump(prediction, f)
