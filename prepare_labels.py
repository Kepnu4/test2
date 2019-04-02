import os
import sys
import argparse
import glob
import random
import xml.etree.ElementTree as ET
import pandas as pd

def write_files(data, path, images_path):
    with open(path, 'w') as fout:
        for file_name, objs in data:
            fout.write(os.path.join(images_path, file_name) + ' {} '.format(len(objs)))
            for obj in objs:
                fout.write('{} {} {} {}'.format(*obj['bb']) + ' {} '.format(obj['name']))
            fout.write('\n')

def find_objs(root):
    objs = []
    for obj in root:
        if not 'object' in obj.tag:
            continue
        obj_dict = {}
        for field in obj:
            if 'name' in field.tag:
                obj_dict['name'] = field.text
            if 'polygon' in field.tag:
                px, py = [], []
                for point in field:
                    if not 'pt' in point.tag:
                        continue
                    x, y = 0, 0
                    for coord in point:
                        if 'x' == coord.tag[-1]:
                            x = int(coord.text)
                        if 'y' in coord.tag[-1]:
                            y = int(coord.text)
                    px.append(x)
                    py.append(y)
                bb = [min(px), min(py), max(px), max(py)]
                obj_dict['bb'] = bb
        objs.append(obj_dict)
    return objs

def main(args):
    descr = None    
    for descr_file in glob.glob(os.path.join(args.descr_dir, '*.csv')):
        cur = pd.read_csv(descr_file)
        descr = pd.concat((descr, cur))
    descr = descr[['ID', 'Файлы', 'XML', 'На срезе визуализируются межпозвоночные диски']] 

    descr = descr[descr['На срезе визуализируются межпозвоночные диски'] == 'Визуализируются (можно размечать)']
    descr = descr.drop('На срезе визуализируются межпозвоночные диски', 1)
    descr = descr.dropna()
    print(descr.shape)


    data_all = []
    has = [0] * 8
    obj_len = []
    classes = ['zdorovyj', 'patologiyu', 'patologicheskij']
    for index, row in descr.iterrows():
        file_name = row['Файлы'].split('/')[0]
        root = ET.fromstring(row['XML'])
        objs = find_objs(root)
        ok = True
        has_classes = 0
        for obj in objs:
            cl = obj['name'].split('-')[-1]
            obj['name'] = cl
            if not cl in classes: 
                ok = False
            else:
                ind = classes.index(cl)
                has_classes |= 1 << ind
                
        if ok and len(objs) > 0: 
            data_all.append((file_name, objs))
            has[has_classes] += 1
            obj_len.append(len(objs))

    print('total {} samples'.format(len(data_all)))
    print('has_classes {}'.format(has))

    random.shuffle(data_all)

    n_test = int(len(data_all) * 0.2)
    n_val = int((len(data_all) - n_test) * 0.2)
    n_train = len(data_all) - n_test - n_val

    write_files(data_all[:n_train], os.path.join(args.out_dir, 'train_list.txt'), args.images_dir)
    write_files(data_all[n_train:n_train + n_val], os.path.join(args.out_dir, 'val_list.txt'), args.images_dir)
    write_files(data_all[n_train+n_val:], os.path.join(args.out_dir, 'train_list.txt'), args.images_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--descr_dir', type=str, default='descr/')
    parser.add_argument('--images_dir', type=str, default='data/images')
    parser.add_argument('--out_dir', type=str, default='data/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    main(args)
    
