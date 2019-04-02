import os
import sys
import argparse
import time
import threading

from imgaug import augmenters as iaa
import cv2 as cv
import numpy as np
import mrcnn
import mrcnn.config
import mrcnn.utils
import mrcnn.model

class MRIConfig(mrcnn.config.Config):
    NAME = 'MRI'
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 250
    VALIDATION_STEPS = 50
    BACKBONE = "resnet50"
    NUM_CLASSES = 4
    IMAGE_RESIZE_MODE = 'square'
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    MEAN_PIXEL = np.array([38.23, 39.67, 40.36])
    USE_MINI_MASK = False
    RPN_ANCHOR_SCALES = (32, 64, 128)
    RPN_ANCHOR_RATIOS = [0.5, 1]
    LEARNING_RATE = 0.0001
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

class MRIDataset(mrcnn.utils.Dataset):
    DATASET_NAME = 'mri'
    IMAGE_SIZE = (384, 384)

    image_descr = {}

    def load_mri(self, data_list, classes):
        self.classes = classes

        for i, cl in enumerate(classes):
            self.add_class(self.DATASET_NAME, i + 1, cl)

        lines = [line.strip().split() for line in open(data_list, 'r').readlines()]
        for line in lines:
            image_file, nobjects, descr = line[0], int(line[1]), line[2:]
            self.add_image(self.DATASET_NAME, image_id=image_file, path=image_file)

            self.image_descr[image_file] = []
            for obj_id in range(nobjects):
                x1, y1, x2, y2, cl = descr[obj_id * 5:(obj_id + 1) * 5]
                self.image_descr[image_file].append([[int(x1), int(y1)], [int(x2), int(y2)], cl])

    def load_mask(self, image_id):
        masks = []
        cls = []
        image_path = self.image_info[image_id]['path']

        for object in self.image_descr[image_path]:
            image = cv.imread(image_path)
            mask = np.zeros(image.shape[:2], dtype=np.bool)
            p1, p2, cl = object
            mask[p1[1]:p2[1], p1[0]:p2[0]] = True
            cls.append(self.classes.index(cl) + 1)
            masks.append(mask)

        return np.stack(masks, axis=-1), np.array(cls)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == self.DATASET_NAME:
            return info['id']
        else:
            super(self.__class__, self).image_reference(image_id)


def main(args):
    config = MRIConfig()
    model = mrcnn.model.MaskRCNN(mode='training', config=config, model_dir=args.model_dir)

    if args.weights_path == 'last':
        model.load_weights(model.find_last(), by_name=True)
    else:
        model.load_weights(args.weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask", "rpn_model"])

    data_train = MRIDataset()
    data_train.load_mri(args.train_data_list, args.classes)
    data_train.prepare()

    data_val = MRIDataset()
    data_val.load_mri(args.val_data_list, args.classes)
    data_val.prepare()

    augmentation = iaa.SomeOf((0, None), [
        iaa.Fliplr(0.5),
        iaa.Multiply((0.8, 1.2)),
        iaa.ContrastNormalization((0.8, 1.2)),
    ])

    if args.train_head:
        model.train(data_train, data_val, learning_rate=config.LEARNING_RATE, epochs=20, augmentation=augmentation, layers='heads')
    model.train(data_train, data_val, learning_rate=config.LEARNING_RATE, epochs=50, augmentation=augmentation, layers='all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_list', type=str, default='data/train_list.txt')
    parser.add_argument('--val_data_list', type=str, default='data/val_list.txt')
    parser.add_argument('--model_dir', type=str, default='logs/')
    parser.add_argument('--weights_path', type=str, default='logs/mask_rcnn_coco.h5')
    parser.add_argument('--classes', type=str, default='zdorovyj,patologiyu,patologicheskij')
    parser.add_argument('--train_head', type=bool, default=True)

    args = parser.parse_args()
    args.classes = args.classes.split(',')

    main(args)
