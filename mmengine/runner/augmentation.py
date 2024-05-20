from .augmentation_modules import segmentation
from .augmentation_modules import get_segmentor, get_generator

from detectron2.data import MetadataCatalog

from .augmentation_modules import Config
from .augmentation_modules import convert_label
from .augmentation_modules import label2image

import pickle
import random


class Augmentation():
    def __init__(self) -> None:
        self.opt = Config()
        self.seg_model = get_segmentor("Mask2Former", 0)
        self.spade = get_generator("SPADE", 0, self.opt)
        self.metadata = MetadataCatalog.get(
            'coco_2017_train_panoptic_separated'
        )

        with open(self.opt.bert_embedding_path, mode='rb') as f:
            self.category_distance = pickle.load(f)

    def panoptic_segmentation(self, tensor):
        label, instance = segmentation(
            tensor,
            self.seg_model
        )

        return label, instance

    def augmentation(self, tensor):
        def use_augmentation(probability):
            return False if random.random() >= probability else True

        if not use_augmentation(self.opt.probability):
            return tensor
        else:
            # tensor->b*c*H*W
            label, instance = self.panoptic_segmentation(tensor)

            gen = label2image(
                self.spade.model,
                convert_label(label.clone()),
                instance,
                tensor,
                self.opt,
                self.category_distance,
                self.metadata
            )

            return gen
