from augmentation_modules import segmentation
from .augmentation_modules.SegGen.seggen.model_factory import get_segmentor, get_generator

from detectron2.data import MetadataCatalog

from config import factory
from convert_label import convert_label
from label2im import label2image

import pickle


class Augmentation():
    def __init__(self) -> None:
        self.opt = factory()
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

    def augmentation(self, tensor, probability):
        print(probability)
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
