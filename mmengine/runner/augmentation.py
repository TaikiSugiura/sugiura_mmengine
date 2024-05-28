from .augmentation_modules import segmentation
from .augmentation_modules import get_segmentor, get_generator

from detectron2.data import MetadataCatalog

from .augmentation_modules import Config
from .augmentation_modules import convert_label
from .augmentation_modules import label2image

import pickle
import random
import torch
from torchvision.utils import save_image
import os


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

        def save_tensor_as_images(tensor, output_dir, file_prefix="image"):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save each image in the batch
            for i in range(tensor.size(0)):
                save_path = os.path.join(output_dir, f"{file_prefix}_{i}.png")
                save_image(tensor[i], save_path)

        if not use_augmentation(self.opt.probability):
            print(torch.max(tensor))
            save_tensor_as_images(
                (tensor.clone())/255,
                "/mnt/HDD10TB-1/sugiura/2024_sugiura_mmpose/tools/videos/not",
            )
            return tensor
        else:
            # tensor->b*c*H*W
            label, instance = self.panoptic_segmentation(tensor.clone())

            gen = label2image(
                self.spade.model,
                convert_label(label.clone()),
                instance,
                (tensor.clone())/255,
                self.opt,
                self.category_distance,
                self.metadata
            )

            # gen = ((gen+1)/2).clamp(0, 1).type(torch.FloatTensor)
            save_tensor_as_images(
                (tensor.clone())/255,
                "/mnt/HDD10TB-1/sugiura/2024_sugiura_mmpose/tools/videos/original",
            )
            save_tensor_as_images(
                gen.clone(),
                "/mnt/HDD10TB-1/sugiura/2024_sugiura_mmpose/tools/videos/gen",
            )

            return gen * 255
