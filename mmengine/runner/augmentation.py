import sys
sys.path.append("../SegGen/seggen")
from segmentation import segmentation
from model_factory import get_segmentor


class Augmentation():

    def __init__(self, gpu) -> None:
