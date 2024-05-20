from .model_mask2former import Mask2FormerSegmentor
# from model_maskdino import MaskDINOSegmentor
# from model_maskrcnn import MaskRCNNSegmentor
from .model_spade import SPADEGenerator


def get_segmentor(model_name: str, gpu_id: int = 0, arch_type: str = "swin"):
    if model_name == "MaskDINO":
        return MaskDINOSegmentor(gpu_id, arch_type)
    if model_name == "Mask2Former":
        return Mask2FormerSegmentor(gpu_id, arch_type)
    # if model_name == "Mask-RCNN":
    #     return MaskRCNNSegmentor(gpu_id)
    raise ValueError


def get_generator(model_name, gpu_id, opt):
    if model_name == "SPADE":
        return SPADEGenerator(gpu_id, opt)
    raise ValueError
