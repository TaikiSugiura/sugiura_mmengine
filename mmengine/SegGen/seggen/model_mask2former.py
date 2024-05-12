import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from typing import Tuple

import sys
sys.path.append("../SegGen/Mask2Former")
sys.path.append("../../SegGen/Mask2Former")
from mask2former import add_maskformer2_config

sys.path.append("../SegGen/seggen")
sys.path.append("../../SegGen/seggen")
from segmentor import Segmentor


class Mask2FormerSegmentor(Segmentor):

    def __init__(self, gpu_id: int, arch_type: str = "swin"):
        super().__init__(gpu_id)
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        if arch_type == "swin":
            cfg.merge_from_file(
                '../SegGen/Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml')
            cfg.MODEL.WEIGHTS = \
                '../SegGen/Mask2Former/checkpoints/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
        if arch_type == "resnet":
            cfg.merge_from_file(
                '../SegGen/Mask2Former/configs/coco/panoptic-segmentation/maskformer2_R101_bs16_50ep.yaml')
            cfg.MODEL.WEIGHTS = \
                '../SegGen/Mask2Former/checkpoints/coco/panoptic/maskformer2_R101_bs16_50ep/model_final_b807bd.pkl'
        cfg.OUTPUT_DIR = './output'
        cfg.MODEL.DEVICE = 'cuda:' + str(gpu_id)
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
        cfg.freeze()

        self.model = build_model(cfg)
        DetectionCheckpointer(
            self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS,
                resume=False  # False --> load MODEL.WEIGHTS
        )
        self.model.eval()

    def segmentation(self, batch: torch.tensor) -> Tuple[torch.tensor]:
        """Mask2Former panoptic segmentation

        Args:
            batch (torch.tensor): image input (BCHW, uint8, 0 to 255), unnormalized RGB

        Returns:
            label_map (torch.tensor): label map (BHW, int64)
            instance_map (torch.tensor): instance map (BHW, int32)
        """
        super().segmentation(batch)

        batched_inputs = [
            {"image": b} for b in batch
        ]  # size B list of dict of CHW tensor
        with torch.no_grad():
            batched_outputs = self.model(batched_inputs)
            # size B list of dict

        label_map_list = []
        instance_map_list = []
        for output in batched_outputs:
            panoptic_seg, segments_info = output["panoptic_seg"]
            # panoptic_seg: HW

            segments_info_list = []
            for i in range(len(segments_info)):
                info = segments_info[i]
                new_category_id = info['category_id'] + 1
                segments_info_list.append(new_category_id)

            segments_info_list.insert(0, 0)  # add value 0 at position 0
            segments_info_list = torch.tensor(
                segments_info_list,
                device=panoptic_seg.device)

            label_map = segments_info_list[panoptic_seg.long()]
            label_map_list.append(label_map)
            instance_map_list.append(panoptic_seg)

        label_map = torch.stack(label_map_list)
        instance_map = torch.stack(instance_map_list)
        return label_map, instance_map  # no gradients here
