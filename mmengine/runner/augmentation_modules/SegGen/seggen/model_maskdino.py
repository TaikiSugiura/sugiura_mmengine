import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from typing import Tuple

import sys  # nopep8
sys.path.append("../SegGen/MaskDINO")  # nopep8
from maskdino.config import add_maskformer2_config  # nopep8

sys.path.append("../SegGen/MaskDINO/seggen")
from segmentor import Segmentor


class MaskDINOSegmentor(Segmentor):

    def __init__(self, gpu_id: int, arch_type: str = "swin"):
        super().__init__(gpu_id)
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        if arch_type == "swin":
            cfg.merge_from_file(
                '../SegGen/MaskDINO/configs/coco/panoptic-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml')
            cfg.MODEL.WEIGHTS = \
                '../SegGen/MaskDINO/checkpoints/maskdino_swinl_50ep_300q_hid2048_3sd1_panoptic_58.3pq.pth'
        if arch_type == "resnet":
            cfg.merge_from_file(
                '../SegGen/MaskDINO/configs/coco/panoptic-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml')
            cfg.MODEL.WEIGHTS = \
                '../SegGen/MaskDINO/checkpoints/maskdino_r50_50ep_300q_hid2048_3sd1_panoptic_pq53.0.pth'
        cfg.OUTPUT_DIR = './output'
        cfg.MODEL.DEVICE = 'cuda:' + str(gpu_id)
        cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = True
        cfg.freeze()

        self.model = build_model(cfg)
        DetectionCheckpointer(
            self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS,
                resume=False  # False --> load MODEL.WEIGHTS
        )
        self.model.eval()

    def segmentation(self, batch: torch.tensor) -> Tuple[torch.tensor]:
        """MaskDINO panoptic segmentation

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
