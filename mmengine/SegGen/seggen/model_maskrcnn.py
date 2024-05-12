import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from typing import Tuple

from .segmentor import Segmentor


class MaskRCNNSegmentor(Segmentor):

    def __init__(self, gpu_id: int):
        super().__init__(gpu_id)
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        cfg.OUTPUT_DIR = './output'
        cfg.MODEL.DEVICE = 'cuda:' + str(gpu_id)
        cfg.freeze()

        self.model = build_model(cfg)
        DetectionCheckpointer(
            self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS,
                resume=False  # False --> load MODEL.WEIGHTS
        )
        self.model.eval()

        metadata = MetadataCatalog.get('coco_2017_train_panoptic_separated')
        self.stuff_dict = {}
        for k, v in metadata.stuff_dataset_id_to_contiguous_id.items():
            self.stuff_dict[v] = k

    def segmentation(self, batch: torch.tensor) -> Tuple[torch.tensor]:
        """Mask R-CNN panoptic segmentation

        Args:
            batch (torch.tensor): image input (BCHW, uint8, 0 to 255), unnormalized BGR

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
                if info['isthing']:
                    new_category_id = info['category_id'] + 1
                else:
                    new_category_id = self.stuff_dict[info['category_id']]
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
