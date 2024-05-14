from argparse import Namespace
from detectron2.data import MetadataCatalog
from numpy.random import default_rng
import numpy as np
import sys
import torch

sys.path.append("../SegGen/SPADE")
sys.path.append("../../SegGen/SPADE")  # nopep8
from models.pix2pix_model import Pix2PixModel  # nopep8

sys.path.append("../SegGen/seggen")
sys.path.append("../../SegGen/seggen")
from generator import Generator


rng = default_rng()


class SPADEGenerator(Generator):

    def __init__(self, gpu_id, opt):
        super().__init__(gpu_id)

        self.model = Pix2PixModel(opt)
        self.model.eval()

    def generate(self,
                 label_map: torch.tensor,
                 instance_map: torch.tensor,
                 ):
        """generating images from label maps

        Args:
            label_map (torch.tensor): label map (BHW, int64)
            instance_map (torch.tensor): instance map (BHW, int32)

        Returns:
            gen_image (torch.tensor): generated RGB image (BCHW, float32), denormalized (0.0 to 1.0)
        """

        metadata = MetadataCatalog.get('coco_2017_train_panoptic_separated')
        stuff_data = metadata.stuff_dataset_id_to_contiguous_id

        panoptic_id = np.array(list(stuff_data.keys()))
        stuff_id = panoptic_id[panoptic_id <= 182]
        other_id = panoptic_id[panoptic_id > 182]

        num_other = len(other_id)
        other_id_replaced = rng.choice(stuff_id[stuff_id != 0], size=num_other)
        # panoptic_id_replaced = np.hstack((stuff_id, other_id_replaced))

        new_0 = rng.choice(stuff_id[stuff_id != 0])

        for old, new in zip(other_id, other_id_replaced):
            label_map[label_map == old] = new
        label_map[label_map == 0] = new_0

        input_dict = {
            'label': label_map.unsqueeze(1) - 1,  # BHW --> B1HW
            'instance': instance_map.unsqueeze(1),  # BHW --> B1HW
            'image': torch.tensor([0]),  # not used
        }
        with torch.no_grad():
            gen_im = self.model(input_dict, mode='inference')
            # BCHW, float32, normalized RGB

        gen_im = torch.clip((gen_im + 1) / 2, 0, 1)  # denormalized

        return gen_im
