import torch


def segmentation(opt, target, seg_model):

    with torch.no_grad():
        label_map, instance_map = seg_model.segmentation(target)

    return label_map.unsqueeze(1), instance_map.unsqueeze(1)
