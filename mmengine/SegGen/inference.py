
import torch
import numpy as np
from cv2 import imread as imread_bgr
# from cv2 import imwrite as imwrite_bgr
from skimage.io import imread as imread_rgb
from skimage.io import imsave as imsave_rgb
from seggenact.model_factory import get_segmentor, get_generator


def main():
    gpu_id = 0
    device = torch.device(
        'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu'
    )

    # seg_model = get_segmentor("MaskDINO", gpu_id)  # "swin"  # input: RGB
    # seg_model = get_segmentor("MaskDINO", gpu_id, "resnet")  # input: RGB

    seg_model = get_segmentor("Mask2Former", gpu_id)  # "swin"  # input: RGB?
    # seg_model = get_segmentor("Mask2Former", gpu_id, "resnet")  # input: RGB?

    # seg_model = get_segmentor("Mask-RCNN", gpu_id)  # input: BGR

    gen_model = get_generator("SPADE", gpu_id)

    img_path = '../000000000785.jpg'
    # img_path = '../2376735269_b1f0c7f83f_z.jpg'
    # img_path = '../SPADE/datasets/UCF_first_frame/val_img/v_GolfSwing_g08_c01_000001.jpg'
    # im = imread_bgr(img_path)  # BGR, HWC
    im = imread_rgb(img_path)  # RGB, HWC

    batch_size = 4
    # making a dummy batch
    batch = torch.tensor(
        np.stack([im.copy()] * batch_size)
    ).permute(0, 3, 1, 2)  # BCHW
    batch.to(device, non_blocking=True)

    with torch.no_grad():
        label_map, instance_map = seg_model.segmentation(batch)
        gen_im = gen_model.generate(label_map, instance_map)  # RGB, BCHW

    for i, (lab, inst, gen) in enumerate(zip(label_map, instance_map, gen_im)):
        imsave_rgb(f'./label_map{i}.png', lab.cpu().numpy().astype(np.uint8))
        imsave_rgb(f'./instance_map{i}.png', inst.cpu().numpy().astype(np.uint8))
        imsave_rgb(f'./gen{i}.png', gen.permute(1, 2, 0).cpu().numpy())  # HWC


if __name__ == "__main__":
    main()
