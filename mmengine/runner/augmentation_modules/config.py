from dataclasses import dataclass, field

@dataclass
class Config:
    aspect_ratio: int = 1
    checkpoints_dir: str = '/mnt/HDD10TB-1/sugiura/sugiura_mmengine/mmengine/runner/augmentation_modules/SegGen/SPADE/checkpoints'
    name: str = 'coco_pretrained'
    which_epoch: str = 'latest'
    contain_dontcare_label: bool = False
    init_type: str = 'xavier'
    init_variance: float = 0.02
    isTrain: bool = False
    label_nc: int = 183
    semantic_nc: int = 184
    netG: str = 'spade'
    ngf: int = 64
    no_instance: bool = False
    norm_G: str = 'spectralspadesyncbatch3x3'
    num_upsampling_layers: str = 'normal'
    use_vae: bool = False
    bert_embedding_path: str = '/mnt/HDD10TB-1/sugiura/sugiura_mmengine/mmengine/runner/augmentation_modules/bert/bert_embedding.pickle'
    # crop_size: int = 224
    gpu: int = 0,
    category_sampling: str = 'Semantic'
    shift: bool = True
    paste: bool = True
    probability: float = 0.8
