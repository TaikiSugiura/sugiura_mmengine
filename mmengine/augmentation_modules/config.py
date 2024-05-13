import argparse


def factory():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aspect_ratio',
        default=1.0,
        type=int
    )
    parser.add_argument(
        '--checkpoints_dir',
        default='../SegGen/SPADE/checkpoints'
    )
    parser.add_argument(
        '--name',
        default='coco_pretrained'
    )
    parser.add_argument(
        '--which_epoch',
        default='latest'
    )
    parser.add_argument(
        '--contain_dontcare_label',
        action='store_false'
    )
    parser.add_argument(
        '--init_type',
        default='xavier'
    )
    parser.add_argument(
        '--init_variance',
        default=0.02,
        type=int
    )
    parser.add_argument(
        '--isTrain',
        action='store_true'
    )
    parser.add_argument(
        '--label_nc',
        default=182,
        type=int
    )
    parser.add_argument(
        '--semantic_nc',
        default=184,
        type=int
    )
    parser.add_argument(
        '--netG',
        default='spade'
    )
    parser.add_argument(
        '--ngf',
        default=64,
        type=int
    )
    parser.add_argument(
        '--no_instance',
        action='store_true'
    )
    parser.add_argument(
        '--norm_G',
        default='spectralspadesyncbatch3x3'
    )
    parser.add_argument(
        '--num_upsampling_layers',
        default='normal',
        help=''
    )
    parser.add_argument(
        '--use_vae',
        action='store_true',
        help=''
    )

    parser.add_argument(
        '--bert_embedding_path',
        type=str,
        default='../augmentation_modules/bert/bert_embedding.pickle',
        help=''
    )

    args = parser.parse_args()

    return args
