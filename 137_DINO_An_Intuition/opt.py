import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of images')

    # model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'],
                        help="Name of architecture to train.")
    parser.add_argument('--out_dim', default=65536, type=int,
                        help="""Dimensionality of the DINO head output.
                        For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help="stochastic depth rate")
    parser.add_argument('--norm_last_layer', default=False, action="store_true",
                        help="""Whether or not to weight normalize the last layer of the DINO head.
                        Not normalizing leads to better performance but can make the training unstable.
                        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', type=float, default=0.9995,
                        help="""Base EMA parameter for teacher update.
                        The value is increased to 1 during training with cosine schedule.
                        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")

    # augmentation parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relative to the origin image.
                        Used for large global view cropping.""")
    parser.add_argument('--local_crops_number', type=int, default=8,
                        help="""Number of small local views to generate.""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relative to the origin image.
                        Used for small local view cropping of multi-crop.""")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
                        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float,
                        help="""Final value (after linear warmup) of the teacher temperature.
                        For most experiments, anything above 0.07 is unstable. We recommend
                        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')


    # train parameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    # parser.add_argument('--ckpt_path', type=str, default=None,
    #                     help='pretrained checkpoint path to load')

    parser.add_argument('--use_bf16', default=False, action='store_true',
                        help='use bf16 training')
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate at the end of warmup')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='optimizer weight decay')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()