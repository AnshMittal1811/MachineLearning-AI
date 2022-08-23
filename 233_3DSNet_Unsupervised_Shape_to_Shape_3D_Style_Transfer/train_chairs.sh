LOG_DIR="log"
DATA_DIR="dataset/data"

GENERATOR_NORM="bn"
DISCRIMINATOR_NORM="bn"
DISCRIMINATOR_ACTIVATION="relu"
NUM_LAYERS=2
W_CHAMFER=1.0
W_CYCLE_CHAMFER=0.1
W_ADVERSARIAL=1.0
W_PERCEPTUAL=0.0
W_CONTENT_REC=0.1
W_STYLE_REC=0.1
GAN_TYPE="lsgan"
DECODER_TYPE="meshflow"
NUMBER_POINTS=2500
BATCH_SIZE=16
GEN_LR=0.001
DIS_LR=0.004
W_MULTISCALE_1=0.0
W_MULTISCALE_2=0.0
W_MULTISCALE_3=1.0
LR_DECAY_1=120
LR_DECAY_2=140
LR_DECAY_3=145
NEPOCH=180

python train.py \
--data_dir=$DATA_DIR \
--dir_name=$LOG_DIR \
--dataset "ShapeNet" \
--family "chair" \
--class_choice "armchair" "straight chair,side chair" \
--generator_norm "bn" \
--discriminator_norm "bn" \
--discriminator_activation=$DISCRIMINATOR_ACTIVATION \
--dis_bottleneck_size 1024 \
--batch_size=$BATCH_SIZE \
--generator_update_skips=1 \
--discriminator_update_skips=1 \
--num_layers=2 \
--num_layers_style=1 \
--nb_primitives=25 \
--template_type=SQUARE \
--weight_chamfer=$W_CHAMFER \
--weight_cycle_chamfer=$W_CYCLE_CHAMFER \
--weight_adversarial=$W_ADVERSARIAL \
--weight_perceptual=$W_PERCEPTUAL \
--weight_content_reconstruction=$W_CONTENT_REC \
--weight_style_reconstruction=$W_STYLE_REC \
--lr_decay_1=$LR_DECAY_1 \
--lr_decay_2=$LR_DECAY_2 \
--lr_decay_3=$LR_DECAY_3 \
--nepoch=$NEPOCH \
--generator_lrate=$GEN_LR \
--discriminator_lrate=$DIS_LR \
--decode_style \
--gan_type "lsgan" \
--adaptive \
--share_decoder \
--share_content_encoder \
--share_discriminator_encoder \
--reload_pointnet_path trained_models/pointnet_autoencoder_25_squares.pth \
--perceptual_by_layer \
--number_points=$NUMBER_POINTS \
--decoder_type=$DECODER_TYPE \
--w_multiscale_1=$W_MULTISCALE_1 \
--w_multiscale_2=$W_MULTISCALE_2 \
--w_multiscale_3=$W_MULTISCALE_3 \
--save_optimizers \
#--multiscale_loss \
#--share_style_mlp \
#--share_style_encoder \

