LOG_DIR="log"
DATA_DIR="dataset/data"
RELOAD_MODEL_PATH="./trained_models/3dsnet_models/chairs/meshflow/3dsnet"
RELOAD_POINTNET_PATH="aux_models/pointnet_autoencoder_25_squares.pth"

python train.py \
--run_single_eval \
--data_dir=$DATA_DIR \
--dir_name=$LOG_DIR \
--reload_model_path="$RELOAD_MODEL_PATH" \
--reload_pointnet_path=$RELOAD_POINTNET_PATH \
--class_choice "hippos" "horses" \
--generator_norm "bn" \
--discriminator_norm "bn" \
--discriminator_activation "relu" \
--dis_bottleneck_size 1024 \
--style_bottleneck_size 512 \
--generator_lrate 0.001 \
--discriminator_lrate 0.004 \
--batch_size=4 \
--batch_size_test=2 \
--generator_update_skips=1 \
--discriminator_update_skips=1 \
--num_layers=2 \
--num_layers_style=1 \
--nb_primitives=25 \
--template_type=SQUARE \
--weight_chamfer=10 \
--weight_cycle_chamfer=0 \
--weight_adversarial=1 \
--weight_content_reconstruction=1 \
--weight_style_reconstruction=1 \
--lr_decay_1=120 \
--lr_decay_2=140 \
--lr_decay_3=145 \
--decode_style \
--share_decoder \
--share_content_encoder \
--share_discriminator_encoder \
--gan_type "lsgan" \
--adaptive \
--decoder_type "atlasnet" \
--num_pairs=19 \
--noise_magnitude=0.1 \
--no_lpips \
#--perceptual_by_layer \
#--no_quantitative_eval
