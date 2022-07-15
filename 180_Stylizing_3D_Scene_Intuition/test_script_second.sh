for i in {0..111};
do
    CUDA_VISIBLE_DEVICES=0,1 python ddp_test_nerf.py --config configs/test_family_second.txt --render_splits test --style_ID $i;
done
