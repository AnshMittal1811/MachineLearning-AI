# Render images
python experts_test_fast.py model.accelerate.bake=True --config-name $1
# Evaluate metrics
python -m mnh.metric -rewrite output_images/$1/experts_cuda/color/valid/