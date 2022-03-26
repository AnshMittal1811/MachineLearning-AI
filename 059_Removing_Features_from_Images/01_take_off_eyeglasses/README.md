# Portrait Eyeglasses and Shadow Removal by Leveraging 3D Synthetic Data

Official pytorch implementation of paper "Portrait Eyeglasses and Shadow Removal by Leveraging 3D Synthetic Data". (CVPR 2022)

<div><img src="./img/img.jpg" width="40%"></div>

## Quick Usage

Download the pretrained model from [here](https://drive.google.com/file/d/1Ea8Swdajz2J5VOkaXIw_-pVJk9EWYrpx/view?usp=sharing), and put it in the "ckpt" directory. Then run the following script:

	python easy_use.py

 As default, the input images are expected to be put in "data" directory and the results will be saved in "results" directory. You can also change them by different arguments.

## Synthetic Dataset

Download the synthetic dataset in [Google Drive](https://drive.google.com/file/d/1X1qkozQbVyz5lUA8xd-lYfy1jauOji46/view?usp=sharing).

The meaning of filename: img-[Glass]-[Subject]-[Expression]-[Node Type]-[HDR]-[HDR Rotation]-[Image Type]

	Glass: Glass ID
	Subject: Subject ID
	Expression: Expression Name
	Node Type: 4 kinds of floating nodes on different positions of the nose. {0, 1, 2, 3}
	HDR: HDR Lighting Name
	HDR Rotation: The angle of vertical axis in degree
	Image Type: Images with different visibility combinations of eyeglasses and shadows or segmentations
	
To get the label of shadow segmentation:

	python generate_shadow_label.py --syndata_dir <synthetic_data_dir>


## Citation

If our paper helps your research, please cite it in your publications:

	To be updated