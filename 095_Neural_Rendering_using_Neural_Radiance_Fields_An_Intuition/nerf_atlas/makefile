PHONY:

clean:
	-@rm outputs/*.png
	-@rm outputs/results.txt
	@-rm outputs/*.mp4

original: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --epochs 80_000 --save models/lego.pt \
	--near 2 --far 6 --batch-size 4 --crop-size 16 --model plain -lr 5e-4 \
	--loss-fns l2 --refl-kind pos #--load models/lego.pt #--omit-bg

coarse_fine: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --epochs 80_000 --save models/lego.pt \
	--near 2 --far 6 --batch-size 4 --crop-size 10 --model coarse_fine -lr 3e-4 \
	--loss-fns l2 --refl-kind view #--load models/lego.pt #--omit-bg

volsdf: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 200 --epochs 50_000 --crop-size 24 --test-crop-size 25 \
	--near 2 --far 6 --batch-size 2 --model volsdf --sdf-kind mlp \
	-lr 3e-4 --loss-window 750 --valid-freq 250 --loss-fns l2 \
	--save-freq 2500 --sigmoid-kind upshifted --replace refl --notraintest \
	--depth-images --refl-kind pos --light-kind field --depth-query-normal --normals-from-depth \
	--save models/lego_volsdf.pt --load models/lego_volsdf.pt

voxel: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --epochs 5_000 --nosave --loss-window 1000 --save-freq 2500 \
	--near 2 --far 6 --batch-size 10 --crop-size 40 --model voxel -lr 1e-2 \
	--loss-fns l2

dyn_voxel: clean
	python3 runner.py -d data/dynamic/${dnerf_dataset}/ --data-kind dnerf \
	--size 128 --epochs 50_000 --save models/dvoxel_${dnerf_dataset}.pt --loss-window 1000 --save-freq 2500 \
	--near 2 --far 6 --batch-size 2 --crop-size 44 --model voxel --dyn-model voxel -lr 1e-2 \
	--loss-fns l2 --test-crop-size 64 --depth-images --flow-map --spline 4 --steps 80 \
  --voxel-tv-sigma 1e-3 --voxel-tv-rgb 1e-4 --voxel-tv-bezier 1e-4 --voxel-tv-rigidity 1e-4 \
  --higher-end-chance 1 --offset-decay 30 --ffjord-div-decay 0.3 \
  --sigmoid-kind upshifted --voxel-random-spline-len-decay 1e-5 --replace sigmoid \
  --notraintest --seed -1 --refl-kind pos-linear-view \
  --rigidity-map --load models/dvoxel_${dnerf_dataset}.pt

volsdf_with_normal: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 192 --epochs 50_000 --crop-size 16 \
	--near 2 --far 6 --batch-size 4 --model volsdf --sdf-kind mlp \
	-lr 1e-3 --loss-window 750 --valid-freq 250 --nosave \
	--sdf-eikonal 0.1 --loss-fns l2 --save-freq 5000 --sigmoid-kind fat \
	--refl basic --normal-kind elaz --light-kind point

rusin: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --epochs 50_000 --crop-size 10 \
	--near 2 --far 6 --batch-size 3 --model volsdf --sdf-kind mlp \
	-lr 1e-3 --loss-window 750 --valid-freq 250 \
	--sdf-eikonal 0.1 --loss-fns l2 --save-freq 5000 --sigmoid-kind fat \
	--nosave --light-kind field --refl-kind rusin

nerfactor_ds := pinecone
nerf-sh: clean
	python3 runner.py -d data/nerfactor/${nerfactor_ds}/ \
	--data-kind original --size 128 --epochs 0 --crop-size 25 \
	--near 2 --far 6 --batch-size 5 --model plain \
	-lr 1e-3 --loss-window 750 --valid-freq 250 \
	--loss-fns l2 --save-freq 5000 --sigmoid-kind leaky_relu \
	--refl-kind sph-har --save models/${nerfactor_ds}-sh.pt \
  --notest --depth-images --normals-from-depth \
  --load models/${nerfactor_ds}-sh.pt

nerfactor_volsdf: clean
	python3 runner.py -d data/nerfactor/${nerfactor_ds}/ \
	--data-kind original --size 256 --epochs 50_000 --crop-size 11 \
	--near 2 --far 6 --batch-size 4 --model volsdf --sdf-kind mlp \
	-lr 1e-4 --loss-window 750 --valid-freq 250 --light-kind field --occ-kind all-learned \
	--loss-fns l2 rmse --color-spaces rgb xyz hsv --save-freq 2500 --sigmoid-kind leaky_relu \
	--refl-kind diffuse --save models/${nerfactor_ds}-volsdf.pt --depth-images \
	--normals-from-depth --notest --sdf-eikonal 1e-t \
  --load models/${nerfactor_ds}-volsdf.pt --depth-query-normal \
  #--smooth-normals 1e-2 --smooth-eps-rng \

nerfactor_volsdf_direct: clean
	python3 runner.py -d data/nerfactor/${nerfactor_ds}/ \
	--data-kind original --size 128 --crop-size 14 --epochs 50_000 \
	--near 2 --far 6 --batch-size 4 --model volsdf --sdf-kind siren \
	-lr 1e-3 --loss-window 750 --valid-freq 500 \
	--loss-fns l2 --save-freq 2500 --occ-kind all-learned \
	--refl-kind rusin --save models/${nerfactor_ds}-volsdfd.pt --light-kind field \
  --color-spaces rgb --depth-images --normals-from-depth \
  --sdf-eikonal 1e-2 --smooth-normals 1e-2 --smooth-eps-rng \
  --sigmoid-kind normal --notest \
  --load models/${nerfactor_ds}-volsdfd.pt


# TODO fix this dataset, using it is a complete trash-fire
food: clean
	python3 runner.py -d data/food/ --data-kind shiny --size 64 \
	--epochs 50_000  --save models/food.pt --model ae --batch-size 4 \
	--crop-size 24 --near 2 --far 6 -lr 5e-4 --no-sched --valid-freq 499 \


dnerf_dataset = bouncingballs
dnerf: clean
	python3 -O runner.py -d data/dynamic/${dnerf_dataset}/ --data-kind dnerf --size 64 \
	--epochs 50_000 --save models/dyn_${dnerf_dataset}.pt --model plain --batch-size 2 \
	--crop-size 18 --near 2 --far 6 -lr 1e-3 --valid-freq 500 --spline 6 \
  --loss-window 2000 --loss-fns l2 --test-crop-size 48 --depth-images --save-freq 2500 \
  --flow-map --dyn-model plain --rigidity-map --refl-kind pos-linear-view \
  --higher-end-chance 1 --offset-decay 30 --ffjord-div-decay 0.3 \
  --sigmoid-kind upshifted --notraintest --opt-step 3 --dyn-refl-latent 3 \
  #--load models/dyn_${dnerf_dataset}.pt

dnerf_original: clean
	python3 -O runner.py -d data/dynamic/${dnerf_dataset}/ --data-kind dnerf --size 32 \
	--epochs 50_000 --save models/dyn_n_${dnerf_dataset}.pt --model plain --batch-size 2 \
	--crop-size 20 --near 2 --far 6 -lr 1e-3 --valid-freq 500 \
  --sigmoid-kind fat --loss-window 2000 --loss-fns l2 \
  --render-over-time 8 --notraintest --test-crop-size 64 --depth-images --save-freq 2500 \
  --flow-map --dyn-model plain --rigidity-map --refl-kind pos \
  --higher-end-chance 1 --offset-decay 30 --ffjord-div-decay 0.3 \
  --sigmoid-kind upshifted \
  --load models/dyn_n_${dnerf_dataset}.pt

dnerf_volsdf: clean
	python3 runner.py -d data/dynamic/$(dnerf_dataset)/ --data-kind dnerf --size 128 \
	--epochs 50_000  --save models/dvs_$(dnerf_dataset).pt --model volsdf --sdf-kind mlp \
  --batch-size 2 --crop-size 16 --near 2 --far 6 -lr 3e-4 --valid-freq 500 --spline 6 \
  --refl-kind pos-linear-view --sigmoid-kind upshifted --loss-window 1000 --dyn-model plain \
  --notraintest --render-over-time 12 --loss-fns l2 --save-freq 1000 \
  --load models/dvs_$(dnerf_dataset).pt --sdf-eikonal 1e-5

gibson: clean
	python3 runner.py -d data/gibson_dataset/ --data-kind dnerf --size 256 \
	--epochs 100_000 --save models/gibson.pt --model plain --spline 12 \
  --batch-size 1 --crop-size 24 --near 1e-3 --far 8 -lr 1e-4 --valid-freq 500 \
  --refl-kind pos-linear-view --sigmoid-kind fat --loss-window 1000 --dyn-model plain \
  --loss-fns l2 --save-freq 2500 --depth-images --rigidity-map --flow-map --opt-step 5 \
  --offset-decay 30 --ffjord-div-decay 0.3 --notraintest --test-crop-size 48 \
  --load models/gibson.pt

long_dnerf: clean
	python3 runner.py -d data/dynamic/${dnerf_dataset}/ --data-kind dnerf --size 64 \
	--epochs 50_000 --save models/ldyn_${dnerf_dataset}.pt --model plain --batch-size 1 \
	--crop-size 20 --near 2 --far 6 -lr 3e-4 --valid-freq 500 --spline 5 \
  --refl-kind pos --sigmoid-kind upshifted --loss-window 500 --loss-fns l2 fft \
  --render-over-time 8 --notraintest --test-crop-size 64 --depth-images \
  --dyn-model long --clip-gradients 1 \
  --load models/ldyn_${dnerf_dataset}.pt

dnerf_gru: clean
	python3 runner.py -d data/dynamic/${dnerf_dataset}/ --data-kind dnerf --size 64 \
	--epochs 80_000  --save models/djj_gru_ae.pt --model ae --batch-size 2 \
	--crop-size 24 --near 2 --far 6 -lr 1e-3 --no-sched --valid-freq 499 \
  --gru-flow #--load models/djj_gru_ae.pt

# testing out dnerfae dataset on dnerf
dnerf_dyn: clean
	python3 runner.py -d data/dynamic/jumpingjacks/ --data-kind dnerf --size 64 \
	--epochs 80_000  --save models/djj_gamma.pt --model ae --batch-size 1 \
	--crop-size 40 --near 2 --far 6 -lr 5e-4 --no-sched --valid-freq 499 --dyn-model ae \
	--serial-idxs --time-gamma --loss-window 750 #--load models/djj_gamma.pt

sdf: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 128 --epochs 5000 --save models/lego_sdf.pt --crop-size 128 \
	--near 2 --far 6 --batch-size 6 --model sdf --sdf-kind mlp \
  -lr 5e-4 --loss-window 750 --valid-freq 100 \
  --nosave --sdf-eikonal 0.1 --loss-fns l2 --save-freq 2500

scan_number := 83
dtu: clean
	python3 runner.py -d data/DTU/scan$(scan_number)/ --data-kind dtu \
	--size 192 --epochs 50000 --save models/dtu$(scan_number).pt --save-freq 5000 \
	--near 0.3 --far 1.8 --batch-size 3 --crop-size 18 --model volsdf -lr 3e-4 \
	--loss-fns l2 --valid-freq 499 --sdf-kind siren --opt-step 5 \
	--loss-window 1000 --sdf-eikonal 0.1 --sigmoid-kind fat #--load models/dtu$(scan_number).pt

dtu_diffuse: clean
	python3 runner.py -d data/DTU/scan$(scan_number)/ --data-kind dtu \
	--size 128 --epochs 10_000 --save models/dtu_diffuse_$(scan_number).pt \
	--near 0.4 --far 2 --batch-size 2 --crop-size 12 --test-crop-size 32 \
  --model volsdf -lr 3e-4 --light-kind field \
	--valid-freq 500 --sdf-kind siren --refl-kind diffuse --occ-kind all-learned \
  --depth-images --depth-query-normal --normals-from-depth --msssim-loss \
  --save-freq 2500 --notraintest --loss-window 1000 --sdf-eikonal 1e-5 --loss-fns l2 \
  --sigmoid-kind upshifted_softplus \
  --load models/dtu_diffuse_$(scan_number).pt

dtu_diffuse_lit: clean
	python3 -O runner.py -d data/DTU/scan$(scan_number)/ --data-kind dtu \
	--size 200 --epochs 1 --nosave \
	--near 0.01 --far 1.3 --batch-size 1 --crop-size 16 --test-crop-size 40 \
  -lr 5e-4 --light-kind point --point-light-position 0 -8 8 --light-intensity 4000 \
	--valid-freq 500 --sdf-kind mlp --refl-kind diffuse --all-learned-to-joint \
  --save-freq 2500 --notraintest --replace light \
  --load models/dtu_diffuse_$(scan_number).pt --render-frame 13

# -- Begin NeRV tests

# hotdogs | armadillo, fun datasets :)
nerv_dataset := armadillo
nerv_point: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind curl-mlp \
	--save models/nerv_${nerv_dataset}.pt \
	--size 200 --crop-size 11 --epochs 50_000 --loss-window 1500 \
	--near 2 --far 6 --batch-size 4 -lr 3e-4 --refl-kind rusin \
	--light-kind dataset --seed -1 --loss-fns l2 rmse \
	--valid-freq 500 --save-freq 2500 --occ-kind all-learned \
  --depth-images --depth-query-normal --skip-loss 100 \
  --notraintest --has-multi-light --all-learned-occ-kind pos-elaz \
  --normals-from-depth --msssim-loss --display-smoothness --gamma-correct \
  --load models/nerv_${nerv_dataset}.pt # --all-learned-to-joint \
  #--smooth-normals 1e-5 --smooth-eps 1e-3 --smooth-surface 1e-5 \

nerv_point_diffuse: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_diffuse_${nerv_dataset}.pt --nosave \
	--size 100 --crop-size 11 --epochs 25_000  --loss-window 1500 \
	--near 2 --far 6 --batch-size 4 -lr 3e-4 --refl-kind diffuse \
	--sdf-eikonal 1 --light-kind dataset --seed -1 \
	--loss-fns l2 rmse --valid-freq 500 --save-freq 2500 \
  --occ-kind learned-const --replace occ \
  --color-spaces rgb xyz hsv --depth-images --depth-query-normal \
  --sigmoid-kind leaky_relu --skip-loss 100 \
  --notraintest --clip-gradients 1 \
  --normals-from-depth --msssim-loss --depth-query-normal --display-smoothness \
  --load models/nerv_diffuse_${nerv_dataset}.pt

nerv_point_diffuse_unknown_lighting:
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_diff_ul_${nerv_dataset}.pt \
	--size 200 --crop-size 11 --epochs 50_000  --loss-window 1500 \
	--near 2 --far 6 --batch-size 4 -lr 1e-4 --refl-kind diffuse \
	--sdf-eikonal 1 --light-kind field --seed -1 \
	--loss-fns l2 rmse --valid-freq 500 --save-freq 2500 --occ-kind all-learned \
  --color-spaces rgb xyz hsv --depth-images --depth-query-normal \
  --sigmoid-kind sin --skip-loss 100 \
  --notraintest --replace sigmoid --clip-gradients 1 \
  --normals-from-depth --msssim-loss --depth-query-normal --display-smoothness \
  #--load models/nerv_diff_ul_${nerv_dataset}.pt

nerv_point_diffuse_to_learned: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
  --name learned_from_diffuse${nerv_dataset} \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_from_diffuse_${nerv_dataset}.pt \
	--size 200 --crop-size 14 --epochs 50_000  --loss-window 1500 \
	--near 2 --far 6 --batch-size 4 -lr 8e-4 \
	--sdf-eikonal 1 --light-kind dataset --seed -1 \
	--loss-fns l2 rmse --valid-freq 500 --save-freq 2500 --occ-kind all-learned \
  --color-spaces rgb hsv xyz --depth-images --depth-query-normal \
  --sigmoid-kind tanh --skip-loss 100 \
  --notraintest \
  --normals-from-depth --msssim-loss --depth-query-normal --display-smoothness \
  --train-parts refl occ --convert-analytic-to-alt \
  --load models/nerv_diffuse_${nerv_dataset}.pt \
  #--load models/nerv_from_diffuse_${nerv_dataset}.pt

# converts a model to a pathtraced model
nerv_point_alt_to_pathtrace: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
  --name pathtrace_${nerv_dataset} \
	--data-kind nerv_point \
	--save models/nerv_path_final_${nerv_dataset}.pt \
	--size 32 --crop-size 6 --epochs 50_000  --loss-window 1500 \
	--near 2 --far 6 --batch-size 3 -lr 2e-4 \
	--sdf-eikonal 1 --light-kind dataset --seed -1 \
	--loss-fns l2 rmse --valid-freq 500 --save-freq 2500 --occ-kind all-learned \
  --color-spaces rgb hsv xyz --depth-images --depth-query-normal --skip-loss 100 \
  --notraintest --normals-from-depth --msssim-loss --depth-query-normal --display-smoothness \
  --volsdf-direct-to-path \
  --load models/nerv_diffuse_${nerv_dataset}.pt \
  #--load models/nerv_from_diffuse_${nerv_dataset}.pt

nerv_point_final: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
  --name final_${nerv_dataset} \
	--data-kind nerv_point \
	--load models/nerv_path_final_${nerv_dataset}.pt \
	--size 200 --crop-size 6 --epochs 0 \
	--near 2 --far 6 --batch-size 3 --light-kind dataset \
  --depth-images --depth-query-normal \
  --notraintest --normals-from-depth --msssim-loss --depth-query-normal \
  --nosave

nerv_point_sdf: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model sdf --sdf-kind mlp \
	--save models/nerv_sdf_${nerv_dataset}.pt \
	--size 200 --crop-size 32 --epochs 20_000 --loss-window 500 \
	--near 2 --far 6 --batch-size 3 -lr 5e-4 --refl-kind multi_rusin \
	--sdf-eikonal 0.1 --light-kind dataset \
	--loss-fns l2 l1 rmse --valid-freq 250 --save-freq 1000 --seed -1 \
	--occ-kind learned --sdf-isect-kind bisect \
  --integrator-kind direct --color-spaces rgb hsv xyz \
	--load models/nerv_sdf_${nerv_dataset}.pt

nerv_point_alternating: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_alt_${nerv_dataset}.pt \
	--size 200 --crop-size 12 --epochs 50_000 --loss-window 500 \
	--near 2 --far 6 --batch-size 4 -lr 5e-4 --refl-kind rusin \
	--sdf-eikonal 0.1 --light-kind dataset \
	--loss-fns l1 l2 --valid-freq 250 --save-freq 2500 --seed -1 \
	--occ-kind all-learned --volsdf-alternate --notraintest \
	--sdf-isect-kind bisect --color-spaces rgb hsv xyz \
	--load models/nerv_alt_${nerv_dataset}.pt

# experimenting with path tracing and nerv
nerv_point_path: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_path_${nerv_dataset}.pt \
	--size 32 --crop-size 6 --epochs 20_000 --loss-window 500 \
	--near 2 --far 6 --batch-size 3 -lr 5e-4 --refl-kind rusin \
	--sdf-eikonal 0.1 --light-kind dataset --seed -1 \
	--loss-fns l2 --valid-freq 500 --occ-kind all-learned \
  --color-spaces rgb --save-freq 1000 \
  --integrator-kind path --depth-images --notraintest --skip-loss 500 \
  --smooth-eps 2e-3 --smooth-occ 1e-3 --sigmoid-kind upshifted_softplus \
  --normals-from-depth --msssim-loss --depth-query-normal --display-smoothness \
  --smooth-normals 1e-3 --normals-from-depth \
  #--load models/nerv_path_${nerv_dataset}.pt #--path-learn-missing

nerv_point_subrefl: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_weighted_${nerv_dataset}.pt \
	--size 200 --crop-size 12 --epochs 30_000 --loss-window 1500 \
	--near 2 --far 6 --batch-size 4 -lr 3e-4 --refl-kind weighted \
	--sdf-eikonal 0.1 --light-kind dataset --seed -1 \
	--loss-fns l2 rmse --valid-freq 500 --occ-kind all-learned \
  --color-spaces rgb hsv xyz \
  --notraintest --omit-bg \
  --load models/nerv_weighted_${nerv_dataset}.pt

nerv_point_fourier: clean
	python3 runner.py -d data/nerv_public_release/${nerv_dataset}/ \
	--data-kind nerv_point --model volsdf --sdf-kind mlp \
	--save models/nerv_fourier_${nerv_dataset}.pt \
	--size 200 --crop-size 14 --epochs 50_000 --loss-window 1500 \
	--near 2 --far 6 --batch-size 4 -lr 8e-4 --refl-kind fourier \
	--sdf-eikonal 0.1 --light-kind dataset --seed -1 \
	--loss-fns l2 rmse --valid-freq 500 --occ-kind all-learned \
  --color-spaces rgb hsv xyz \
  --notraintest --depth-images \
  --smooth-normals 1e-3 --smooth-eps 1e-3 --notraintest \
  --normals-from-depth --msssim-loss --depth-query-normal --display-smoothness \
  --smooth-surface 1e-3 --sdf-isect-kind bisect \
  --draw-colormap \
  --load models/nerv_fourier_${nerv_dataset}.pt

# -- End NeRV tests

test_original: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --epochs 0 --near 2 --far 6 --batch-size 5 \
  --crop-size 26 --load models/lego.pt

bendy: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --epochs 80_000 --save models/bendy_lego.pt --bendy \
	--near 2 --far 6 --batch-size 4 --crop-size 16 --model plain -lr 1e-3 \
	--loss-fns l2 --refl-kind pos --load models/bendy_lego.pt #--omit-bg

ae: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --epochs 80_000 --save models/lego_ae.pt \
	--near 2 --far 6 --batch-size 5 --crop-size 20 --model ae -lr 1e-3 \
	--valid-freq 499 --no-sched --loss-fns l2 #--load models/lego_ae.pt #--omit-bg

og_upsample: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--render-size 16 --size 64 --epochs 80_000 --save models/lego_up.pt \
	--near 2 --far 6 --batch-size 4 --model plain -lr 5e-4 \
	--loss-fns l2 --valid-freq 499 --no-sched --neural-upsample --nosave

rig_nerf: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 128 --epochs 50_000 --save models/rig_lego.pt --seed -1 \
	--near 2 --far 6 --batch-size 4 --crop-size 16 --model rig -lr 2e-4 \
  --test-crop-size 48 --save-freq 2500 --notraintest --depth-images --sigmoid-kind fat \
	--loss-fns fft l2 --refl-kind view --load models/rig_lego.pt #--omit-bg

dyn_rig_nerf: clean
	python3 -O runner.py -d data/dynamic/${dnerf_dataset}/ --data-kind dnerf \
	--size 100 --epochs 100_000 --save models/dyn_rig_${dnerf_dataset}.pt --seed -1 \
	--near 2 --far 6 --batch-size 4 --crop-size 16 --model rig --dyn-model rig -lr 1e-4 \
  --test-crop-size 48 --save-freq 2500 --notraintest --depth-images --sigmoid-kind fat \
	--loss-fns fft --refl-kind pos --render-over-time 8 \
  --loss-window 500 --spline 5 --save-freq 1000 \
  --load models/dyn_rig_${dnerf_dataset}.pt

# [WIP]
pixel_single: clean
	python3 runner.py -d data/celeba_example.jpg --data-kind pixel-single --render-size 16 \
  --crop-size 16 --save models/celeba_sp.pt --mip cylinder --model ae


# scripts

gan_sdf:
	python3 gan_sdf.py --epochs 15_000 --num-test-samples 256 --sample-size 1000 \
  --eikonal-weight 1 --nosave --noglobal --render-size 256 --crop-size 128 --load

volsdf_gan:
	python3 gan_sdf.py --epochs 25_000 --num-test-samples 256 --sample-size 900 \
  --eikonal-weight 0 --target volsdf --volsdf-model models/lego_volsdf.pt \
	--refl-kind pos --bounds 2 --noglobal --render-size 256 --crop-size 128 --G-model mlp \
  --load --G-rep 3

volsdf_gan_no_refl:
	python3 gan_sdf.py --epochs 25_000 --num-test-samples 256 --sample-size 1024 \
  --eikonal-weight 1e-2 --target volsdf --volsdf-model models/lego_volsdf.pt \
	--bounds 1.5 --noglobal --render-size 128 --G-model mlp

project_pts: clean
	python3 scripts/project_pts.py -d data/nerf_synthetic/lego/ --model models/rig_lego.pt \
  --size 128

psp: clean
	python3 scripts/rig_physics.py -d data/nerf_synthetic/lego/ --model models/rig_lego.pt

mpi: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 64 --epochs 30_000 --save models/lego_mpi.pt \
	--near 2 --far 6 --batch-size 4 --crop-size 50 --model mpi -lr 1e-4 \
	--loss-fns l2 --refl-kind pos --train-imgs 1

# evaluates the reflectance of a rusin model
eval_rusin:
	python3 eval_rusin.py --refl-model models/nerv_hotdogs.pt

fieldgan: clean
	python3 fieldgan.py --image data/mondrian.jpg --epochs 2500
	#python3 fieldgan.py --image data/food/images/IMG_1268.png --epochs 2500

# experimental things, none of it is guaranteed to work

rnn_nerf: clean
	python3 -O rnn_runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 256 --epochs 7_500 --save models/rnn_lego.pt \
	--near 2 --far 6 --batch-size 4 --crop-size 12 -lr 1e-3 \
  --save-freq 2500 \
	--loss-fns l2 --valid-freq 499 --load models/rnn_lego.pt

monsune: clean
	python3 runner.py -d data/video/monsune_outta_my_mind.mp4 --data-kind single-video \
	--size 64 --epochs 30_000 --save models/monsune.pt --steps 32 \
  --dyn-model long --spline 4 --start-sec 46 --end-sec 48 --video-frames 100 \
	--near 0.01 --far 3 --batch-size 2 --crop-size 20 --model plain -lr 3e-4 --segments 8 \
	--loss-fns l2 fft --valid-freq 500 --refl-kind pos --save-freq 2500 --depth-images \
  --loss-window 1000 --train-imgs 20 --notest --train-parts camera all --sigmoid-kind fat \
  --load models/monsune.pt

fencing: clean
	python3 runner.py -d data/video/fencing.mp4 --data-kind single-video \
	--size 100 --epochs 0 --save models/fencing_video.pt --steps 32 \
  --dyn-model long --spline 4 --start-sec 47 --end-sec 49 --video-frames 100 \
	--near 0.01 --far 2 --batch-size 2 --crop-size 20 --model plain -lr 8e-5 --segments 10 \
  --clip-gradients 1 \
	--loss-fns l2 fft --valid-freq 500 --refl-kind pos --save-freq 2500 --sigmoid-kind upshifted \
  --depth-images --loss-window 1000 --train-imgs 40 --notest --train-parts camera all \
  --load models/fencing_video.pt --cam-save-load models/fencing_cam.pt --render-over-time 0 \
  --no-sched --seed -1

dance_off: clean
	python3 -O runner.py -d data/video/shoichi_chris_small.mp4 --data-kind single-video --size 512 \
	--epochs 10_000 --save models/dance_off.pt --model plain --batch-size 2 \
	--crop-size 20 --near 2 --far 6 -lr 5e-4 --valid-freq 500 --spline 6 --seed -1 \
  --loss-window 2000 --loss-fns l2 fft --test-crop-size 48 --depth-images --save-freq 2500 \
  --flow-map --dyn-model long --rigidity-map --refl-kind pos-linear-view \
  --static-vid-cam-angle-deg 75 --render-over-time-end-sec 15 --render-over-time 0 \
  --higher-end-chance 2 --offset-decay 0 --ffjord-div-decay 0 \
  --sigmoid-kind fat --notraintest --opt-step 3 --long-vid-progressive-train 5 \
  --dyn-refl-latent 32 \
  --end-sec 15 --notraintest --notest --load models/dance_off.pt


spline: clean
	python3 runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 128 --epochs 0 --save models/lego_spline.pt \
	--near 2 --far 6 --batch-size 2 --crop-size 24 --model spline -lr 3e-4 \
	--loss-fns l2 --valid-freq 500 --refl-kind view --sigmoid-kind upshifted \
	--depth-images --test-crop-size 32 --notraintest \
  --load models/lego_spline.pt

uniform_adam: clean
	python3 -O runner.py -d data/nerf_synthetic/lego/ --data-kind original \
	--size 128 --epochs 80_000 --save models/lego_uni.pt --opt-kind uniform_adam \
	--near 2 --far 6 --batch-size 4 --crop-size 20 --model plain -lr 3e-4 \
	--loss-fns l2 --refl-kind view --load models/lego_uni.pt --save-freq 2500


generate_animation: clean
	python3 scripts/2d_recon.py
