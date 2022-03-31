# NeRF Atlas

A repository which contains NeRF and a bunch of extensions to NeRF.


Important Note:
WORK IN PROGRESS, things may be subtly _borken_ 🦮.

---

# What's a NeRF?

Ne(ural) R(adiance) F(ield)s represent a surface by approximating how much light is blocked at
each point in space, and the color of the surface that is blocking it. This approximation can be
used with volumetric rendering to view a surface from new angles.

The original paper implements this idea, and demonstrates its capacity for reconstructing a
surface or scene from a set of views. It has insanely high quality of reconstruction, but takes
forever to train, isn't modifiable at all, and takes a stupid amount of memory. In order to
fix those issues, there are a bunch of different repos which implement single changes to the
original that give it new capabilities, make it train faster, or give it wings, but
no projects mix multiple different modifications. This repo is supposed to be a single
unified interface to many different extensions.  I've implemented a number of other projects,
but there are more. Please feel free to contribute any you would like!

NeRF is similar to projects like [COLMAP](https://demuc.de/colmap/) in that it can perform
surface reconstruction from a series of images, but has not yet been shown to be able to scale
to incredibly large scales while jointly working without prior known camera angles.

The original project is [here](https://www.matthewtancik.com/nerf),
but math to understand why it works is [here](https://pbr-book.org/3ed-2018/Volume_Scattering).

## Usage

```sh
python3 runner.py -h
<All the flags>
```

See makefile for example usages. i.e.
```sh
make dnerf
make dnerfae
make original
```

One note for usage:
- I've found that using large crop-size with small number of batches may lead to better
  training.
- DNeRF is great at reproducing images at training time, but I had trouble getting good test
  results. I noticed that they have an [additional loss](https://github.com/albertpumarola/D-NeRF/issues/1) in their code,
  which isn't mentioned in their paper, but it's not clear to me whether it was used.

## Dependencies

PyTorch, NumPy, tqdm, matplotlib, imageio.

Install them how you want.

Notes on dependency versions:

This library makes use of `tensordot(..., dims=0)`, which is [broken in PyTorch
1.9](https://github.com/pytorch/pytorch/issues/61096). If you find this while using the library,
please downgrade to 1.8

---

## Extensions:

Currently, this repository contains a few extensions on "Plain" NeRF.

Model Level:

- TinyNeRF: One MLP for both density and output spectrum.
- PlainNeRF: Same architecture as original, probably different parameters.
- NeRFAE (NeRF Auto Encoder): Our extension, which encodes every point in space as a vector in a
  latent material space, and derives density and RGB from this latent space. In theory this
  should allow for similar points to be learned more effectively.
- [VolSDF](https://arxiv.org/pdf/2106.12052.pdf) Extends NeRF by volume rendering an SDF.
  This is probably one of the most practical ways to do volume rendering, i.e. seems like the
  most promising method because it merges both volumetric and surface rendering.
- [D-NeRF](https://arxiv.org/abs/2011.13961) for dynamic scenes, using an MLP to encode a
  positional change.
  - Convolutional Update Operator based off of [RAFT's](https://arxiv.org/pdf/2003.12039.pdf).
    Interpolation is definitely not what it's design was intended for, but is more memory
    efficient.
- \[WIP\][Pixel NeRF](https://arxiv.org/pdf/2012.02190.pdf) for single image NeRF
  reconstruction.

Encoding:

- Positional Encoding, as in the original paper.
- [Fourier Features](https://github.com/tancik/fourier-feature-networks).
- Learned Features based on [Siren](https://arxiv.org/abs/2006.09661): Pass low dimensional
  features through an MLP and then use `sin` activations. Not sure if it works.
- [MipNeRF](https://arxiv.org/abs/2103.13415) can be turned on with cylinder or conic volumes.

Training/Efficiency:

- DataParallel can be turned on and off.
- Train on cropped regions of the image for smaller GPUs.
- Neural Upsampling with latent spaces inspired by
  [GIRAFFE](https://arxiv.org/pdf/2011.12100.pdf). The results don't look great, but to be fair
  the paper also has some artifacts.

Note: NeRF is stupid slow. Writing out one of these extensions takes about half an hour,
training it takes about a day, and my GPU is much smaller than any used in the papers.

Voxel Implementation:

There is a relatively simple voxel implementation, both for static and dynamic scenes. This is
actively being worked on and extended so that it will be more efficient.

**Datasets Supported**:

- NeRF Synthetic (`--data-kind original`)
- Dynamic NeRF (`--data-kind dnerf`)
- NeRV (`--data-kind nerv_point`)
- DTU Scans (`--data-kind dtu`)
- NeRFActor (same as NeRF synthetic)
- **New** Collocated NeRF Synthetic (`--data-kind nerv_point`)
  - [Link to New Dataset](https://drive.google.com/drive/folders/1-0oZ8OGNR2WDw0R9gbNzt1LpeMaxF_vM)

---

### Example outputs

![Example Output Gif](examples/example.gif)

- Collecting datasets for this is difficult. If you have a dataset that you'd like contributed,
  add _a script to download it_ to the `data/` directory!

The outputs I've done are low-res because I'm working off a 3GB gpu and NeRF is memory intensive
during training. That's why I've explored a few different encodings and other tricks to speed it
up.

##### Dynamic NeRF Auto-Encoded

![dnerfae jumpingjacks](examples/dnerfae.gif)

A new change to NeRF using NeRF with an auto-encoder at every point in space. Since we're
mapping to a latent space at every point, it's possible to learn a transformation on that latent
space, for modifying density and visual appearance over time. One downside is that it is much
slower to train because of the higher number of dimensions, and may overfit due to the higher
number of dimensions.

The visualization is on the training set. On the test set it does not perform as well. I suspect
it lacks some regularization for temporal consistency, but I'll continue to look for ways to
make testing better.

##### Smoothly Interpolated Movement

![DNeRF Hook 1](examples/hook1.gif)
![DNeRF Hook 2](examples/hook2.gif)
![DNeRF Squat](examples/squat.gif)

Developed a model which allows for guaranteed smooth interpolation at infinite resolution as
compared to prior work, with no need for additional regularization.

##### DTU

![DTU shiny cans](examples/dtu.gif)

##### VolSDF

![VolSDF Lego](examples/volsdf.gif)

Implementation of [VolSDF](https://arxiv.org/abs/2106.12052), which produces better quality
output on the low sample counts necessary to run on my machine. It also noticeably has much less
flickering than standard NeRF, because within a region near the surface it is less likely
(guaranteed?) to not have holes unlike standard NeRF.

## Contributing

If you would like to contribute, feel free to submit a PR, but I may be somewhat strict,
apologies in advance.

Please maintain the same style:
- 2 spaces, no tabs
- Concise but expressive names
- Default arguments and type annotations when possible.
- Single line comments for functions, intended for developers.

#### Full options

The full set of options for training as of #416d073ed91573f36450ada1439d681227d8045e are below:

```
usage: runner.py [-h] -d DATA
optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  path to data (default: None)
  --data-kind {original,single_video,dnerf,dtu,pixel-single,nerv_point,shiny}
                        Kind of data to load (default: original)
  --derive-kind         Attempt to derive the kind if a single file is given (default: True)
  --outdir OUTDIR       path to output directory (default: outputs/)
  --timed-outdir        Create new output dir with date+time of run (default: False)
  --size SIZE           post-upsampling size (default: 32)
  --render-size RENDER_SIZE
                        pre-upsampling size (default: 16)
  --epochs EPOCHS       number of epochs to train for (default: 30000)
  --batch-size BATCH_SIZE
                        # views for each training batch (default: 8)
  --neural-upsample     add neural upsampling (default: False)
  --crop-size CROP_SIZE
                        what size to use while cropping (default: 16)
  --test-crop-size TEST_CROP_SIZE
                        what size to use while cropping at test time (default: 0)
  --steps STEPS         Number of depth steps (default: 64)
  --mip {cone,cylinder}
                        Use MipNeRF with different sampling (default: None)
  --sigmoid-kind {normal,thin,tanh,cyclic,upshifted,fat,softmax,leaky_relu,relu,sin,upshifted_softplus,upshifted_relu}
                        What activation to use with the reflectance model. (default: upshifted)
  --feature-space FEATURE_SPACE
                        The feature space size when neural upsampling. (default: 32)
  --model {tiny,plain,ae,volsdf,coarse_fine,mpi,voxel,rig,hist,spline,sdf}
                        which model to use? (default: plain)
  --dyn-model {plain,ae,rig,long,voxel}
                        Which dynamic model to use? (default: None)
  --bg {black,white,mlp,random}
                        What background to use for NeRF. (default: black)
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        learning rate (default: 0.0005)
  --seed SEED           Random seed to use, -1 is no seed (default: 1337)
  --decay DECAY         Weight decay value (default: 0)
  --notest              Do not run test set (default: False)
  --data-parallel       Use data parallel for the model (default: False)
  --omit-bg             Omit black bg with some probability. Only used for faster training (default: False)
  --train-parts {all,refl,occ,path-tf,camera} [{all,refl,occ,path-tf,camera} ...]
                        Which parts of the model should be trained (default: ['all'])
  --loss-fns {l2,l1,rmse,fft,ssim} [{l2,l1,rmse,fft,ssim} ...]
                        Loss functions to use (default: ['l2'])
  --color-spaces {rgb,hsv,luminance,xyz} [{rgb,hsv,luminance,xyz} ...]
                        Color spaces to compare on (default: ['rgb'])
  --tone-map            Add tone mapping (1/(1+x)) before loss function (default: False)
  --gamma-correct-loss GAMMA_CORRECT_LOSS
                        Gamma correct by x in training (default: 1.0)
  --has-multi-light     For NeRV point if there is a multi point light dataset (default: False)
  --style-img STYLE_IMG
                        Image to use for style transfer (default: None)
  --no-sched            Do not use a scheduler (default: False)
  --sched-min SCHED_MIN
                        Minimum value for the scheduled learning rate. (default: 5e-05)
  --serial-idxs         Train on images in serial (default: False)
  --replace [{refl,occ,bg,sigmoid,light,time_delta,al_occ} [{refl,occ,bg,sigmoid,light,time_delta,al_occ} ...]]
                        Modules to replace on this run, if any. Take caution for overwriting existing parts. (default: [])
  --all-learned-occ-kind {pos,pos-elaz}
                        What parameters the Learned Ambient Occlusion should take (default: pos)
  --volsdf-direct-to-path
                        Convert an existing direct volsdf model to a path tracing model (default: False)
  --volsdf-alternate    Use alternating volume rendering/SDF training volsdf (default: False)
  --latent-size LATENT_SIZE
                        Latent-size to use in shape models. If not supported by the shape model, it will be ignored. (default: 32)
  --refl-order REFL_ORDER
                        Order for classical Spherical Harmonics & Fourier Basis BSDFs/Reflectance models (default: 2)
  --inc-fourier-freqs   Multiplicatively increase the fourier frequency standard deviation on each run (default: False)
  --rig-points RIG_POINTS
                        Number of rigs points to use in RigNeRF (default: 128)

reflectance:
  --refl-kind {pos,view,pos-gamma-correct-view,view-light,basic,diffuse,rusin,rusin-helmholtz,sph-har,fourier,weighted}
                        What kind of reflectance model to use (default: view)
  --weighted-subrefl-kinds {pos,view,pos-gamma-correct-view,view-light,basic,diffuse,rusin,rusin-helmholtz,sph-har,fourier} [{pos,view,pos-gamma-correct-view,view-light,basic,diffuse,rusin,rusin-helmholtz,sph-har,fourier} ...]
                        What subreflectances should be used with --refl-kind weighted. They will not take a spacial component, and only rely on view direction, normal, and light direction. (default: [])
  --normal-kind {None,elaz,raw}
                        How to include normals in reflectance model. Not all surface models support normals (default: None)
  --space-kind {identity,surface,none}
                        Space to encode texture: surface builds a map from 3D (identity) to 2D (default: identity)
  --alt-train {analytic,learned}
                        Whether to train the analytic or the learned model, set per run. (default: learned)
  --refl-bidirectional  Allow normals to be flipped for the reflectance (just Diffuse for now) (default: False)
  --view-variance-decay VIEW_VARIANCE_DECAY
                        Regularize reflectance across view directions (default: 0)

integrator:
  --integrator-kind {None,direct,path}
                        Integrator to use for surface rendering (default: None)
  --occ-kind {None,hard,learned,learned-const,all-learned,joint-all-const}
                        Occlusion method for shadows to use in integration. (default: None)
  --smooth-occ SMOOTH_OCC
                        Weight to smooth occlusion by. (default: 0)
  --decay-all-learned-occ DECAY_ALL_LEARNED_OCC
                        Weight to decay all learned occ by, attempting to minimize it (default: 0)
  --all-learned-to-joint
                        Convert a fully learned occlusion model into one with an additional raycasting check (default: False)

light:
  --light-kind {field,point,dataset,None}
                        Kind of light to use while rendering. Dataset indicates light is in dataset (default: None)
  --light-intensity LIGHT_INTENSITY
                        Intensity of light to use with loaded dataset (default: 100)
  --point-light-position POINT_LIGHT_POSITION [POINT_LIGHT_POSITION ...]
                        Position of point light (default: [0, 0, -3])

sdf:
  --sdf-eikonal SDF_EIKONAL
                        Weight of SDF eikonal loss (default: 0)
  --surface-eikonal SURFACE_EIKONAL
                        Weight of SDF eikonal loss on surface (default: 0)
  --smooth-normals SMOOTH_NORMALS
                        Amount to attempt to smooth normals (default: 0)
  --smooth-surface SMOOTH_SURFACE
                        Amount to attempt to smooth surface normals (default: 0)
  --smooth-eps SMOOTH_EPS
                        size of random uniform perturbation for smooth normals regularization (default: 0.001)
  --smooth-eps-rng      Smooth by random amount instead of smoothing by a fixed distance (default: False)
  --smooth-n-ord {1,2} [{1,2} ...]
                        Order of vector to use when smoothing normals (default: [2])
  --sdf-kind {mlp,siren,local,curl-mlp,spheres,triangles}
                        Which SDF model to use (default: mlp)
  --sphere-init         Initialize SDF to a sphere (default: False)
  --bound-sphere-rad BOUND_SPHERE_RAD
                        Intersect the learned SDF with a bounding sphere at the origin, < 0 is no sphere (default: -1)
  --sdf-isect-kind {sphere,secant,bisect}
                        Marching kind to use when computing SDF intersection. (default: bisect)
  --volsdf-scale-decay VOLSDF_SCALE_DECAY
                        Decay weight for volsdf scale (default: 0)

dnerf:
  --spline SPLINE       Use spline estimator w/ given number of poitns for dynamic nerf delta prediction (default: 0)
  --time-gamma          Apply a gamma based on time (default: False)
  --with-canon WITH_CANON
                        Preload a canonical NeRF (default: None)
  --fix-canon           Do not train canonical NeRF (default: False)
  --render-over-time RENDER_OVER_TIME
                        Fix camera to i, and render over a time frame. < 0 is no camera (default: -1)

camera parameters:
  --near NEAR           near plane for camera (default: 2)
  --far FAR             far plane for camera (default: 6)
  --cam-save-load CAM_SAVE_LOAD
                        Location to save/load camera to (default: None)

Video parameters:
  --start-sec START_SEC
                        Start load time of video (default: 0)
  --end-sec END_SEC     Start load time of video (default: None)
  --video-frames VIDEO_FRAMES
                        Use N frames of video. (default: 200)
  --segments SEGMENTS   Decompose the input sequence into some # of frames (default: 10)
  --dyn-diverge-decay DYN_DIVERGE_DECAY
                        Decay divergence of movement field (default: 0)
  --ffjord-div-decay FFJORD_DIV_DECAY
                        FFJORD divergence of movement field (default: 0)
  --delta-x-decay DELTA_X_DECAY
                        How much decay for change in position for dyn (default: 0)
  --spline-len-decay SPLINE_LEN_DECAY
                        Weight for length of spline regularization (default: 0)
  --voxel-random-spline-len-decay VOXEL_RANDOM_SPLINE_LEN_DECAY
                        Decay for length, randomly sampling a chunk of the grid instead of visible portions (default: 0)
  --random-spline-len-decay RANDOM_SPLINE_LEN_DECAY
                        Decay for length, randomly sampling a bezier spline (default: 0)
  --voxel-tv-sigma VOXEL_TV_SIGMA
                        Weight of total variation regularization for densitiy (default: 0)
  --voxel-tv-rgb VOXEL_TV_RGB
                        Weight of total variation regularization for rgb (default: 0)
  --voxel-tv-bezier VOXEL_TV_BEZIER
                        Weight of total variation regularization for bezier control points (default: 0)
  --voxel-tv-rigidity VOXEL_TV_RIGIDITY
                        Weight of total variation regularization for rigidity (default: 0)
  --offset-decay OFFSET_DECAY
                        Weight of total variation regularization for rigidity (default: 0)

reporting parameters:
  --name NAME           Display name for convenience in log file (default: )
  -q, --quiet           Silence tqdm (default: False)
  --save SAVE           Where to save the model (default: models/model.pt)
  --save-load-opt       Save opt as well as model (default: False)
  --log LOG             Where to save log of arguments (default: log.json)
  --save-freq SAVE_FREQ
                        # of epochs between saves (default: 5000)
  --valid-freq VALID_FREQ
                        how often validation images are generated (default: 500)
  --display-smoothness  Display smoothness regularization (default: False)
  --nosave              do not save (default: False)
  --load LOAD           model to load from (default: None)
  --loss-window LOSS_WINDOW
                        # epochs to smooth loss over (default: 250)
  --notraintest         Do not test on training set (default: False)
  --duration-sec DURATION_SEC
                        Max number of seconds to run this for, s <= 0 implies None (default: 0)
  --param-file PARAM_FILE
                        Path to JSON file to use for hyper-parameters (default: None)
  --skip-loss SKIP_LOSS
                        Number of epochs to skip reporting loss for (default: 0)
  --msssim-loss         Report ms-ssim loss during testing (default: False)
  --depth-images        Whether to render depth images (default: False)
  --normals-from-depth  Render extra normal images from depth (default: False)
  --depth-query-normal  Render extra normal images from depth (default: False)
  --plt-cmap-kind       <OMITTED, TOO LONG>
  --gamma-correct       Gamma correct final images (default: False)
  --render-frame RENDER_FRAME
                        Render 1 frame only, < 0 means none. (default: -1)
  --exp-bg              Use mask of labels while rendering. For vis only. (default: False)
  --flow-map            Render a flow map for a dynamic nerf scene (default: False)
  --rigidity-map        Render a flow map for a dynamic nerf scene (default: False)
  --display-regularization
                        Display regularization in addition to reconstruction loss (default: False)
  --y-scale {linear,log,symlog,logit}
                        Scale kind for y-axis (default: linear)

meta runner parameters:
  --torchjit            Use torch jit for model (default: False)
  --train-imgs TRAIN_IMGS
                        # training examples (default: -1)
  --draw-colormap       Draw a colormap for each view (default: False)
  --convert-analytic-to-alt
                        Combine a model with an analytic BRDF with a learned BRDF for alternating optimization (default: False)
  --clip-gradients CLIP_GRADIENTS
                        If > 0, clip gradients (default: 0)
  --versioned-save      Save with versions (default: False)
  --higher-end-chance HIGHER_END_CHANCE
                        Increase chance of training on either the start or the end (default: 0)

auto encoder parameters:
  --latent-l2-weight LATENT_L2_WEIGHT
                        L2 regularize latent codes (default: 0)
  --normalize-latent    L2 normalize latent space (default: False)
  --encoding-size ENCODING_SIZE
                        Intermediate encoding size for AE (default: 32)

optimization parameters:
  --opt-kind {adam,sgd,adamw,rmsprop,uniform_adam}
                        What optimizer to use for training (default: adam)

```

# Fun example

![Fencing](examples/fencing.gif)



