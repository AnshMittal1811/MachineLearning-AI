# Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis
[![report](https://img.shields.io/badge/arxiv-report-red)](http://arxiv.org/abs/2204.11798) 
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() -->

![Teaser image](./docs/teaser.png)

> **Abstract:** *This work targets at using a general deep learning framework to synthesize free-viewpoint images of arbitrary human performers, only requiring a sparse number of camera views as inputs and skirting per-case fine-tuning. The large variation of geometry and appearance, caused by articulated body poses, shapes and clothing types, are the key bot tlenecks of this task. To overcome these challenges, we present a simple yet powerful framework, named Generalizable Neural Performer (GNR), that learns a generalizable and robust neural body representation over various geometry and appearance. Specifically, we compress the light fields for novel view human rendering as conditional implicit neural radiance fields with several designs from both geometry and appearance aspects. We first introduce an Implicit Geometric Body Embedding strategy to enhance the robustness based on both parametric 3D human body model prior and multi-view source images hints. On the top of this, we further propose a Screen-Space Occlusion-Aware Appearance Blending technique to preserve the high-quality appearance, through interpolating source view appearance to the radiance fields with a relax but approximate geometric guidance.* <br>

Wei Cheng, Su Xu, Jingtan Piao, Chen Qian, Wayne Wu, Kwan-Yee Lin, Hongsheng Li<br>
**[[Demo Video]](https://www.youtube.com/watch?v=2COR4u1ZIuk)** | **[[Project Page]](https://generalizable-neural-performer.github.io/)** | **[[Data]](https://generalizable-neural-performer.github.io/genebody.html)** | **[[Paper]](https://arxiv.org/pdf/2204.11798.pdf)**

## Updates

- [26/04/2022] Code is coming soon!
- [26/04/2022] Part of data released!
- [26/04/2022] Techincal report released.
- [24/04/2022] The codebase and project page are created.

## Upcoming Events
- [28/04/2022] SMPLx fitting toolbox and benchmark release.
- [01/05/2022] GeneBody Train40 release.
- [07/05/2022] Code and pretrain model release.
- [01/06/2022] Extended370 release.


## Data Usage
To download and use the GeneBody dataset set, please read the instructions in [Dataset.md](./docs/Dataset.md).

## Annotations
GeneBody provides the per-view per-frame segmentation, using [BackgroundMatting-V2](https://github.com/PeterL1n/BackgroundMattingV2), and register the fitted [SMPLx](https://github.com/PeterL1n/BackgroundMattingV2) using our enhanced multi-view smplify repo in [here](https://github.com/generalizable-neural-performer/bodyfitting). Annotation details can be found in the documentation [Annotation.md]()

## Benchmarks
We also provide benchmarks of start-of-the-art methods on GeneBody Dataset, methods and requirements are listed in [Benchmarks.md](./docs/Benchmarks.md).

## Citation

```
@article{
    author = {Wei, Cheng and Su, Xu and Jingtan, Piao and Wayne, Wu and Chen, Qian and Kwan-Yee, Lin and Hongsheng, Li},
    title = {Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis},
    publisher = {arXiv},
    year = {2022},
  }
```
