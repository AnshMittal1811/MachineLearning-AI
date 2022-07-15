## Deep 3D Semantic Scene Extrapolation

Scene extrapolation is a challenging variant of the scene completion problem, which aims to predict the missing part(s)
of a scene. While the 3D scene completion algorithms in the literature try to fill the occluded part of a scene such as a chair
behind a table, this study focuses on extrapolating the available half-scene information to a full one. Our approaches are based on convolutional neural networks (CNN) and generative adversarial networks (GAN). As input, the half of 3D voxelized scenes is taken, then the models complete the other half of scenes as output. The tried models include CNN, GAN, U-Net and hybrid of 3D and 2D inputs. Some of these models are listed below. Read the full paper [here](http://user.ceng.metu.edu.tr/~ys/pubs/extrap-tvcj18.pdf).

#### The hybrid CNN model architecture, which takes a 3D scene and its top view projection as 2D.
![alt text](https://github.com/AliAbbasi/D3DSSE/blob/master/utils/images/hybrid_architecture.PNG)

#### The GAN model builds from CNN baseline as the generator and two global, local discriminator.
![alt text](https://github.com/AliAbbasi/D3DSSE/blob/master/utils/images/gan_architecture.png)

#### Some results from models.
![alt text](https://github.com/AliAbbasi/D3DSSE/blob/master/utils/images/results_figure.png)

Read the full details of models, approaches, results, discussion, limitations in my [MSc thesis](https://drive.google.com/file/d/1D0rpLWZJhU6RCKdt8PkTyVb0mnPBUcBO/view?usp=sharing).
