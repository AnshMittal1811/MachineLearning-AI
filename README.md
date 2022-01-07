# 100 days of Machine Learning

This is the 100 days of Machine Learning, Deep Learning, Artificial Intelligence, and Optimization mini-projects that I picked up at the start of January 2022. I have used Google Colab for this work as it required various libraries and datasets to be downloaded. The following are the problems that I tackled: 

* GradCAM Implementation on Dogs v/s Cats using VGG16 pretrained models

Cat (GradCAM)          |  Dog (GradCAM)
:-------------------------:|:-------------------------:
![](https://github.com/AnshMittal1811/MachineLearning-AI/blob/master/001_GradCAM_basics/gradcam_cat.jpg)  |  ![](https://github.com/AnshMittal1811/MachineLearning-AI/blob/master/001_GradCAM_basics/gradcam_dog.jpg)

* Multi-task Learning (focussed on Object Localization)

![](https://github.com/AnshMittal1811/MachineLearning-AI/blob/master/002_Multi_task_Learning/Image_predict.png)

* Implementing GradCAM on Computer Vision problems
  1. GradCAM for Semantic Segmentation
  2. GradCAM for ObjectDetection


Computer Vision domains         |  CAM methods used         | Detected Images         | CAM-based images
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
Semantic Segmentation  | GradCAM  | ![](https://github.com/AnshMittal1811/MachineLearning-AI/blob/master/003_GradCAM_for_CV/SemanticSegmentation.png)  | ![](https://github.com/AnshMittal1811/MachineLearning-AI/blob/master/003_GradCAM_for_CV/GradCAMonSS.png)
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
Object Detection      | EigenCAM  | ![](https://github.com/AnshMittal1811/MachineLearning-AI/blob/master/003_GradCAM_for_CV/ObjectDetection.png)  | ![](https://github.com/AnshMittal1811/MachineLearning-AI/blob/master/003_GradCAM_for_CV/EigenCAMonOD.png)
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
Object Detection      | AblationCAM  | ![](https://github.com/AnshMittal1811/MachineLearning-AI/blob/master/003_GradCAM_for_CV/ObjectDetection.png)  | ![](https://github.com/AnshMittal1811/MachineLearning-AI/blob/master/003_GradCAM_for_CV/AblationCAMonOD.png)
