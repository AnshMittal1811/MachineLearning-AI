# Face Recognition Project
## Overview
This project implements a face detection and recognition in Python (based on Eigenfaces, SVD, and PCA). 

**Notice**: the part of the code was taken from the source [[4]](#initial-code-source) and extended.

**Technologies and devices used:**
* Python 2.7
* Python libraries:
	* OpenCV v.2.4.12
	* NumPy
	* Tkinter
	* Os (to scan files)
* Ordinary Web Camera
* Tested on the device "Banana Pro"

## Testing the script
### Recognize the image from the web camera
1) Run **"main.py"** in the terminal:
    ```
    python main.py
    ```
2) Press on the **"Space"** button on your keyboard. Here, the program will try to recognize the face from the web camera among the existing faces in the DB. During the first recognition, the program will compute eigenfaces, SVD, and PCA. Once it's computed, it'll work much faster.
 
 <p align="center">
 <img width="75%" alt="Recognized face from the DB" src="https://raw.githubusercontent.com/kagan94/Face-recognition-via-SVD-and-PCA/master/report_imgs/recognized_img.jpg"/>
 </p>
 
### Add new image to the face DB
* either add this image to the folder **"target_imgs"**
* or while running "main.py" in the terminal (command **"python image_preprocessing.py"**) or run this script from your IDE press on **"Space" to capture new image from a web camera**.

<p align="center">
<img width="50%" alt="Capture and save image from web camera" 
src="https://raw.githubusercontent.com/kagan94/Face-recognition-via-SVD-and-PCA/master/report_imgs/captured_img_from_web_cam.jpg"/>
</p>

After a new image is added, you will need to stop the script "main.py" and run pre-processing step again (to grayscale the image, detect a face, and resize it) by typing command in the terminal **"python image_preprocessing.py"**:
```
python image_preprocessing.py
```

## Project Description in details
* **Face Detection**

	The OpenCV library includes Cascade Classification for object recognition, which can be used for real-time face detection. 
	You can find [All available cascades](https://github.com/opencv/opencv/tree/master/data) in the **```data```** folder in the compiled OpenCV files. 

	Briefly, The Haar feature-based cascade classifiers created by Paul Viola and Michael Jones to detect different types of objects that rely on edge and specific features in the images (e.g. each face contains at least by 3 features: nose, mouth, eyes) [[1]](#face-detection-using-haar-cascades). Also, the Haar classifier is based on assumption that regions with eyes and mouth are darker than cheeks and forehead [[2]](#opencv-and-computer-vision-book).

	We used already pre-trained Haar cascade classifiers for face detection **"haarcascade_frontalface.xml"**.
* **Image Preprocessing**
	* Grayscale image
	* Detect and extract face
	* Resize extracted face to the target *Width x Height*
	* Save preprocessed image
* **Face Recognition**
	* **Transform the image into a vector**. For example, we have an image 100x100 pixels. As a result, we will have a vector of size 1x10,000 (flatten image).
	Read all pre-processed train images and flatten them (they must have the same size)
	* Compute the mean face. The example of the computed mean face is below (~5 images in the face DB):
	
	<p align="center">
  <img alt="Computed mean face" src="https://raw.githubusercontent.com/kagan94/Face-recognition-via-SVD-and-PCA/master/report_imgs/mean_face.jpg"/>
  </p>
  
	* **Subtract the mean face** from each image before performing SVD and PCA
	* **Compute the SVD** for the matrix from the previous step.
	  
	  As a result, we got: <br>
	  <p align="center"><b><i> U, S, Vt = SVD(A) </i></b></p>
	  Where the matrix <b>U</b> represents the eigenfaces. To reconstruct our initial matrix <b>A</b> we multiply <b>U, S, Vt</b>:
	  <p align="center"><b><i> A' = U x S x Vt </i></b></p>
	  On this step, we can reduce the dimensionality of the initial matrix <b>A</b>
	  by sorting only the most important features and slicing matrices <b>U, S, Vt</b> accordingly, then multiplication of all these 3 matrices will give an approximation of the initial matrix of images but with reduced dimensionality.
	* **Project onto PCA space**
	In order to select the most similar face to the input face, we will need to project the received features onto the PCA space where each feature will be in a new dimension, so the number of features = the number of dimensions.
	![Feature representation](https://raw.githubusercontent.com/kagan94/Face-recognition-via-SVD-and-PCA/master/report_imgs/PCA_feature_representation.jpg)
	Image source [[3]](#explanation-of-pca).

	  To do the PCA projections, we need to obtain weights of these features for each dimension by multiplying the reconstructed array of images with subtracted mean face by eigenfaces.
	* **Project test image onto PCA space** and find the most similar face in the face database.
	
	  We need to grayscale the target image, flatten it (convert to vector), and project onto PCA space by multiplying the obtained eigenfaces from step #3 by target image (flatten and grayscale).
	  
	  To find the identical face we will compute and compare the Euclidian distance between feature weights of test image and weights of all other images obtained in step #3. The image with the smallest score (sum of distances in all feature dimentions) will be classified as the result.

## Limitations of this approach
* Currently, the algorithm has an assumption that there is only 1 face photo in database instead of having several photos of 1 face,
  but made from different angles or/and in different conditions.
* PCA relies on linear assumptions.
* We do not take into account different variations of the face position.
  
  The photo with face can be taken from different angles. That is why it is better either to use several photos of the same face
  which were made from different angles or using 3D face model.
* Face detection does not take into account alignment of the face
  and recognizes only 1 face from the picture (even if there are several faces in one image).
* We do not know how the algorithm will behave with the same face,
  but with differences in age (e.g. photo that was made 10 years ago)

## Plans for future improvements
* Test the accuracy of the algorithm
* Extract only <b>K</b> important features to reduce dimensions during projecting onto PCA space
* Add histogram equalization to frame from web camera/image to improve the quality of the recognition
* Measure execution time of the script and detect the most time-consuming parts of the code (by using timeit or similar tools)
* Add threshold to detect non-existing image in the face DB

## Alternative and similar methods
* [Fisherfaces](http://www.bytefish.de/blog/fisherfaces/) (Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisherfaces: Recognition using class)
* Face recognition based on LDA (Linear discrimination analysis).
The key idea of LDA is to and an optimal projection that maximizes the distance between the inter-classes
(different features) and minimize the inter-class data (the most similar points that represent particular feature).

  Original article about LDA for face recognition - 
  <b><i>[Subspace Linear Discriminant Analysis for Face Recognition. W. Zhao, R. Chellappa P.J. Phillips, 1999.
](http://ieeexplore.ieee.org/abstract/document/670971/)</i></b>

* [Face Recognition Using Laplacianfaces. Xiaofei He, Shuicheng Yan, Yuxiao Hu, Partha Niyogi, and Hong-Jiang Zhang, IEEE](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1388260)

* Local Binary Patterns Histograms (Ahonen, T., Hadid, A., and Pietikainen, M. "Face Recognition with Local Binary Patterns.". Computer Vision - ECCV 2004 (2004), 469–481.)
[One of the articles about this approach](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7936534).

* Higher Order Singular Value Decomposition (HOSVD or multilinear SVD) can be applied to recognize person face with several face images. 
 [HOSVD - Theory and Applicatins, Göran Bergqvist and Erik G. Larsson](https://liu.diva-portal.org/smash/get/diva2:316227/FULLTEXT01.pdf).
 [Multilinear Analysis of Image Ensembles: <i><b>TensorFaces</b></i>](http://web.cs.ucla.edu/~dt/papers/eccv02/eccv02-preprint.pdf)


## References
1. [Face Detection using Haar Cascades](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)
2. Mastering OpenCV with Practical Computer Vision Projects (Chapter 8) by D.Baggio, S.Emami, D.Escrivá, K.Ievgen, N.Mahmood, J.Saragih, R.Shilkrot  <a name="opencv-and-computer-vision-book"></a>
3. [Geometric explanation of PCA](https://learnche.org/pid/latent-variable-modelling/principal-component-analysis/geometric-explanation-of-pca)  <a name="explanation-of-pca"></a>
4. [Eigenfaces and Forms](https://wellecks.wordpress.com/tag/eigenfaces/)  <a name="initial-code-source"></a>
5. Original paper about eigenfaces:<br>["Turk, M., and Pentland, A. "Eigenfaces for recognition.". Journal of Cognitive Neuroscience"](http://www.face-rec.org/algorithms/PCA/jcn.pdf)
6. [Face recognition datasets](http://www.face-rec.org/databases/)
7. [Principal component analysis in Python](http://baxincc.cc/questions/11278/principal-component-analysis-in-python)
8. [PCA in Python (Jupiter notebook)](http://www.shogun-toolbox.org/static/notebook/current/pca_notebook.html)
9. [PCA via SVD (Matlab example)](http://mghassem.mit.edu/pcasvd/)
10. [Matlab to Numpy syntax](https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html)
11. [Yale public face dataset](http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip)
12. [Code for the book "Mastering OpenCV with Practical Computer Vision Projects" by Packt Publishing 2012.](https://github.com/MasteringOpenCV/code)
13. Data-Driven Modeling & Scientific Computation. Chapter 15: Linear Algebra and Singular Value Decomposition (SVD)
14. [Description of eigenfaces algorithm in OpenCV doc.](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#eigenfaces)
