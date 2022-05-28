# The meaning for each folder in carla_dataset
+ masks folder contains all masks file, including train mask and test mask
+ train folder contain all train file, which ended with .npy. Each file contains sampled image, and the corresponding sampled information(i.e., location and rotation information of vehicle and camera).
+ test folder as the same to train folder
+ train_new folder contain all sample images extracted from each file of train folder.
+ test_new folder as the same to train_new folder
+ train_label_new folder which contain the label (refer to example file in train_label_new folder) of each training file
+ test_label_new folder as the same to train_label_new folder
## Note that
1. After you download the dataset and place them into each folder, you should create a yaml in data folder, see data/carala.yaml for example. 
2. To run compute the adversarial loss, you need to get the label(annotation) for each image. In our implementation, for simplicity, we used the Faster RCNN to inference each rendered image(due to the character of rendered image, we can get relatively accuracy label) and get the corresponding label.