# import scipy
# from matplotlib import pyplot as plt

import os
import numpy as np
import cv2
import Tkinter as tk
from helpers import *

# Matlab to Numpy syntax: https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html
# Based on https://wellecks.wordpress.com/tag/eigenfaces/ and other sources
# Example of face data set: http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip


######################################
# GLOBALS
######################################

# Default values for projections, eigenfaces and mean face
train_proj, e_faces, mean_face_flatten = None, None, None

# Target folder with pre-processed imgs (to perform SVD and PCA on them)
f_list = [f_name for f_name in os.listdir(pre_processed_imgs_dir) if os.path.isfile(os.path.join(pre_processed_imgs_dir, f_name))]
n = sum([True for f in f_list])  # Count total no of face images

# test_img_path = os.path.join(imgs_dir, f_list[-1])
# test_img = cv2.imread(test_img_path, 0).flatten()


######################################
# Helpers
######################################
root = None
new_img_content = None
fullname_var = None
    
def onSaveNewImage(*args):
    img_name = fullname_var.get() + '.jpg'
    img_path = os.path.join(imgs_dir, img_name)
    cv2.imwrite(img_path, new_img_content)

    if root is not None:
        root.destroy()


def draw_rectangles(img, rectangles):
    for (x, y, w, h) in rectangles:
        pt1, pt2 = (x, y), (x + w, y + h)
        cv2.rectangle(img, pt1, pt2, color=(0, 255, 0))


def euclidean_dist(vector1, vector2):
    ''' Euclidean distance between vectors '''
    dist = np.sqrt(np.sum((vector1 - vector2) ** 2))  # or use this instead np.linalg.norm(x - y)
    return dist


def norm(array):
    # Use norm1 to normalize
    return array / np.linalg.norm(array)


def compute_svd_pca():
    global W, H
    im_width, im_height = (W, H)

    # Create a vector of images
    X = np.array([cv2.imread(os.path.join(pre_processed_imgs_dir, filename), 0).flatten() for filename in f_list])

    # compute the mean face
    mu = np.mean(X, 0)

    # Subtract the mean face from each image before performing SVD and PCA
    ma_data = X - mu

    print("Computing SVD of data matrix")
    # Decompose the mean-centered matrix into three parts

    U, S, Vt = np.linalg.svd(ma_data.transpose(), full_matrices=False)
    V = Vt.T

    # Sort the PCs by descending order of the singular values (i.e. by the proportion of total variance they explain)
    ind = np.argsort(S)[::-1]
    U, S, V = U[:, ind], S[ind], V[:, ind]
    e_faces = U

    # TODO: Add slicing of SVD
    # v = v[:,:numcomp]

    # Weights is an n x n matrix
    weights = np.dot(ma_data, e_faces)  # TODO: Maybe swap + .T to e_faces

    # Some intermediate save:
    save_mean_face = False
    if save_mean_face:
        # Save mean face
        mean_face = mu.reshape(im_width, im_height)
        cv2.imwrite(os.path.join(res_dir, 'mean_face.jpg'), mean_face)
        # plt.imshow(mean_face, cmap='gray'); plt.show()

    save_eigenvectors = False
    if save_eigenvectors:
        print("Writing eigenvectors to disk...")
        for i in xrange(n):
            f_name = os.path.join(res_dir, 'eigenvector_%s.png' % i)
            im = U[:, i].reshape(im_width, im_height)
            cv2.imwrite(f_name, im)

    save_reconstructed = False
    if save_reconstructed:
        k = 2
        print '\n', 'Save the reconstructed images based on only "%s" eigenfaces' % k
        for img_id in range(n):
            # for k in range(1, total + 1):
            recon_img = mu + np.dot(weights[img_id, :k], e_faces[:, :k].T)
            recon_img.shape = (im_width, im_height)  # transform vector to initial image size
            cv2.imwrite(os.path.join(res_dir, 'img_reconstr_%s_k=%s.png' % (f_list[img_id], k)), recon_img)

    ###########################################
    # We have already projected our training images into pca subspace as yn=weights or Yn = E.T * (Xn - mean_face).
    train_proj = weights
    mean_face_flatten = mu

    return train_proj, e_faces, mean_face_flatten


def recognize_face(face_gray):
    global train_proj, e_faces, mean_face_flatten, f_list

    face_gray_flatten = face_gray.flatten()  # convert face to vector
    # cv2.imshow('Grayscale face', face_gray); cv2.waitKey(0); exit()

    # If we didn't compute SVD + PCA earlier, then compute it
    if None in [train_proj, e_faces, mean_face_flatten]:
        train_proj, e_faces, mean_face_flatten = compute_svd_pca()

    # Subtract mean face from the target face
    print mean_face_flatten.shape
    test_f = face_gray_flatten - mean_face_flatten

    # Projecting our test image into PCA space
    test_proj = np.dot(e_faces.T, test_f)

    # TODO: Add threshold to detect non-existing image
    # Calculate the distance between one test image and all other training images
    d = np.zeros((n, 1))
    for i in range(n):
        d[i] = euclidean_dist(train_proj[i], test_proj)
    min_dist_id = d.argmin()

    # found_face = X[min_dist_id].reshape((W, H))  # reshape image to initial form

    # if we use all of the PCs we can reconstruct the noisy signal perfectly
    # S = np.diag(s)
    # Mhat = np.dot(U, np.dot(S, V.T))
    # print("Using all PCs, MSE = %.6G" %(np.mean((M - Mhat)**2))
    #
    # # if we use only the first 20 PCs the reconstruction is less accurate
    # Mhat2 = np.dot(U[:, :20], np.dot(S[:20, :20], V[:,:20].T))
    # print("Using first 20 PCs, MSE = %.6G" %(np.mean((M - Mhat2)**2))

    ###########################################
    # Reopen matched image and parse filename
    found_face_filename = f_list[min_dist_id]
    found_face_img = cv2.imread(os.path.join(pre_processed_imgs_dir, found_face_filename))

    print 'File name is "%s"' % found_face_filename
    cv2.imshow(found_face_filename, found_face_img)
    cv2.waitKey(0)


#################################################################
# Start the program
#################################################################
if __name__ == '__main__':
    # global root, new_img_content, fullname_var

    video_cam_recognition = True
    single_img_recognition = False

    if video_cam_recognition:
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            exit('Web camera is not connected')

        print 'Available commands:'
        print '\n', 'Press "Enter" to capture current frame as image from web camera and add it to the database'
        print '\n', 'Press "Space" to recognize face from current frame from web camera \n'

        try:
            while True:
                # Capture frame-by-frame
                ret, frame = video_capture.read()
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                detected_face_gray, detected_face_coords = detect_face(frame_gray)  # try to detect face
                
                frame_with_mask = None
                if detected_face_gray is not None:
                    # Mark found face
                    mask = np.zeros_like(frame)  # init mask
                    draw_rectangles(mask, [detected_face_coords])
                    frame_with_mask = cv2.add(frame, mask)

                # Show current frme
                frame_to_show = frame_with_mask if frame_with_mask is not None else frame
                cv2.imshow('Video', frame_to_show)  # show either frame (if face isn't detected or frame with mask)

                # Process picture when SPACE is pressed
                k = cv2.waitKey(1)
                if k % 256 == 32 and detected_face_gray is not None:
                    # Run recognition part
                    print '>> Start recognizing the images...'
                    recognize_face(detected_face_gray)
                    print '<< Finished the recognition.'

                    # cv2.imwrite('target_face.png', frame)  # save target img
                elif k & 0xFF in [ord('\r'), ord('\n')]:
                    print 'Enter pressed (save image)'
                    
                    if detected_face_gray is not None:
                        # Run save new image form
                        root = tk.Tk()
                        new_img_content = frame
                        fullname_var = tk.StringVar(root)

                        tk.Label(root, text='Fill First and Last Name').grid(row=0)
                        tk.Entry(root, textvariable=fullname_var).grid(row=1)
                        tk.Button(root, text='Save new image', command=onSaveNewImage).grid(row=2)

                        root.mainloop()

                    else:
                        print 'Face is not detected...'

                # Exit
                elif k & 0xFF == ord('q'):
                    break
                # print repr(chr(k%256))  # print pressed button

        except KeyboardInterrupt:
            print('Ctrl + C issued...')

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

    # Sigle face recognition provided by user
    elif single_img_recognition:
        faceCascade = cv2.CascadeClassifier(filename='haarcascade_frontalface.xml')
        img = cv2.imread(filename='my_face.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get list of detected faces as rectangles
        rects = faceCascade.detectMultiScale(img_gray)
        # draw_rectangles(img, rects)

        # Exit, if no faces detected
        if not len(rects):
            exit()

        largest_rect = find_larget_face(rects)
        x, y, w, h = largest_rect

        face = img[y:y + h, x:x + w]  # crop
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Full image', img)
        # cv2.imshow('face', face)
        # cv2.imshow('Grayscale face', face_gray)
        # cv2.waitKey(0)
