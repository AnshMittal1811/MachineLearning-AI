import os


def dataloader(filepath, valpath):
    # init
    up_fold = 'image_up/'
    down_fold = 'image_down/'
    disp = 'disp_up/'

    image = [
        img for img in os.listdir(filepath + up_fold) if img.find('.png') > -1
    ]
    image_val = [
        img for img in os.listdir(valpath + up_fold) if img.find('.png') > -1
    ]

    train = image[:]
    val = image_val[:]

    up_train = [filepath + up_fold + img for img in train]
    down_train = [filepath + down_fold + img for img in train]
    disp_train = [filepath + disp + img[:-4] + '.npy' for img in train]

    up_val = [valpath + up_fold + img for img in val]
    down_val = [valpath + down_fold + img for img in val]
    disp_val = [valpath + disp + img[:-4] + '.npy' for img in val]

    return up_train, down_train, disp_train, up_val, down_val, disp_val
