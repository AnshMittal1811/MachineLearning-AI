import os


def dataloader(filepath):

    up_fold = 'image_up/'
    down_fold = 'image_down/'
    image = [
        img for img in os.listdir(filepath + up_fold) if img.find('.png') > -1
    ]

    up_test = [filepath + up_fold + img for img in image]
    down_test = [filepath + down_fold + img for img in image]
    print("up size: ", len(up_test), len(up_test[0]))
    print(up_test)

    return up_test, down_test
