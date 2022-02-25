# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper 'Generating Diverse
# and Meaningful Captions: Unsupervised Specificity Optimization for Image
# Captioning (Lindh et al., 2018)'
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Diverse_and_Specific_Image_Captioning
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# These are helper functions to find the MSCOCO images that have the wrong type.

from os import listdir, path as os_path
import imghdr


# Returns a dict with image types as keys and the number of that type as the value
def check_types(filedir):
    type_counts = {}

    filenames = listdir(filedir)

    for filename in filenames:
        img_type = imghdr.what(os_path.join(filedir, filename))
        try:
            prev_count = type_counts[img_type]
        except KeyError:
            prev_count = 0

        type_counts[img_type] = prev_count + 1

    return type_counts


def find_png(filedir):
    filenames = listdir(filedir)

    for filename in filenames:
        img_type = imghdr.what(os_path.join(filedir, filename))
        if img_type == 'png':
            print("PNG found!", filename)
            break


if __name__ == '__main__':
    type_dict = check_types("coco_images/val2014")
    print(type_dict)
    find_png("coco_images/val2014")
