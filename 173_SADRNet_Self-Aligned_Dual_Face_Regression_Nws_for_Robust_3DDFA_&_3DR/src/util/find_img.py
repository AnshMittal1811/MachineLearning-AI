from src.run.predict import *


def compare_img(a, b):
    return np.abs(np.array(a) / 255.0 - np.array(b) / 255.0).mean()


def find_in_dataset(all_img_data, img: Image, thresh_hold=0.01):
    for i in range(len(all_img_data)):
        item = all_img_data[i]
        tmp = item.get_image()
        if compare_img(tmp, img) < thresh_hold:
            print(i, item.image_path)
            plt.imshow(tmp)
            plt.show()


if __name__ == "__main__":
    dataset = make_dataset(config.VAL_DIR, 'val')
    all_data = dataset.val_data

    for root, dirs, files in os.walk('data/example'):
        files.sort()
        for file in files:
            if file.split('.')[-1] == 'jpg':
                print(file)
                image = Image.open(f'{root}/{file}').resize((256,256))
                find_in_dataset(all_data, image)

    # 3.jpg
    # 1817
    # data / dataset / AFLW2000_crop / image03897 / image03897.jpg
    # 6.jpg
    # 1365
    # data / dataset / AFLW2000_crop / image02832 / image02832.jpg
    # 2.jpg
    # 696
    # data / dataset / AFLW2000_crop / image01201 / image01201.jpg
    # 8.jpg
    # 243
    # data / dataset / AFLW2000_crop / image00401 / image00401.jpg
    # 4.jpg
    # 14
    # data / dataset / AFLW2000_crop / image00032 / image00032.jpg
    # 1.jpg
    # 1955
    # data / dataset / AFLW2000_crop / image04276 / image04276.jpg
    # 7.jpg
    # 1052
    # data / dataset / AFLW2000_crop / image02069 / image02069.jpg
    # 5.jpg
    # 847
    # data / dataset / AFLW2000_crop / image01606 / image01606.jpg

    # 9.jpg
    # 743
    # data / dataset / AFLW2000_crop / image01310 / image01310.jpg
    # 10.jpg
    # 1802 data/dataset/AFLW2000_crop/image03871/image03871.jpg
    # 11.jpg
    # 1812 data/dataset/AFLW2000_crop/image03891/image03891.jpg
    # 12.jpg
    # 980 data/dataset/AFLW2000_crop/image01927/image01927.jpg
    # 13.jpg
    # 362 data/dataset/AFLW2000_crop/image00593/image00593.jpg
    # 14.jpg
    # 1467 data/dataset/AFLW2000_crop/image03101/image03101.jpg