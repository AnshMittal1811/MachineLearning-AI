import os

class Reader:
    def __init__(self):
        pass

    def read(dataset_folder_path):
        labels = None
        text_arr = None
        tags_arr = None

        with open(os.path.join(dataset_folder_path, 'label'), encoding='utf-8') as f:
            labels = f.readlines()

        with open(os.path.join(dataset_folder_path, 'seq.in'), encoding='utf-8') as f:
            text_arr = f.readlines()

        with open(os.path.join(dataset_folder_path, 'seq.out'), encoding='utf-8') as f:
            tags_arr = f.readlines()

        assert len(text_arr) == len(tags_arr) == len(labels) # test by using 'assert'
        return text_arr, tags_arr, labels

if __name__ == '__main__':
    text_arr, tags_arr, labels = Reader.read('dataset/atis/test')
    print(text_arr, tags_arr, labels)