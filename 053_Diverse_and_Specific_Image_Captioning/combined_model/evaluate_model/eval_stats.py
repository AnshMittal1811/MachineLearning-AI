# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper 'Generating Diverse
# and Meaningful Captions: Unsupervised Specificity Optimization for Image
# Captioning (Lindh et al., 2018)'
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Diverse_and_Specific_Image_Captioning
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Run the scripts from the coco-eval dir to have access to the COCO imports
# EXCEPT when loading the TRAINING captions which should be run from
# coco/PythonAPI under the coco tools

from os import path as os_path
import sqlite3
import ijson

# Store the training captions in an sqlite table
_DB_PATH = "annotations.db"
_COCO5K_CAPTIONS = "data/dataset_coco.json"
_COCO5K_TABLE = "coco5k_table"
_ID_TABLE = "id_table"


# Store the ground-truth labels for the Karpathy 5k splits of MS COCO
def store_coco5k_captions():
    caption_data = []
    with open(_COCO5K_CAPTIONS) as capfile:
        for image_data in ijson.items(capfile, 'images.item'):
            split = image_data['split']
            filepath = os_path.join(image_data['filepath'], image_data['filename'])
            image_id = image_data['imgid']
            for sent in image_data['sentences']:
                caption_data.append( (' '.join(sent['tokens']), split, image_id, filepath) )

    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute('DROP TABLE IF EXISTS ' + _COCO5K_TABLE)
        conn.execute('CREATE TABLE ' + _COCO5K_TABLE + ' (caption TEXT, split TEXT, image_id INTEGER, filepath TEXT)')
        conn.executemany('INSERT INTO ' + _COCO5K_TABLE + ' VALUES (?,?,?,?)', caption_data)
        conn.commit()


# Store a model's output from a results file
def store_generated_captions(caption_file, table_name, generated_data_name, replace_data=False):
    caption_data = []
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS ' + table_name + ' (caption TEXT, split TEXT, image_id INTEGER)')
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*) FROM ' + table_name + ' WHERE split=?', (generated_data_name,))
        result = cur.fetchone()
        if result is not None and result[0] > 0:
            if replace_data:
                print("WARNING: There is already a table with name: " + table_name + " and a data batch (split) called: " + generated_data_name + ". Replacing with new data.")
                conn.execute('DELETE FROM ' + table_name + ' WHERE split=?', (generated_data_name,))
            else:
                print("ERROR: There is already a table with name: " + table_name + " and a data batch (split) called: " + generated_data_name)
                return
        with open(caption_file) as capfile:
            try:
                for imgData in ijson.items(capfile, 'item'):
                    caption_data.append( (imgData['caption'], generated_data_name, imgData['image_id']) )
            except ijson.backends.python.UnexpectedSymbol:
                print("ERROR UnexpectedSymbol exception with num entries in caption_data:", len(caption_data))

        conn.executemany('INSERT INTO ' + table_name + ' VALUES (?,?,?)', caption_data)
        conn.commit()


# Export stored captions to a text file, with one caption on each line
def export_captions(filepath, splits):
    with sqlite3.connect(_DB_PATH) as conn:
        with open(filepath, 'wt') as outfile:
            for split in splits:
                cur = conn.cursor()
                cur.execute('SELECT caption FROM ' + _COCO5K_TABLE + ' WHERE split=?', (split,))
                for caption in cur:
                    outfile.write(caption[0] + '\n')
                cur.close()


# Create a conversion table between "img id" and "coco id"
def store_imgid_to_cocoid():
    img_to_coco = []

    with open(_COCO5K_CAPTIONS) as capfile:
        for image_data in ijson.items(capfile, 'images.item'):
            split = image_data['split']
            coco_id = image_data['imgid']
            filename = image_data['filename']
            split_underscore = filename.split('_')
            img_id = -999
            if len(split_underscore) == 3:
                split_dot = split_underscore[2].split('.')
                if len(split_dot) == 2:
                    try:
                        img_id = int(split_dot[0])
                    except ValueError:
                        img_id = -999

            if img_id == -999:
                print(split, filename, coco_id)
            else:
                img_to_coco.append((coco_id, img_id, split,))

    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute('DROP TABLE IF EXISTS ' + _ID_TABLE)
        conn.execute('CREATE TABLE ' + _ID_TABLE + ' (coco_id INTEGER PRIMARY KEY, img_id INTEGER, split TEXT)')
        conn.executemany('INSERT INTO ' + _ID_TABLE + ' VALUES (?,?,?)', img_to_coco)
        conn.commit()


# Some light data exploration to get familiar with the dataset
def explore():
    with sqlite3.connect(_DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM ' + _COCO5K_TABLE + ' LIMIT 1')
        result = cur.fetchone()
        if result is not None:
            print("SELECT *", result)
        cur.close()

        cur = conn.cursor()
        cur.execute('SELECT DISTINCT(split) FROM ' + _COCO5K_TABLE)
        for split in cur:
            print("SPLIT", split)
        cur.close()

        for split in ['train', 'restval', 'val', 'test']:
            cur = conn.cursor()
            cur.execute('SELECT COUNT(*) FROM ' + _COCO5K_TABLE + ' WHERE split=?', (split,))
            result = cur.fetchone()
            if result is not None:
                print("TOTAL count in", split, result)
            cur.close()
            cur = conn.cursor()
            cur.execute('SELECT COUNT(*) FROM ' + _COCO5K_TABLE + ' WHERE split=? AND filepath=""', (split,))
            result = cur.fetchone()
            if result is not None:
                print("EMPTY PATH count in", split, result)
            cur.close()

        cur = conn.cursor()
        cur.execute('SELECT COUNT(DISTINCT image_id), COUNT(*) FROM ' + _COCO5K_TABLE + ' WHERE split IN("train", "restval", "val", "test")')
        result = cur.fetchone()
        if result is not None:
            print("distinct image_ids and total number of coco5k rows", result)
        cur.close()


# Diversity: Calculate Distinct caption stats
def calculate_distinct(table_name, splits, verbose=True):
    with sqlite3.connect(_DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute('SELECT COUNT(DISTINCT caption) FROM ' + table_name + ' WHERE split IN ("' + '","'.join(splits) + '")')
        num_distinct = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM ' + table_name + ' WHERE split IN ("' + '","'.join(splits) + '")')
        num_total = cur.fetchone()[0]
        cur.close()

    fraction_distinct = float(num_distinct) / float(num_total)
    if verbose:
        print("Total generated captions =", num_total)
        print("Number of distinct =", num_distinct)
        print("Fraction distinct =", fraction_distinct)

    return fraction_distinct


# Novel Sentences: percentage of generated captions not seen in the training set.
def calculate_novelty(table_name, split, verbose=True):
    with sqlite3.connect(_DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*), caption FROM ' + table_name + ' WHERE split=? AND caption NOT IN (SELECT caption FROM ' + _COCO5K_TABLE + ' WHERE split IN("train", "restval"))', (split,))
        num_novel = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM ' + table_name + ' WHERE split=?', (split,))
        num_total = cur.fetchone()[0]
        cur.close()

    fraction_novel = float(num_novel)/float(num_total)

    if verbose:
        print("Total generated captions =", num_total)
        print("Number of novel =", num_novel)
        print("Fraction novel =", fraction_novel)
        print("Fraction seen in training data =", 1-fraction_novel)

    return fraction_novel


# Vocabulary Size: number of unique words used in all generated captions
# Returns number of unique words used in the captions of this table + split
def calculate_vocabulary_usage(table_name, split, verbose=True):
    # Build a set of unique words used in the captions of this table+split
    vocab = set()

    with sqlite3.connect(_DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute('SELECT caption FROM ' + table_name + ' WHERE split=?', (split,))
        for caption in cur:
            vocab.update(caption[0].split(' '))
        cur.close()

    if verbose:
        print("Total vocabulary used =", len(vocab))
        if 'UNK' in vocab:
            print('UNK is part of vocab.')
        else:
            print('UNK is NOT part of vocab.')

        print("Vocab:", vocab)

    return len(vocab)


# Print the generated captions onto the images, both from the baseline generator and the model we're comparing
def export_captions_on_images(model_table, model_split, baseline_table, baseline_split, coco_split, outdir):
    from PIL import Image
    from PIL import ImageDraw

    _IMG_PATH = 'data/coco_images/'

    image_info = {}

    with sqlite3.connect(_DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute('SELECT img_id, filepath, coco_id, image_id, ' + _COCO5K_TABLE + '.split, ' + _ID_TABLE + '.split FROM ' + _COCO5K_TABLE + ' INNER JOIN ' + _ID_TABLE + ' ON image_id=coco_id WHERE ' + _COCO5K_TABLE + '.split=? ORDER BY img_id', (coco_split,))
        for result in cur:
            image_info[result[0]] = {}
            image_info[result[0]]['image_id'] = result[0]
            image_info[result[0]]['filepath'] = result[1]
        cur.close()

        cur = conn.cursor()
        if baseline_split is None:
            cur.execute('SELECT image_id, caption FROM ' + baseline_table + ' ORDER BY image_id')
        else:
            cur.execute('SELECT image_id, caption FROM ' + baseline_table + ' WHERE split=? ORDER BY image_id', (baseline_split,))
        for result in cur:
            image_info[result[0]]['baseline_caption'] = result[1]
        cur.close()

        cur = conn.cursor()
        if model_split is None:
            cur.execute('SELECT image_id, caption FROM ' + model_table + ' ORDER BY image_id')
        else:
            cur.execute('SELECT image_id, caption FROM ' + model_table + ' WHERE split=? ORDER BY image_id', (model_split,))
        for result in cur:
            image_info[result[0]]['model_caption'] = result[1]
        cur.close()

    for info in image_info.values():
        orig_path = os_path.join(_IMG_PATH, info['filepath'])
        if os_path.exists(orig_path):
            image = Image.open(orig_path)
            draw_handle = ImageDraw.Draw(image)
            draw_handle.text((12, 6), 'PATH: ' + info['filepath'] + '\nBASELINE: ' + info['baseline_caption'] + '\nMODEL:' + info['model_caption'])
            image.save(os_path.join(outdir, str(info['image_id']) + '.jpg'))
        else:
            print("WARNING: Invalid path ", info['filepath'])


if __name__ == '__main__':
    store_coco5k_captions()
    explore()
    # store_imgid_to_cocoid()
    # calculate_distinct(_COCO5K_TABLE, ['train', 'restval'])
    # calculate_distinct(_COCO5K_TABLE, ['val'])
    # calculate_distinct(_COCO5K_TABLE, ['test'])
    # calculate_novelty(_COCO5K_TABLE, 'test')
