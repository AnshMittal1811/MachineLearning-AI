download_image.py is used to download the images corresponding to the URLs in bamboo.tran.txt and bamboo.val.txt.

Usage:
python download_image.py [-h] [--input_path INPUT_PATH] [--output_folder OUTPUT_FOLDER] [--max_retries MAX_RETRIES] [--max_workers MAX_WORKERS]

Example:
python download_image.py --input_path bamboo.train.txt --output_folder bamboo.train --max_retries 5 --max_workers 100

Arguments:
--input_path: The path of the original file. Each line of the file is a data record, including an image URL and the corresponding label, separated by a space.
--output_folder: The output folder, which contains:
    (1) <output_folder>/images: The folder to store downloaded images.
    (2) <output_folder>/records.txt: The file which contains the data records of successful image download. Each line of the file is a data record, including the local image path and the corresponding label, separated by a space.
    (3) <output_folder>/timeout_urls.txt: The file which contains the data records of image download timeout. The file format is the same as the original file.
    (4) <output_folder>/expired_urls.txt: The file which contains the data records with an expired URL. Each line of the file is composed of an image URL, the corresponding label and the request exception message.
--max_retries: max number of retries after image download timeout (default: 5).
--max_workers: max number of threads (default 100), when the network bandwidth is insufficient, the argument value can be adjusted down to avoid download timeout.

Notice:
1. URL expiration exists, for example, in bamboo.val.txt:
    line 451: https://live.staticflickr.com/65535/50632638498_9b585877a0_c.jpg n01632458
2. The download speed is affected by hard disk writing and network bandwidth, and the argument `max_workers` can be adjusted according to the actual download situation.
