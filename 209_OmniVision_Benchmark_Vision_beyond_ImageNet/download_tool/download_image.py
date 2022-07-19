import argparse
import concurrent.futures
import os
import time
import urllib.error
import urllib.request

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="bamboo.val.txt")
parser.add_argument("--output_folder", type=str, default="bamboo.val")
parser.add_argument("--max_retries", type=int, default=5)
parser.add_argument("--max_workers", type=int, default=100)
args = parser.parse_args()


def download_image(url, label, timeout):
    result = {
        "status": "",
        "url": url,
        "label": label,
    }
    cnt = 0
    while True:
        try:
            response = urllib.request.urlopen(url, timeout=timeout)
            image_path = os.path.join(args.output_folder, "images", url.split("/")[-1])
            with open(image_path, "wb") as f:
                block_sz = 8192
                while True:
                    buffer = response.read(block_sz)
                    if not buffer:
                        break
                    f.write(buffer)
            result["status"] = "SUCCESS"
        except Exception as e:
            if not isinstance(e, urllib.error.HTTPError):
                cnt += 1
                if cnt <= args.max_retries:
                    continue
            if isinstance(e, urllib.error.HTTPError):
                result["status"] = "EXPIRED"
                result["exception_message"] = str(e)
            else:
                result["status"] = "TIMEOUT"
        break
    return result


def main():
    start = time.time()

    if os.path.exists(args.output_folder) and os.listdir(args.output_folder):
        try:
            c = input(
                f"'{args.output_folder}' already exists and is not an empty directory, "
                f"existing files may be overwritten, continue? (y/n) "
            )
            if c.lower() not in ["y", "yes"]:
                exit(0)
        except KeyboardInterrupt:
            print()
            exit(0)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    image_folder_path = os.path.join(args.output_folder, "images")
    record_path = os.path.join(args.output_folder, "records.txt")
    expired_url_path = os.path.join(args.output_folder, "expired_urls.txt")
    timeout_url_path = os.path.join(args.output_folder, "timeout_urls.txt")
    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)
    record_file = open(record_path, "w", encoding="utf8")
    timeout_url_file = open(timeout_url_path, "w", encoding="utf8")
    expired_url_file = open(expired_url_path, "w", encoding="utf8")

    distinct_urls = set()
    data, duplicated_url_records = [], []
    with open(args.input_path, "r", encoding="utf8") as f:
        for line in f:
            url, label = line.strip().split()
            if url in distinct_urls:
                # This URL has appeared before and does not need to be downloaded again.
                duplicated_url_records.append((url, label))
                continue
            distinct_urls.add(url)
            # data: [(url_1, label_1), (url_2, label_2), ...]
            data.append((url, label))
    print(f"number of records: {len(data) + len(duplicated_url_records)}")
    print(f"number of distinct urls: {len(distinct_urls)}")

    downloaded_urls, timeout_urls = set(), set()
    expired_url_exception_messages = {}
    with tqdm(total=len(data)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit up to `chunk_size` tasks at a time to avoid too many pending tasks.
            chunk_size = min(50000, args.max_workers * 500)
            for i in range(0, len(data), chunk_size):
                futures = [
                    executor.submit(download_image, url, label, 30)
                    for url, label in data[i : i + chunk_size]
                ]
                for future in concurrent.futures.as_completed(futures):
                    r = future.result()
                    status, url, label = r["status"], r["url"], r["label"]
                    if status == "SUCCESS":
                        image_path = os.path.join("images", url.split("/")[-1])
                        record_file.write(f"{image_path} {label}\n")
                        downloaded_urls.add(url)
                    elif status == "TIMEOUT":
                        timeout_url_file.write(f"{url} {label}\n")
                        timeout_urls.add(url)
                    elif status == "EXPIRED":
                        exception_message = r["exception_message"]
                        expired_url_file.write(f"{url} {label} | {exception_message}\n")
                        expired_url_exception_messages[url] = exception_message
                    else:
                        assert False
                    pbar.update(1)

    # Process the duplicated URLs.
    n_records, n_timeout, n_expired = (
        len(downloaded_urls),
        len(timeout_urls),
        len(expired_url_exception_messages),
    )
    for url, label in duplicated_url_records:
        if url in downloaded_urls:
            image_path = os.path.join("images", url.split("/")[-1])
            record_file.write(f"{image_path} {label}\n")
            n_records += 1
        elif url in timeout_urls:
            timeout_url_file.write(f"{url} {label}\n")
            n_timeout += 1
        elif url in expired_url_exception_messages:
            expired_url_file.write(f"{url} {label} | {expired_url_exception_messages[url]}\n")
            n_expired += 1
        else:
            assert False
    record_file.close()
    timeout_url_file.close()
    expired_url_file.close()

    end = time.time()
    print(f"consuming time {end - start:.1f} sec")
    print(f"{len(downloaded_urls)} images downloaded.")
    print(f"{n_records} records recorded in {record_path}.")
    print(f"{n_timeout} urls failed due to request timeout, see details in {timeout_url_path}.")
    print(f"{n_expired} urls failed due to url expiration, see details in {expired_url_path}.")

    # check
    with open(record_path, "r", encoding="utf8") as f:
        # [(image_path_1, label_1), ...]
        records = set(tuple(line.strip().split()) for line in f)
    with open(timeout_url_path, "r", encoding="utf8") as f:
        # [(url_1, label_1), ...]
        timeout_records = set(tuple(line.strip().split()) for line in f)
    with open(expired_url_path, "r", encoding="utf8") as f:
        # [(url_1, label_1), ...]
        expired_records = set(tuple(line.split(" | ")[0].split()) for line in f)
    with open(args.input_path, "r", encoding="utf8") as f:
        for line in f:
            url, label = line.strip().split()
            image_path = os.path.join("images", url.split("/")[-1])
            assert ((image_path, label) in records) + ((url, label) in timeout_records) + (
                (url, label) in expired_records
            ) == 1


if __name__ == "__main__":
    root  = '../meta_url_4_challenge'
    output_folder_root = '../meta'
    for file in os.listdir(root):
        args.input_path = os.path.join(root,file) 
        nid,status = file.split('.')
        args.output_folder =  os.path.join(output_folder_root,nid,status) 
        # import pdb;pdb.set_trace()
        main()
