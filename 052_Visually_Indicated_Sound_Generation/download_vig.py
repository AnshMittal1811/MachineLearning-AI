import os, sys
import numpy as np
import youtube_dl


class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)

def my_hook(d):
    if d['status'] == 'finished':
    	print 'Done Downloading'


ydl_opts = {
	# we want 480p video, file size < 300MB
    'format': 'bestvideo[height<=480][filesize<300M]+bestaudio/best[height<=480]',
    # downloaded videos will be saved in specific folders, with video_id.ext format
    'outtmpl': '~/dataset/vig/%(id)s.%(ext)s',
    # time out threshold is 100s
    'socket_timeout': 100,
    'logger': MyLogger(),
    # 'progress_hooks': [my_hook],
}

root_url = 'http://www.youtube.com/watch?v='
dl_list = []
num_all_video = 16384
split_id = int(sys.argv[1])
split_all = 4
err_ids = []
completed_ids = []

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
	with open('vig_youtube_id.txt') as fin:
		for line_id, line in enumerate(fin.readlines()):
			if line_id%split_all == split_id:
				video_id = line.strip().split(',')[0]
				dl_list = ['%s%s'%(root_url, video_id)]
				try:
					ydl.download(dl_list)
					print '%d/%d: %s done downloading'%(line_id, num_all_video, video_id)
					completed_ids.append(video_id)
				except:
					print '%d/%d: %s fail downloading'%(line_id, num_all_video, video_id)
					err_ids.append(video_id)

err_ids = np.array(err_ids)
completed_ids = np.array(completed_ids)

np.save('vig_dl_err_ids_%d.npy'%(split_id), err_ids)
np.save('vig_dl_complete_ids_%d.npy'%(split_id), completed_ids)

