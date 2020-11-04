from module.shotdetect.findframe import *
from pymediainfo import MediaInfo
from moviepy.editor import VideoFileClip
import cv2
import glob
import math
import numpy as np
import os
import csv
import json

# execution options
NORM_OPTION = True # diff normalization
twoframepersecond = True # 2fps or 1fps

# for test
SAVE_DIFF = False
EXTRACT_RESULT = False
transform = ('original', 'logo', 'border', 'topborder', 'crop', 'format', 'resolution', 'framerate', 'camcording')


def save_diff(videoname, frame_diffs):
	norm_list = []
	# MINMAX normalization
	MIN = min(frame_diffs)
	MAX = max(frame_diffs)
	for idx, x in enumerate(frame_diffs):
		temp = (x - MIN) / (MAX - MIN)
		norm_list.append(temp)

	f = open(videoname+'.csv', 'w')
	print(videoname+'.csv')
	wr = csv.writer(f)
	wr.writerow(['diff']+norm_list)
	f.close()


def checkresult(videopath, videoname, global_thr, idx_list):
	dstpath = os.path.join(str(global_thr), videoname)
	if not os.path.isdir(dstpath):
		os.makedirs(dstpath)

	try:
		media_info = MediaInfo.parse(videopath)
		for track in media_info.tracks:
			if track.track_type == 'General':
				frameRate = math.floor(float(track.frame_rate))
	except:
		myvideo = VideoFileClip(videopath)
		frameRate = myvideo.fps
	finally:
		cap = cv2.VideoCapture(videopath)
		success, frame = cap.read()
		i = 0
		while (success):
			frameId = cap.get(1)
			if twoframepersecond:
				if ((frameId - 1) % math.floor(frameRate/2) == 0):
					if i in idx_list:
						cv2.imwrite(os.path.join(dstpath, str(i)+'.jpg'), frame)
					i += 1
			else:
				if ((frameId - 1) % math.floor(frameRate) == 0):
					if i in idx_list:
						cv2.imwrite(os.path.join(dstpath, str(i)+'.jpg'), frame)
					i += 1
			success, frame = cap.read()
		cap.release()


def MINMAX_norm(list):
	norm_list = []

	MIN = min(list)
	MAX = max(list)
	for idx, x in enumerate(list):
		temp = (x - MIN) / (MAX - MIN)
		norm_list.append(temp)

	return norm_list


def ExtractFrame(videopath):
	frame_list = []
	cap = cv2.VideoCapture(videopath)

	try:
		media_info = MediaInfo.parse(videopath)
		for track in media_info.tracks:
			if track.track_type == 'General':
				frameRate = math.floor(float(track.frame_rate))
	except:
		myvideo = VideoFileClip(videopath)
		frameRate = myvideo.fps
		# print('pymediainfo error')
		# print(frameRate)
	finally:
		success, frame = cap.read()
		i = 0
		while (success):
			frameId = cap.get(1)
			if twoframepersecond:
				if ((frameId - 1) % math.floor(frameRate/2) == 0):
					frame_list.append(frame)
					i += 1
			else:
				if ((frameId - 1) % math.floor(frameRate) == 0):
					frame_list.append(frame)
					i += 1
			success, frame = cap.read()
		cap.release()

		return frame_list


def SBD(videopath, OPTION='minmax'):
	frame_diffs = []
	frames = []

	frame_list = ExtractFrame(videopath)

	"""
	1. Calculate the difference between adjacent frames
	"""
	curr_frame = None
	prev_frame = None
	for idx, frame in enumerate(frame_list):
		luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
		curr_frame = luv
		if curr_frame is not None and prev_frame is not None:
			diff = cv2.absdiff(curr_frame, prev_frame)
			diff_sum = np.sum(diff)
			diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
			frame_diffs.append(diff_sum_mean)
		elif curr_frame is not None and prev_frame is None:
			diff_sum_mean = 0
			# frame_diffs.append(diff_sum_mean)
		prev_frame = curr_frame
	if NORM_OPTION:
		frame_diffs = MINMAX_norm(frame_diffs)

	FRAME = Frame(0, 0)
	for idx, frame in enumerate(frame_diffs):
		frame = Frame(idx+1, frame_diffs[idx])
		frames.append(frame)

	"""
	2. Find the boundary using the calculated difference
		1) detect the possible frame
		2) optimize the possible frame
	"""
	if OPTION == 'global':
		frame_return, start_id_spot, end_id_spot = FRAME.thresholding(frames, frame_diffs, judge=1.8)
		new_frame, start_id_spot, end_id_spot = FRAME.thresholding_optimFrame(frame_return, frames)
	elif OPTION == 'local':
		# frame_return, start_id_spot, end_id_spot = FRAME.find_possible_frame(frames)
		frame_return, start_id_spot, end_id_spot = FRAME.minmax_findFrame(frames, windowsize=5)
		frame_diffs_numpy = np.array(frame_diffs); average = np.average(frame_diffs_numpy)
		new_frame, start_id_spot, end_id_spot = FRAME.optimize_frame(frame_return, frames, average)
	elif OPTION == 'minmax':
		frame_return, start_id_spot, end_id_spot = FRAME.minmax_findFrame(frames, windowsize=10)
		new_frame, start_id_spot, end_id_spot = FRAME.minmax_optimFrame(frame_return, frames, min=10, max=33)

	# for test
	if SAVE_DIFF:
		video_name = videopath.split('\\')[-1]
		save_diff(video_name, frame_diffs)

	"""
	3. Return results
	"""
	# video_name = videopath.split('\\')[-1]
	# start = np.array(start_id_spot[1:])[np.newaxis, :]
	# end = np.array(end_id_spot[:-1])[np.newaxis, :]
	# spot = np.concatenate((end.T, start.T), axis=1)
	# np.savetxt(os.path.join(dst_path, video_name+'.txt'), start.T, fmt='%d', delimiter='\t')
	if OPTION:
		return start_id_spot
	else:
		return None


def SBD_ffmpeg(frame_list, OPTION='minmax'):
	frame_diffs = []
	frames = []

	# convert from PIL image to OpenCV
	# new_framelist = []
	# for pil_frame in frame_list:
	# 	numpy_frame = np.array(pil_frame)
	# 	cv_frame = cv2.cvtColor(numpy_frame, cv2.COLOR_RGB2BGR)
	# 	new_framelist.append(cv_frame)

	"""
	1. Calculate the difference between adjacent frames
	"""
	curr_frame = None
	prev_frame = None
	for idx, frame in enumerate(frame_list):
		luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
		curr_frame = luv
		if curr_frame is not None and prev_frame is not None:
			diff = cv2.absdiff(curr_frame, prev_frame)
			diff_sum = np.sum(diff)
			diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
			frame_diffs.append(diff_sum_mean)
		elif curr_frame is not None and prev_frame is None:
			diff_sum_mean = 0
			# frame_diffs.append(diff_sum_mean)
		prev_frame = curr_frame

	if NORM_OPTION:
		frame_diffs = MINMAX_norm(frame_diffs)

	FRAME = Frame(0, 0)
	for idx, frame in enumerate(frame_diffs):
		frame = Frame(idx+1, frame_diffs[idx])
		frames.append(frame)

	"""
	2. Find the boundary using the calculated difference
		1) detect the possible frame
		2) optimize the possible frame
	"""
	if OPTION == 'global':
		frame_return, start_id_spot, end_id_spot = FRAME.thresholding(frames, frame_diffs, judge=1.8)
		new_frame, start_id_spot, end_id_spot = FRAME.thresholding_optimFrame(frame_return, frames)
	elif OPTION == 'local':
		# frame_return, start_id_spot, end_id_spot = FRAME.find_possible_frame(frames)
		frame_return, start_id_spot, end_id_spot = FRAME.minmax_findFrame(frames, windowsize=5)
		frame_diffs_numpy = np.array(frame_diffs); average = np.average(frame_diffs_numpy)
		new_frame, start_id_spot, end_id_spot = FRAME.optimize_frame(frame_return, frames, average)
	elif OPTION == 'minmax':
		frame_return, start_id_spot, end_id_spot = FRAME.minmax_findFrame(frames, windowsize=5)
		new_frame, start_id_spot, end_id_spot = FRAME.minmax_optimFrame(frame_return, frames, min=4, max=5)

		# diff_list = [start_id_spot[idx+1] - start_id_spot[idx] for idx, i in enumerate(start_id_spot[1:])]
		# print(f'{min(diff_list)} / {max(diff_list)}')

	"""
	3. Return results
	"""
	if OPTION:
		return start_id_spot, end_id_spot
	else:
		return None


def TEST():
	def simulated_dataset(video, global_thr):
		if not os.path.isdir(str(global_thr)):
			os.makedirs(str(global_thr))
		datasetdir = '../AGAIN/'
		csvpath = os.path.join(str(global_thr), video + '.csv')
		f = open(csvpath, 'w')
		wr = csv.writer(f)
		for tp in transform:
			videoname = video
			if tp == 'original':
				videopath = os.path.join(datasetdir, tp, video)
			else:
				if tp == 'camcording':
					videoname = video.replace('.flv', '_' + tp + '.mp4')
				else:
					videoname = video.replace('.flv', '_' + tp + '.flv')
				videopath = os.path.join(datasetdir, tp, videoname)

			if os.path.isfile(videopath):
				print(videopath, end=' ')
				result = SBD(videopath, global_thr=global_thr)
				wr.writerow([tp] + result)
				if EXTRACT_RESULT:
					# if tp in ('original', 'framerate', 'camcording'):
					if tp in ('origianl', 'camcording', 'framerate'):
						checkresult(videopath, videoname, global_thr, result)
		f.close()

	def VCDB_24(query):
		datasetdir = '../test_VCDB_dir/'
		# src_path = os.path.join(datasetdir, query)
		# video_list = os.listdir(src_path)
		# video_list = [file for file in video_list if (file.endswith('.flv') or file.endswith('.mp4'))]
		src_path_glob = os.path.join(datasetdir, query, '*')
		videopath_list = glob.glob(src_path_glob)
		videopath_list = [file for file in videopath_list if (file.endswith('.flv') or file.endswith('.mp4'))]

		if not os.path.isdir(query):
			os.makedirs(query)
		csvpath = os.path.join(query, query + '.csv')
		f = open(csvpath, 'w')
		wr = csv.writer(f)
		c = 0
		for idx, video_path in enumerate(videopath_list):
			c += 1
			print(video_path + "   %d/%d" % (c, len(videopath_list)))
			video = video_path.split('\\')[-1]
			result = SBD(video_path, global_thr=1.4)
			wr.writerow([video] + result)
			if EXTRACT_RESULT:
				checkresult(video_path, video, query, result)
		f.close()


	query_dir = '../test_VCDB_dir/'
	query_list = os.listdir(query_dir)
	for query in query_list:
		VCDB_24(query)

	video_list = os.listdir('../AGAIN/original')
	for video in video_list:
		simulated_dataset(video)


if __name__ == '__main__':
	dst_path = '../VCDB_core/boundary/local/'
	if not os.path.isdir(dst_path):
		os.makedirs(dst_path)

	onefile = dict()
	onefile["VCDB_core"] = []
	VCDB_path = '../VCDB_core/videos/*'
	queries = glob.glob(VCDB_path)
	for query in queries:
		video_list = glob.glob(os.path.join(query, '*'))
		for video in video_list:
			videoname = os.path.basename(video)
			result = SBD(video)

			# text file
			txtpath = os.path.join(dst_path, videoname+'.txt')
			f = open(txtpath, "w")
			for idx in result:
				f.write(str(idx)+' ')
			f.close()

			# json file
			jsonpath = os.path.join(dst_path, videoname + '.json')
			json_object = {
				"video": videoname,
				"indexes": result
			}
			with open(jsonpath, 'w', encoding='utf-8') as make_file:
				json.dump(json_object, make_file, indent='\t')

			# one file
			onefile["VCDB_core"].append(json_object)
	onefilepath = 'VCDB_core.json'
	with open(onefilepath, 'w', encoding='utf-8') as make_file:
		json.dump(onefile, make_file, indent='\t')