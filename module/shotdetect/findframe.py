import numpy as np

class Frame:
	"""
	class to hold information about each frame
	"""
	def __init__(self, id, diff):
		self.id = id
		self.diff = diff

	def __lt__(self, other):
		if self.id == other.id:
			return self.id < other.id
		return self.id < other.id

	def __gt__(self, other):
		return other.__lt__(self)

	def __eq__(self, other):
		return self.id == other.id and self.id == other.id

	def __ne__(self, other):
		return not self.__eq__(other)

	def getMAXdiff(self, list=[]):
		"""
		find the max_diff_frame in the window
		"""
		LIST = list[:]
		temp = LIST[0]
		for i in range(0, len(LIST)):
			if temp.diff > LIST[i].diff:
				continue
			else:
				temp = LIST[i]

		return temp

	def getTopNdiff(self, N, a=[]):
		"""
		find the top N max_diff_frame in the window
		"""
		result = []
		diff_list = [i for i in sorted(a, key=lambda x:x.diff, reverse=True)]

		if N > len(a):
			N = len(a)
		for idx in range(0,N):
			result.append(diff_list[idx])
		return result

	def find_possible_frame(self, list_frames):
		#parameter
		window_size = 4
		m_suddenJudge = 1.0
		m_MinLengthOfShot = 1
		possible_frame = []
		window_frame = []
		start_id_spot = []
		start_id_spot.append(0)
		end_id_spot = []

		length = len(list_frames)
		index = 0
		while (index < length):
			frame_item = list_frames[index]
			window_frame.append(frame_item)
			if len(window_frame) < window_size:
				index += 1
				if index == length - 1:
					window_frame.append(list_frames[index])
				else:
					continue

			max_diff_frame = self.getMAXdiff(window_frame)
			max_diff_id = max_diff_frame.id

			if len(possible_frame) == 0:
				possible_frame.append(max_diff_frame)
				continue
			last_max_frame = possible_frame[-1]

			"""
			Check whether the difference of the selected frame is more than 
			(m_suddenJudge) times the average difference of the other frames in the window.
			"""
			sum_start_id = last_max_frame.id + 1
			sum_end_id = max_diff_id - 1
			id_no = sum_start_id
			sum_diff = 0
			while True:
				sum_frame_item = list_frames[id_no]
				sum_diff += sum_frame_item.diff
				id_no += 1
				if id_no > sum_end_id:
					break
			average_diff = sum_diff / (sum_end_id - sum_start_id + 1)
			if max_diff_frame.diff >= (m_suddenJudge * average_diff):
				possible_frame.append(max_diff_frame)
				window_frame = []
				index = possible_frame[-1].id + m_MinLengthOfShot
				continue
			else:
				index = max_diff_frame.id + 1
				window_frame = []
				continue

		"""
		save the index of the possible frames
			possible_frame: list of possible frames in class FRAME type
			start_id_spot, end_id_spot: The start and end indexes of the intervals bounded by possible frames
		"""
		for i in range(0, len(possible_frame)):
			start_id_spot.append(possible_frame[i].id)
			end_id_spot.append(possible_frame[i].id - 1)
		sus_last_frame = possible_frame[-1]
		last_frame = list_frames[-1]
		if sus_last_frame.id < last_frame.id:
			possible_frame.append(last_frame)
			end_id_spot.append(possible_frame[-1].id)

		return possible_frame, start_id_spot, end_id_spot


	def optimize_frame(self, tag_frames, list_frames, diff_thr):
		#parameter
		frame_count = 4
		diff_threshold = diff_thr
		diff_optimize = 1.7
		new_tag_frames = []
		start_id_spot = []
		start_id_spot.append(0)
		end_id_spot = []

		for tag_frame in tag_frames: # possible frames
			tag_id = tag_frame.id
			"""
			1. check whether the difference of the possible frame is no less than (diff_thr).
			"""
			if tag_frame.diff < diff_threshold:
				continue
			"""
			2. check whether the difference is more than twice the average difference of 
			the previous (frame_count) frames and the subsequent (frame_count) frames (total (frame_count)*2 frames).
			"""
			# (1) get the previous frames
			pre_start_id = tag_id - frame_count
			pre_end_id = tag_id - 1
			if pre_start_id < 0:
				continue

			pre_sum_diff = 0
			check_id = pre_start_id
			while True:
				pre_frame_info = list_frames[check_id]
				pre_sum_diff += pre_frame_info.diff
				check_id += 1
				if check_id > pre_end_id:
					break

			# (2) get the subsequent frames
			back_start_id = tag_id + 1
			back_end_id = tag_id + frame_count
			if back_end_id >= len(list_frames):
				continue

			back_sum_diff = 0
			check_id = back_start_id
			while True:
				back_frame_info = list_frames[check_id]
				back_sum_diff += back_frame_info.diff
				check_id += 1
				if check_id > back_end_id:
					break

			# (3) calculate the difference of the previous (frame_count) frames and the subsequent (frame_count) frames
			sum_diff = pre_sum_diff + back_sum_diff
			average_diff = sum_diff / (frame_count * 2)

			# (4) check whether the requirement is met or not
			if tag_frame.diff > (diff_optimize * average_diff):
				new_tag_frames.append(tag_frame)

		"""
		save the index of the possible frames
		"""
		if new_tag_frames == []:
			return [], [], []

		for i in range(0, len(new_tag_frames)):
			start_id_spot.append(new_tag_frames[i].id)
			end_id_spot.append(new_tag_frames[i].id - 1)
		last_frame = list_frames[-1]
		if new_tag_frames[-1].id < last_frame.id:
			new_tag_frames.append(last_frame)
		end_id_spot.append(new_tag_frames[-1].id)

		return new_tag_frames, start_id_spot, end_id_spot

	def thresholding(self, frames, frame_diffs, judge):
		possible_frame = []
		start_id_spot = []
		start_id_spot.append(0)
		end_id_spot = []

		"""
		If a diff is higher than the average of all diffs * (judge), it is considered as possible frame.
		"""
		temp = np.array(frame_diffs); average = np.average(temp)
		for frame in frames:
			if frame.diff >= judge * average:
				start_id_spot.append(frame.id)
				end_id_spot.append(frame.id-1)
				possible_frame.append(frame)
		end_id_spot.append(len(frames))

		return possible_frame, start_id_spot, end_id_spot

	def thresholding_optimFrame(self, tag_frames, list_frame):
		new_possible_frame = []
		start_id_spot = []
		start_id_spot.append(0)
		end_id_spot = []

		"""
		Among possible frames, diffs attached without gaps use only the maximum of them.
		"""
		prev = tag_frames[0]
		temp = [prev]
		for curr in tag_frames[1:]:
			if curr.id - prev.id <= 1:
				temp.append(curr)
			else:
				max = self.getMAXdiff(temp)
				new_possible_frame.append(max)
				del temp
				temp = [curr]
			prev = curr
		max = self.getMAXdiff(temp)
		new_possible_frame.append(max)

		if new_possible_frame == []:
			return [], [], []

		for i in range(0, len(new_possible_frame)):
			start_id_spot.append(new_possible_frame[i].id)
			end_id_spot.append(new_possible_frame[i].id - 1)
		last_frame = list_frame[-1]
		if new_possible_frame[-1].id < last_frame.id:
			new_possible_frame.append(last_frame)
		end_id_spot.append(new_possible_frame[-1].id)

		return new_possible_frame, start_id_spot, end_id_spot

	def minmax_findFrame(self, list_frames, windowsize=5, N=1):
		# parameter
		window_size = windowsize
		possible_frame = []
		start_id_spot = []
		start_id_spot.append(0)
		end_id_spot = []

		"""
		Find the maximum value within windowsize for all windows
		"""
		length = len(list_frames)
		index = 0
		while (index < length):
			end = index + window_size
			if end > length:
				end = length
			window_frame = list_frames[index:end]

			topN_list = self.getTopNdiff(N=N, a=window_frame)
			sorted_topN_list = [i for i in sorted(topN_list, key=lambda x:x.id)]
			possible_frame += sorted_topN_list
			index += window_size

		for i in range(0, len(possible_frame)):
			start_id_spot.append(possible_frame[i].id)
			end_id_spot.append(possible_frame[i].id - 1)
		sus_last_frame = possible_frame[-1]
		last_frame = list_frames[-1]
		if sus_last_frame.id < last_frame.id:
			possible_frame.append(last_frame)
			end_id_spot.append(possible_frame[-1].id)

		return possible_frame, start_id_spot, end_id_spot

	def minmax_optimFrame(self, tag_frames, list_frames, min=5, max=14):
		new_tag_frames = []
		start_id_spot = []
		start_id_spot.append(0)
		end_id_spot = []

		"""
		Optimize possible frames so that the boundary interval has a minimum (min) and maximum (max) length.
		"""
		prev_frame = None
		for tag_frame in tag_frames:
			curr_frame = tag_frame

			if prev_frame is None:
				new_tag_frames.append(curr_frame)
				prev_frame = curr_frame
				continue
			else:
				if curr_frame.id-prev_frame.id < min:
					if curr_frame.diff <= prev_frame.diff:
						continue
					else:
						new_tag_frames.remove(prev_frame)
						new_tag_frames.append(curr_frame)
						prev_frame = curr_frame
				elif curr_frame.id-prev_frame.id > max:
					new_tag_frames.append(curr_frame)
					prev_frame = curr_frame
				else:
					new_tag_frames.append(curr_frame)
					prev_frame = curr_frame

		if new_tag_frames == []:
			return [], [], []

		for i in range(0, len(new_tag_frames)):
			start_id_spot.append(new_tag_frames[i].id)
			end_id_spot.append(new_tag_frames[i].id - 1)
		last_frame = list_frames[-1]
		if new_tag_frames[-1].id < last_frame.id:
			new_tag_frames.append(last_frame)
		end_id_spot.append(new_tag_frames[-1].id)

		return new_tag_frames, start_id_spot, end_id_spot