import cv2
import os
import numpy as np
from tabulate import tabulate
from tqdm.autonotebook import tqdm


class Video:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_cap = cv2.VideoCapture(video_path)

        self.num_frames = self._get_num_frames()
        self.fps = self._get_fps()
        self.height, self.width = self._get_frames_dimension()

    def __str__(self):
        video_details = [
            ['Video path', self.video_path],
            ['Number of frames', self.num_frames],
            ['FPS', self.fps],
            ['(height, width)', f'({self.height}, {self.width})']
        ]
        return tabulate(video_details)

    def __iter__(self):
        for _ in range(self.num_frames):
            success, frame = self.video_cap.read()
            if success is False or frame is None:
                break
            yield frame

        self.video_cap.release()

    def _get_num_frames(self):
        num_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert num_frames > 0, 'The video contains 0 frames.'
        return num_frames

    def _get_fps(self):
        return int(self.video_cap.get(cv2.CAP_PROP_FPS))

    def _get_frames_dimension(self):
        _, frame = cv2.VideoCapture(self.video_path).read()
        height, width, channels = frame.shape
        return height, width

    def _frames_progress_bar(self, description):
        return tqdm(enumerate(self), desc=description, unit='frame', total=self.num_frames)

    def export_frames(self, frames_path):
        os.makedirs(frames_path, exist_ok=True)
        file_name_zero_padding = len(str(self.num_frames))
        self.video_cap = cv2.VideoCapture(self.video_path)

        for count, frame in self._frames_progress_bar(description='Exporting frames'):
            frame_number = str(count).zfill(file_name_zero_padding)
            frame_path = f'{frames_path}/{frame_number}.jpg'
            cv2.imwrite(frame_path, frame)

    def get_video_writer(self, video_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        dimension = (self.width, self.height)
        video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, dimension)
        return video_writer

    def get_frames_tensor(self):
        frames = []
        for frame in self:
            frames.append(frame)
        return np.array(frames)
