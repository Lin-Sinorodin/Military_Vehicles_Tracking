import os
import torch

from src.video import Video
from src.tracker import MultiObjectTracker
from src.plot_utils import plot_bounding_boxes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
"""
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial/54533223#54533223
"""

print(f'torch {torch.__version__}')
print(torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU')

if __name__ == '__main__':
    input_video_path = 'test_videos/test1.mp4'
    detected_video_path = 'test3_result.mp4'
    detector_path = 'YOLOv5_runs/one_class_YOLOV5s/weights/best.pt'

    video = Video(input_video_path)
    yolov5s_detector = torch.hub.load('ultralytics/yolov5', 'custom', detector_path).to(device)
    tracker = MultiObjectTracker(video, detected_video_path, yolov5s_detector)

    for frame, bounding_boxes in tracker:
        plot_bounding_boxes(frame, bounding_boxes)
        tracker.video_writer.write(frame)
    tracker.video_writer.release()
    print('=> Saved tracking results video')
