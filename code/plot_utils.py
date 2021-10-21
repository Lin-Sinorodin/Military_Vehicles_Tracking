import cv2


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def plot_one_box(bboxes, img, color=(128, 128, 128), label=None, line_thickness=3):
    """
    Plots one bounding box on image 'img' using OpenCV.
    Source: https://github.com/ultralytics/yolov5/blob/master/utils/plots.py
    """
    assert img.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(img) to plot_on_box() input image.'

    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_bounding_boxes(img, deepsort_tracking):
    # draw boxes for visualization
    if len(deepsort_tracking) > 0:
        for output in deepsort_tracking:
            bboxes, id, cls = output[0:4], output[4], output[5]
            label = f'{id}'
            color = compute_color_for_id(id)
            plot_one_box(bboxes, img, color, label=label, line_thickness=2)
