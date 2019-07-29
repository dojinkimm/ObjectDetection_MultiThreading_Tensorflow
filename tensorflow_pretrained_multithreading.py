from __future__ import division, print_function
import numpy as np
import os
import tensorflow as tf
import cv2
import argparse
import imutils

from utils import ops as utils_ops, detection_boxes_tensorflow as vis
from utils import label_map_util
import time
import sys


from thread_w_return import *


def arg_parse():
    parser = argparse.ArgumentParser(description='Tensorflow Pretrained')
    parser.add_argument("--video", help="Path where video is located",
                        default="assets/cars.mp4", type=str)
    parser.add_argument("--frozen", help="Frozen inference pb file",
                        default="faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb")
    parser.add_argument("--conf", dest="confidence", help="Confidence threshold for predictions", default=0.5)
    parser.add_argument("--save", help="Whether to save newly created video", default=False, type=bool)
    return parser.parse_args()


def run_inference_for_single_image(image, tensor_dict, sess, detection_graph):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    out_dict = sess.run(
        tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)}
    )

    # all outputs are float32 numpy arrays, so convert types as appropriate
    out_dict['num_detections'] = int(out_dict['num_detections'][0])
    out_dict['detection_classes'] = out_dict[
        'detection_classes'][0].astype(np.uint8)
    out_dict['detection_boxes'] = out_dict['detection_boxes'][0]
    out_dict['detection_scores'] = out_dict['detection_scores'][0]
    if 'detection_masks' in out_dict:
        out_dict['detection_masks'] = out_dict['detection_masks'][0]
    return out_dict


def detection_gpu(frame_list, device_name, sess, tensor_dict, category_index, confidence, detection_graph):
    frame_with_rect = []
    with tf.device(device_name):
        for frame in frame_list:
            start = time.time()
            output_dict = run_inference_for_single_image(frame, tensor_dict, sess, detection_graph)

            # Visualization of the results of a detection.
            vis.visualize_boxes_and_labels_rcnn(
                frame,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                min_score_thresh=confidence)
            end = time.time()

            cv2.putText(frame, '{:.2f}ms'.format((end - start) * 1000), (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            frame_with_rect.append(frame)
            cv2.imshow(device_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    return frame_with_rect


def main():
    args = arg_parse()

    VIDEO_PATH = args.video

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('labels', 'mscoco_label_map.pbtxt')

    # Load a Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.io.gfile.GFile(args.frozen, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    try:
        # Read Video file
        cap = cv2.VideoCapture(VIDEO_PATH)
    except IOError:
        print("Input video file", VIDEO_PATH, "doesn't exist")
        sys.exit(1)

    if args.save:
        video_width = int(cap.get(3))
        video_height = int(cap.get(4))
        video_fps = int(cap.get(5))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))

    # gpu가 몇개인지 알아낸다
    gpus = tf.config.experimental.list_logical_devices('GPU')
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # gpu개수에 맞춰서 frame을 나눈다
    div = frame_length // len(gpus)
    divide_point = [i for i in range(frame_length) if i != 0 and i % div == 0]
    divide_point.pop()

    frame_list = []
    fragments = []
    count = 0
    while cap.isOpened():
        hasFrame, frame = cap.read()
        if not hasFrame:
            frame_list.append(fragments)
            break
        if count in divide_point:
            frame_list.append(fragments)
            fragments = []
        fragments.append(frame)
        count += 1
    cap.release()

    with detection_graph.as_default():
        with tf.Session() as sess:

            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            # Threading을 통해서 여러 frame들을 detection하게 한다
            thread_detection = [ThreadWithReturnValue(target=detection_gpu,
                                                      args=(frame_list[i], gpu.name, sess, tensor_dict, category_index, args.confidence, detection_graph))
                                for i, gpu in enumerate(gpus)]

            final_list = []
            # Threading 을 시작한다
            for th in thread_detection:
                th.start()

            # Threading 이 끝나면 return 받은 값을 새로운 리스트에 담는다
            for th in thread_detection:
                final_list.extend(th.join())

            # return 받은 value 를 video 에 작성한다
            if args.save:
                for f in final_list:
                    videoWriter.write(f)


if __name__ == "__main__":
    s = time.time()
    main()
    e = time.time()
    print((e-s))

# tensorflow pretrained - 46.5966272354126