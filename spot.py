from re import X
import cv2
import core.utils as utils
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf
import time

from absl import app, flags, logging
from absl.flags import FLAGS
from core.config import cfg
from core.yolov4 import filter_boxes
from PIL import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.saved_model import tag_constants
import spot_library as spot

from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_float('iou', 0.20, 'iou threshold')
flags.DEFINE_float('score', 0.10, 'score threshold')
flags.DEFINE_string('video', './data/video/physci-edited-480p.mov', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', 'output/output.mp4', 'path to output video')
flags.DEFINE_string('direction', '', 'pedestrian direction to map')
flags.DEFINE_integer('mode', 0, '0 - detect pedestrian significantly changing direction, 1 - detect pedestrian who go against the general direction of all objects in the frame, 2 - detect pedestrian who go against the set direction')
flags.DEFINE_boolean('rotate', False, 'rotate image')
flags.DEFINE_boolean('scale', False, 'scale image')
flags.DEFINE_boolean('preprocess', False, 'preprocess the image before tracking')


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 5.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    frame_num = 0

    # get direction detection mode
    detection_mode = FLAGS.mode

    if detection_mode == 2:
        default_direction = FLAGS.direction
    else:
        default_direction = ""

    # get processing attributes
    rotate_flag = FLAGS.rotate
    scale_flag = FLAGS.scale
    preprocess_flag = FLAGS.preprocess

    #create background model
    backSub = cv2.createBackgroundSubtractorKNN()

    #define video writer to output video
    video_writer = None

    if scale_flag:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)*2)
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)*2)

    else:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if rotate_flag:
        tmp = width
        width = height
        height = tmp

    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        return_value, frame = vid.read()
        if return_value:

            # rotate frame by 90 degrees if rotate flag is True
            if rotate_flag:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # scale frame to 200% if scale flag is True
            if scale_flag:
                width = int(frame.shape[1] * 2)
                height = int(frame.shape[0] * 2)
                dim = (width, height)
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

            # enahnce frame if preprocess flag is True
            if preprocess_flag:
                fg, bg, frame = spot.preprocessing(frame, backSub)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        frame_num += 1

        directions = []

        print('Frame #: ', frame_num)
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())        
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        count = len(names)

        session_title = "Tracking pedestrian "

        if(detection_mode == 0):
            session_title += "significantly changing direction"
        else:
            session_title += "not moving in {} direction".format(default_direction)

        session_summary = "Frame {} tracking {} pedestrian".format(frame_num, count)

        cv2.rectangle(frame, (10, 12), (15+(len(session_title)*11), 35), (0, 0, 0), -1)
        cv2.putText(frame, session_title, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 1)

        cv2.rectangle(frame, (10, 42), (15+(len(session_summary)*12), 65), (0, 0, 0), -1)
        cv2.putText(frame, session_summary, (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 1)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            x1 = int(bbox[0])
            x2 = int(bbox[2])
            y1 = int(bbox[1])
            y2 = int(bbox[3])

            area = (x2-x1)*(y2-y1)
            if area > 100000: continue

            # get centroid of the bounding box
            cX = int((x1 + x2)/2)
            cY = int((y1 + y2)/2)
            
            curr_centroid = (cX, cY)
            prev_centroid_1 = track.get_last_centroid(1)
            prev_centroid_2 = track.get_last_centroid(2)

            slope = 0

            dir_1 = ""
            dir_2 = ""

            track_direction = None
            if prev_centroid_1 is not None and prev_centroid_2 is not None:
                track_direction = spot.get_direction(prev_centroid_2, curr_centroid)

                # store directions of all objects in the frame
                directions.append(track_direction)

                #check if position change is significant to compare
                distance = spot.get_distance(prev_centroid_1, curr_centroid)
                if distance > 20:
                    slope = spot.get_slope(prev_centroid_1, curr_centroid)
                    dir_1 = spot.get_direction(prev_centroid_1, curr_centroid)
                    dir_2 = spot.get_direction(prev_centroid_2, prev_centroid_1)

            # update centroid history of object
            track.update_centroid(cX, cY)

            # check if slope of centroid points are significant to decide that the direction change is valid
            sig_slope = abs(slope) >= 1

            direction_change = ""
            # compare directions for each selected centroid if detection mode is set to 0 (tracking significant change)
            if detection_mode == 0:
                direction_change = dir_1 != dir_2
            elif track_direction is not None:
                # compare direction against the set direction, mode 1 to use general mass direction, mode 2 to use user-specified direction 
                direction_change = track_direction != default_direction

            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            annotation = "{}".format(str(track.track_id))
            if track_direction is not None:
                annotation = "{}-{}".format(str(track.track_id), track_direction)

            # mark objects detected with significant change in direction
            if (detection_mode == 0 and direction_change and sig_slope) or (detection_mode > 0 and direction_change):
                annotation = "ALERT-{}".format(track_direction)
                print(track.track_id, prev_centroid_2, prev_centroid_1, curr_centroid, dir_1, dir_2, direction_change, slope, distance)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(annotation))*16, int(bbox[1])), (255, 0, 0), -1)
                cv2.putText(frame, annotation,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255), 1)

                # all_centroids = track.centroids
                # for cent in all_centroids:
                #     cv2.circle(frame, cent, radius=0, color=(255, 0, 0), thickness=5)

            # annotate object for tracking
            else:    
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(annotation))*16, int(bbox[1])), color, -1)
                cv2.putText(frame, annotation,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255), 1)

        # update default direction to the dominant direction of all objects in the frame
        if(detection_mode == 1 and len(directions) > 0):
            default_direction = max(set(directions), key=directions.count)
    
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Output Video", result)

        # write the result
        video_writer.write(result)

        # ESC to end video
        c = cv2.waitKey(1) % 0x100
        if c == 27: break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
