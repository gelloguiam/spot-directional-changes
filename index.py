import cv2
import numpy as np
import sys

# from tracker import *

#decalare constants here
INPUT_VID_FILENAME = "data/video/physci-edited-480p.mov"
INPUT_VID_FILENAME = "data/video/car-overhead-3.mov"
OUTPUT_VID_FILENAME = "data/video/sample.mov"
# tracker = EuclideanDistTracker()
colors = {}


def preprocess(image):                              #enhance video quality before background subtraction
    clahe = cv2.createCLAHE(clipLimit = 1.0)          #apply CLAHE
    enhanced_img = clahe.apply(image)
    return enhanced_img


def clahe_img(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit = 1.0)

    channels = cv2.split(hsv_img)
    channels[0] = clahe.apply(channels[0])
    channels[1] = clahe.apply(channels[1])
    channels[2] = clahe.apply(channels[2])

    hsv_img = cv2.merge(channels)
    hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    img = cv2.detailEnhance(hsv_img, sigma_s=10, sigma_r=0.15)
    return img


def background_subtraction(curr_frame, prev_frame):
    frame_diff = cv2.absdiff(curr_frame, prev_frame)
    return frame_diff


def morphological_transform(image):
    blur = cv2.GaussianBlur(image, (5,5), 0)
    _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    trasnformed = cv2.dilate(th, kernel, iterations=1)
    trasnformed = cv2.morphologyEx(trasnformed, cv2.MORPH_OPEN, kernel)
    # trasnformed = cv2.morphologyEx(trasnformed, cv2.MORPH_CLOSE, kernel)

    return trasnformed


def blob_detection(image, org_img):
    components = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = components
    
    output = org_img.copy()
    box_ids = None

    detections = []

    for i in range(0, numLabels):
        if (i==0): continue                                              #exclude backgrouund

        x = stats[i, cv2.CC_STAT_LEFT]                                   #get bounding box
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area > 200:
            detections.append([x, y, w, h])

            (centroid_x, centroid_y) = centroids[i]                          #get centroids
            component_mask = (labels == i).astype("uint8") * 255             #get component mask
            color_list = (list(np.random.choice(range(256), size=3)))        #generate random color for each blob
            random_color = (int(color_list[0]), int(color_list[1]), int(color_list[2]))
            # output[(component_mask != 0)] = random_color                      #color the blob

    # boxes_ids = tracker.update(detections)
    # for box_id in boxes_ids:

    #     color_list = (list(np.random.choice(range(256), size=3)))        #generate random color for each blob
    #     random_color = (int(color_list[0]), int(color_list[1]), int(color_list[2]))

    #     x, y, w, h, id = box_id

    #     color = ()
    #     if id in colors.keys():
    #         color = colors[id]
    #     else:
    #         colors[id] = random_color
    #         color = random_color

    #     cv2.putText(output, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    #     cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)

        # cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 1)

    return output


def main():
    cap = cv2.VideoCapture(INPUT_VID_FILENAME)              #load video stream
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    if not cap.isOpened():
        print("Could not open video file " + str(INPUT_VID_FILENAME))
        sys.exit(1)

    # height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*2)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*2)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*2)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(OUTPUT_VID_FILENAME, codec, fps, (width, height))

    backSub = cv2.createBackgroundSubtractorKNN()


    prev_frame = np.array([])     
    frame_number = 0                          #save previous frame
    while(cap.isOpened()):                                  #loop all frames
        ret, frame = cap.read()                             #capture each frame

        frame_number = frame_number +1
        if frame_number < 400: continue

        width = int(frame.shape[1] * 2)
        height = int(frame.shape[0] * 2)
        dim = (width, height)

        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        mask = backSub.apply(frame)
        extracted = cv2.bitwise_and(frame, frame, mask=mask)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #convert to grayscale
        enhanced_img = preprocess(gray)                     #apply CLAHE

        if (prev_frame.size != 0):
            foreground = background_subtraction(enhanced_img, prev_frame)
            transformed = morphological_transform(mask)
            objects = blob_detection(transformed, frame)

            # bg = 1 - transformed
            fg_mask = transformed
            bg_mask = 255 - transformed

            extracted_fg = cv2.bitwise_and(frame, frame, mask=fg_mask)
            extracted_bg = cv2.bitwise_and(frame, frame, mask=bg_mask)

            extracted_fg = clahe_img(extracted_fg)

            # extracted_bg = cv2.GaussianBlur(extracted_bg, (9,9), 0)
            extracted_bg = cv2.edgePreservingFilter(extracted_bg, flags=1, sigma_s=60, sigma_r=0.1)

            final = cv2.add(extracted_fg, extracted_bg)

            # cv2.imshow("Original", hsv)
            # cv2.imshow("Gray", gray)
            # cv2.imshow("CLAHE", enhanced_img)
            # cv2.imshow("Foreground", foreground)
            cv2.imshow("FG", extracted_fg)
            cv2.imshow("BG", extracted_bg)
            cv2.imshow("Objects", final)

            video_writer.write(final)

            c = cv2.waitKey(1) % 0x100                      #listen for ESC key
            if c == 27: break

        prev_frame = enhanced_img

    cap.release()                                           #close video stream
    cv2.destroyAllWindows()                                 #close all windows

if __name__ == "__main__":
    main()