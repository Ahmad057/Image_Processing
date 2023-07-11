import cv2, os
import json
import math
import time

import warnings
import numpy as np
import pandas as pd
from bson import json_util
from yolov8 import YOLOv8
from extract_motion_vector import *
warnings.filterwarnings('ignore')


model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)


def key_frame_detect(index):
    with open('key.txt', 'r') as file:
        text = file.read()
        values = text.split(',')

        values = [int(value) for value in values if value != ""]
        return index in values


def coco_json_classes():
    """
    This method is reading the coco json object file
    and returning the names of the all classes
    :return: class labels of coco dataset
    """
    try:
        coco_json_path = 'coco_object.json'
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        return [category["name"] for category in coco_data['categories']]

    except Exception:
        print("Coco object file not exist")

class_names = coco_json_classes()



def abs_coordinates(coordinates, img_shape):
    actual_height, actual_width = img_shape[0], img_shape[1]

    _x1 = int((coordinates[0] / 640) * actual_width)
    _y1 = int((coordinates[1] / 640) * actual_height)

    _x2 = int((coordinates[2] / 640) * actual_width)
    _y2 = int((coordinates[3] / 640) * actual_height)

    print(_x1 ,_y1, _x2, _y2)
    return _x1 ,_y1, _x2, _y2
  

def get_bbox_from_file():
    data_dict = {}

    # Read the text file
    with open('bbox.txt', 'r') as file:
        # Iterate over each line in the file
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            values = line.split()  # Split the line into individual values

            index = int(values[0])  # Extract the index value
            data = list(map(int, values[1:]))  # Convert the remaining values to integers

            if index not in data_dict:
                data_dict[index] = []  # Create a new list for the index if it doesn't exist

            data_dict[index].append(data)  # Add the data to the corresponding index

    return data_dict



def start_process(video_name):
    path = "videos"
    cap = cv2.VideoCapture(os.path.join(path, video_name))


    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (frame_width, frame_height)
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter('output_video/processed.avi', fcc, fps, size)

    bbox_data = get_bbox_from_file()
    file_write = open("pred_bbox_vectors.txt", "a")

    key_frame = 0
    objects_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        frame_status = ""
        if not ret:
            break
        
        if detection_flag := key_frame_detect(key_frame):
            count_obj = 0
            frame_status = "Detection"
            objects_list = []

            # boxes, scores, class_ids = yolov8_detector(frame)
            
            # print(bbox_data[key_frame])
            # print("frame-no: ", key_frame)

            
            
            # for count_obj, (box, class_id) in enumerate(zip(boxes, class_ids)):
            #     box = list(box)
            #     class_id = int(class_id)
                
            #     left_bbox, top_bbox, right_bbox, bottom_bbox = map(int, box)
            #     # print("bbox: ", left_bbox, top_bbox, right_bbox, bottom_bbox)
            #     top_bbox = max(top_bbox, 0)
            #     left_bbox = max(left_bbox, 0)
            #     cv2.rectangle(frame, (int(left_bbox), int(top_bbox)), (int(right_bbox), int(bottom_bbox)), (0, 0, 255), 5)
            #     cv2.putText(frame, f"ID: {count_obj}", (int(left_bbox), int(top_bbox) - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.5, (255, 0, 255), 2)

            for box in bbox_data[key_frame]:
                left_bbox, top_bbox, right_bbox, bottom_bbox = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                # left_bbox, top_bbox, right_bbox, bottom_bbox = abs_coordinates([left_bbox, top_bbox, right_bbox, bottom_bbox], frame.shape)

                top_bbox = max(top_bbox, 0)
                left_bbox = max(left_bbox, 0)
                cv2.rectangle(frame, (int(left_bbox), int(top_bbox)), (int(right_bbox), int(bottom_bbox)), (0, 0, 255), 5)
              

            
                objects_list.append([int(left_bbox), int(top_bbox), int(right_bbox), int(bottom_bbox)])
            # objects_list = boxes
                file_write.write(f"{key_frame} {left_bbox} {top_bbox} {right_bbox} {bottom_bbox}\n")


        else:
            frame_status = "Tracking"
            motion_vector, motion_file_coor = get_motion_vector(key_frame, objects_list)
            # print("motion vector: ", motion_vector)
            if motion_vector is None:
                print(motion_vector)
            
            objects_list = motion_vector
            if motion_vector is not None:
                for i, j in zip(range(len(motion_vector)), range(len(motion_file_coor))):
                    if len(motion_vector[i]) > 0:
                        # print("&&&&&&&&&&&7: ", motion_file_coor[j])
                        x1, y1 = motion_vector[i][0], motion_vector[i][1]
                        x2, y2 = motion_vector[i][2], motion_vector[i][3]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 155), 5)
            
                        file_write.write(f"{key_frame} {x1} {y1} {x2} {y2} \t {motion_file_coor[j][0]} {motion_file_coor[j][1]} {motion_file_coor[j][2]} {motion_file_coor[j][3]} \t {motion_file_coor[j][4]} {motion_file_coor[j][5]} {motion_file_coor[j][6]} {motion_file_coor[j][7]} \t {motion_file_coor[j][8]} {motion_file_coor[j][9]} {motion_file_coor[j][10]} {motion_file_coor[j][11]} \t {motion_file_coor[j][12]} {motion_file_coor[j][13]} {motion_file_coor[j][14]} {motion_file_coor[j][15]}\n")

        cv2.putText(frame, f"Here: {frame_status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 0, 255), 5)
        cv2.imshow("Frame", frame)
        out.write(frame)
        

        key_frame+=1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    file_write.close()
    cap.release()
    cv2.destroyAllWindows()


start_process("4.mp4")

