# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.
TEST_DATA=../all_models
Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite
Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt
"""
import argparse
import collections
import common
import cv2
import numpy as np
import os
from PIL import Image
import re
import tflite_runtime.interpreter as tflite

###########################################
import socket
import pickle
import struct

TCP_IP = '222.251.196.102'
TCP_PORT = 8485

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect((TCP_IP, TCP_PORT))
connection = clientsocket.makefile('wb')
###########################################

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))
    ## CHANGED THIS SO IT ONLY RETURNS BOXES FOR BUSES
    return [make(i) for i in range(top_k) if ((scores[i] >= score_threshold) and (class_ids[i] == 5))]

def get_digits(objs, image):
    height, width, channels = image.shape
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        h = y1 - y0
        w = x1 - x0
        cropped_img = image[int(y):int(y) + int(h/3), int(x) + int(1*w/10):int(x) + int(7*w/10)]  #--- Notice this part where you have to add the stride as well ---
        #cv2.imshow('img',cropped_img)
        # copy to draw on
        draw = cropped_img.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
        # # preprocess image for network
        cropped_img = preprocess_image(cropped_img)
        cropped_img, _ = resize_image(cropped_img, 416, 448)
        #plt.imshow(cropped_img);

    
        # process image
        start = time.time()
        boxes_d, scores, labels = model.predict_on_batch(np.expand_dims(cropped_img, axis=0))
        print("processing time: ", time.time() - start)
    
        boxes_d = post_process(boxes_d, draw, cropped_img)
        labels = labels[0]
        scores = scores[0]
        boxes_d = boxes_d[0]
        print(labels.shape)
        print(boxes_d.shape)
        print(scores[0])
                        
       ## de momento poner lo aki pero todo esto pasarlo fuera
       visualize_boxes(draw, boxes_d, labels, scores, class_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    
       # # 5. plot    
       # plt.imshow(draw)
       # plt.show()
    
def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    cap = cv2.VideoCapture(0)
    ###############################For sending to the socket
    #cap.set(3, 320);
    #cap.set(4, 240);

    img_counter = 0

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
######################################################################33

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)

        common.set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)
        cv2_im_bus = append_objs_to_img(cv2_im, objs, labels)
        get_digits(objs, cv2_im)
        
        cv2.imshow('frame', cv2_im_bus)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()