import argparse
import time
from typing import Dict, List

# image
import cv2
import face_recognition
from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt

'''
# face detector
face_net = cv2.dnn.readNet('models/opencv_face_detector.pbtxt',
                           "models/opencv_face_detector_uint8.pb")
FACE_MIN_CONFIDENCE = 0.7
'''

# age
age_net = cv2.dnn.readNet('models/age_deploy.prototxt',
                          'models/age_net.caffemodel')
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']


def load_image_file(file):
    """load image into numpy array (RGB)"""
    im = Image.open(file)
    im = im.convert('RGB')
    return np.array(im)


'''
def detect_face_locations(frame):
    """ find locations of face (top, right, bottom, left)"""
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123],swapRB=False, crop=False)
    frame_copy = frame.copy()

    face_net.setInput(blob)
    detections = face_net.forward()
    face_locations = []
    for index in range(0, detections.shape[2]):
        confidence = detections[0, 0, index, 2]
        if confidence > FACE_MIN_CONFIDENCE:
            box_left = int(detections[0, 0, index, 3] * frame_width)
            box_top = int(detections[0, 0, index, 4] * frame_height)
            box_right = int(detections[0, 0, index, 5] * frame_width)
            box_bottom = int(detections[0, 0, index, 6] * frame_height)
            face_locations.append((box_top, box_right, box_bottom, box_left))
    return face_locations
'''


def select_face_from_ref(locs: List, ref: Dict[str, int]):
    # select the closest face centroid
    selected = {'index': None, 'distance': None}
    centroid = {
        'x': (ref['position_right'] + ref['position_left']) / 2,
        'y': (ref['position_bottom'] + ref['position_top']) / 2}
    for index, loc in enumerate(locs):
        loc_x = (loc[1] + loc[3]) / 2
        loc_y = (loc[2] + loc[0]) / 2
        dist = (centroid['x']-loc_x)**2 + (centroid['y']-loc_y)**2
        if selected['index'] == None:
            selected['index'] = index
            selected['distance'] = dist
        else:
            if dist < selected['distance']:
                selected['index'] = index
                selected['distance'] = dist
    return locs[selected['index']]


def predict(img, ref_position: Dict[str, int] = None):
    """predict age from image"""
    frame = load_image_file(img)
    face_locations = face_recognition.face_locations(frame)

    # select face
    if face_locations and ref_position:
        print("INFO: use face_locations and ref_position")
        face_location = select_face_from_ref(face_locations, ref_position)
    elif ref_position:
        # use reference position to force encoding
        print("INFO: use only ref_position")
        face_location = (ref_position['position_top'],
                         ref_position['position_right'],
                         ref_position['position_bottom'],
                         ref_position['position_left'])
    elif face_locations:
        print("INFO: use only face_encodings")
        face_location = face_locations[0]
    else:
        print('No face detected')
        return

    # padding
    face_height = face_location[2] - face_location[0]
    face_weight = face_location[1] - face_location[3]
    padding_x = int(face_weight * 0.1)
    padding_y = int(face_height * 0.1)
    face = frame[max(0, face_location[0] - padding_y): min(frame.shape[0], face_location[2] + padding_y),
                 max(0, face_location[3] - padding_x): min(frame.shape[1], face_location[1] + padding_x)]

    '''
    # show image
    plt.imshow(face)
    plt.show()
    '''

    # predict
    blob = cv2.dnn.blobFromImage(
        face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=True)
    age_net.setInput(blob)
    predictions = age_net.forward()
    age = age_list[predictions[0].argmax()]
    confidence = predictions[0].max().item()

    return {'type': age,
            'confidence': confidence,
            'position_top': face_location[0],
            'position_right': face_location[1],
            'position_bottom': face_location[2],
            'position_left': face_location[3],
            'time': int(time.time())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    args = parser.parse_args()
    if args.image:
        print(predict(args.image))


if __name__ == '__main__':
    main()
