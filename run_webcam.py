#!/usr/bin/python3
import argparse
import logging
import time
import mvnc.mvncapi as mvnc

import cv2
import numpy as np

from picamera.array import PiRGBArray
from picamera import PiCamera

from tf_pose.estimator import TfPoseEstimator
from tf_pose.common import CocoPairs, CocoPart
from tf_pose.networks import get_graph_path, model_wh

#from imutils.video import WebcamVideoStream
#from imutils.video import VideoStream

from time import sleep

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.ERROR)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print("No devices found")
    quit()

device = mvnc.Device(devices[0])
device.openDevice()

"""
class VideoStream:
    def __init__(self, src=0, usePiCamera=False,
                 resolution=(320, 240), framerate=32):
        if usePiCamera:
            from imutils.video.pivideostream import PiVideoStream
            self.stream = PiVideoStream(
                resolution=resolution, framerate=framerate
            )
        else:
            self.stream = WebcamVideoStream(src=src)
            
    def start(self):
        return self.stream.start()
    
    def update(self):
        return self.stream.update()
    
    def read(self):
        return self.stream.read()
    
    def stop(self):
        return self.stream.stop()
"""

fps_time = 0

lines = []
mouse_lbutton_is_down = False
aux_line = []
last_mouse_pos = []

def parts_distance(part1, part2):
    return np.sqrt((part1.x - part2.x) ** 2 + (part1.y - part2.y) ** 2 )


def parts_banister_distance(part, banister):
    p1 = np.array(banister[:2])
    p2 = np.array(banister[2:])
    p3 = np.array([part.x, part.y]) if hasattr(part, 'x') else np.array(part)
    norm = np.linalg.norm
    dist = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    print(dist)
    return dist


def on_mouse(event, x, y, flags, params):

    global aux_line
    global lines
    global last_mouse_pos

    if event == cv2.EVENT_LBUTTONDOWN:
        aux_line = [x, y]
        mouse_lbutton_is_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        lines.append(aux_line + [x, y])
        aux_line = []
    elif event == cv2.EVENT_MOUSEMOVE:
        last_mouse_pos = [x, y]


class HumanIdentifier():
    def __init__(self, distance_modifier=21.):
        self.humans = dict()
        self.distance_modifier = distance_modifier
        self.perspective_coeficients = []

    def process_scene(self, humans):
        perspective_coeficients = [0.]*len(humans)
        mean_distances_from_last_registries = [16000.]*len(humans)
        for idx, human in enumerate(humans):
        
            distances = []
            for pair_idx1, pair_idx2 in [(p1, p2) for p1, p2 in CocoPairs if p1 in human.body_parts.keys() and p2 in human.body_parts.keys()]:
                body_part_1 = human.body_parts[pair_idx1]
                body_part_2 = human.body_parts[pair_idx2]
                distances += [parts_distance(body_part_1, body_part_2)]
            
            perspective_coeficients[idx] = np.mean(distances)
            
            distances_from_last_registries = dict(zip(self.humans.keys(), [None]*len(self.humans)))
            for human_registry_idx in self.humans.keys():
                for part in CocoPart:
                    if part.value in human.body_parts.keys() and part.value in self.humans[human_registry_idx].body_parts.keys():
                        distances_from_last_registries[human_registry_idx] = (distances_from_last_registries[human_registry_idx] or []) + [
                            parts_distance(self.humans[human_registry_idx].body_parts[part.value], human.body_parts[part.value])
                        ]
            
            mean_distances_from_last_registries[idx] = {idx_: np.mean(val) for idx_, val in distances_from_last_registries.items() if val is not None}

        index_mappings = dict()
        for human_registry_idx in self.humans.keys():
            distances = []
            found = False
            possible_humans = [(idx, human) for idx, human in enumerate(humans) if human_registry_idx in list(mean_distances_from_last_registries[idx].keys())]
            for idx, human in possible_humans:
                distances += [(idx, mean_distances_from_last_registries[idx][human_registry_idx])]
            
            for idx, mean_distance in sorted(distances, key=lambda x: x[1]):
                if mean_distance < (perspective_coeficients[idx]*self.distance_modifier) ** 2:
                    index_mappings.update({idx: human_registry_idx})
                    found = True
                    break
                break
        
        if len(index_mappings) < len(humans):
            for idx in [i for i in range(len(humans)) if i not in index_mappings.keys()]:
                for i in range(len(humans)):
                    if i not in index_mappings.values():
                        index_mappings.update({idx: i})

        self.humans = {idx: humans[i] for i, idx in index_mappings.items()}
        self.perspective_coeficients = {idx: perspective_coeficients[i] for i, idx in index_mappings.items()}
        return self.humans

    def is_handling_banister(self, estimator, image, idx):
        
        for line in lines:
            for wrist, position in estimator.get_human_wrists(image, self.humans[idx]).items():
                print("< %.3f" % (self.perspective_coeficients[idx]*self.distance_modifier ** 2,))
                if position is not None and parts_banister_distance(position, line) < (self.perspective_coeficients[idx]*self.distance_modifier) ** 2:
                    return True
        
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--use-picamera', help='Use a Raspberry Pi camera instead of webcam',
                    action='store_true')
    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--input-resolution', type=str, default='320x240',
                        help='only for picamera. desired input resolution. default=320x240')
    parser.add_argument('--input-framerate', type=int, default=32,
                        help='only for picamera. desired input framerate. default=32')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--is-banisters', action="store_true")
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()
    
    logger.debug('cam read+')
    camera = None
    if args.use_picamera:
        size = model_wh(args.input_resolution)
        camera = PiCamera()
        camera.resolution = size
        camera.framerate = args.input_framerate
        rawCapture = PiRGBArray(camera,size=size)
        rawCapture.truncate(0)
        #time.sleep(0.1)
        #cam = camera.capture_continuous(
        #    rawCapture, format="bgr", use_video_port=True
        #)
        #image = next(cam).array
        camera.capture(rawCapture, 'bgr')
        image = rawCapture.array
    else:
        cam = cv2.VideoCapture(args.camera)
        _, image = cam.read()
    
    print("Camera: %d (%s)" % (args.camera, str(type(args.camera))))
    print("Use PI Camera: %d (%s)" % (args.use_picamera, str(type(args.use_picamera))))
    print("Input Resolution: %s (%s)" % (str(model_wh(args.input_resolution)), str(type(model_wh(args.input_resolution)))))
    print("Input Framerate: %d (%s)" % (args.input_framerate, str(type(args.input_framerate))))
    
    """
    cam = VideoStream(src=args.camera, usePiCamera=args.use_picamera,
                      resolution=model_wh(args.input_resolution),
                      framerate=args.input_framerate).start()
    
    trials = 0
    while(trials < 5):
        ret_val = cam.read()
        ret_val, image = ret_val if isinstance(ret_val, tuple) else (None, ret_val)
        if image is not None:
            print('Got it!')
            break
        sleep(0.05)
        trials += 1
    """
    
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        # e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
        with open(get_graph_path(args.model), 'rb') as fp:
            blob = fp.read()
        graph = device.AllocateGraph(blob)
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    HId = HumanIdentifier()
    humans = dict()

    while True:
        cv2.namedWindow('tf-pose-estimation result')
        cv2.setMouseCallback('tf-pose-estimation result', on_mouse)
    
        if args.use_picamera:
            rawCapture.truncate(0)
            camera.capture(rawCapture, 'bgr')
            image = rawCapture.array
        else:
            cam = cv2.VideoCapture(args.camera)
            _, image = cam.read()
        """
        ret_val = cam.read()
        ret_val, image = ret_val if isinstance(ret_val, tuple) else (None, ret_val)
        """
        
        logger.debug('image process+')
        entities = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        if args.is_banisters:
            image = TfPoseEstimator.draw_banisters(image, entities, imgcopy=False)
        else:
            humans = HId.process_scene(entities)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        to_add = np.zeros_like(image, dtype=np.uint8)
        tmp2 = np.amax(np.absolute(e.pafMat.transpose((2, 0, 1))), axis=0)
        tmp2 = cv2.resize(tmp2, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        
        to_add[:,:,0] = tmp2*255
        to_add[:,:,1] = tmp2*255
        to_add[:,:,2] = tmp2*255
        pafmat_img = cv2.applyColorMap(to_add, cv2.COLORMAP_JET)
        
        #tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        #tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
		
        #print(image.shape)
        #print(tmp2_odd.shape)
        #tmp2_odd = cv2.resize(tmp2_odd, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        # a = fig.add_subplot(2, 2, 3)
        # a.set_title('Vectormap-x')
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        # plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
        # plt.colorbar()
        dst = cv2.addWeighted(pafmat_img, 0.2, image, 0.8, 0)

        # a = fig.add_subplot(2, 2, 4)
        # a.set_title('Vectormap-y')
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        # plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
        # plt.colorbar()
        # plt.show()
        
        for l in lines:
            cv2.line(dst, tuple(l[:2]),tuple(l[2:]),(0,0,255),2)
        if len(aux_line) > 0:
            cv2.line(dst, tuple(aux_line), tuple(last_mouse_pos), (0,0,128), 2)

        logger.debug('show+')
        cv2.putText(dst,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        for idx, human in humans.items():
            cv2.putText(dst,
                        "Human #%d handled? %s" % (idx, "YES" if HId.is_handling_banister(e, dst, idx) else "NO"),
                        (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', dst)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
