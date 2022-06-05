import argparse
import torch
from social_distance_track import SocialDistanceTracker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='yolov5/data/coco.yaml', help="coco.yaml")
    parser.add_argument('--source', type=str, default='dataset/Top_Angle.mp4', help='source')  # file/folder, 0 for webcam  dataset/San_Francisco.mp4
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_false', help='save video tracking results')
    parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')

    args = vars(parser.parse_args())

    trakerObject = SocialDistanceTracker(args)
    with torch.no_grad():
        trakerObject.run_tracker()
