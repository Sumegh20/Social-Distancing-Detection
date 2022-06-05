import os
from pathlib import Path
import shutil
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import check_file, check_img_size, cv2, non_max_suppression, scale_coords
from yolov5.utils.plots import Annotator
from yolov5.utils.torch_utils import select_device
from custom_utils import getHighRiskPeople

class SocialDistanceTracker:
    def __init__(self, args):
        self.source = args['source']
        self.weights = args['weights']
        self.data = args['data']
        self.output = args['output']
        self.show_vid = args['show_vid']
        self.save_vid = args['save_vid']
        self.imgsz = args['imgsz']
        self.half = args['half']
        self.update = args['update']
        self.device = args['device']
        self.dnn = args['dnn']
        self.augment = args['augment']
        self.conf_thres = args['conf_thres']
        self.iou_thres = args['iou_thres']
        self.classes = args['classes']
        self.agnostic_nms = args['agnostic_nms']
        self.max_det = args['max_det']

    def run_tracker(self):
        source = str(self.source)

        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        if os.path.exists(self.output):  # output dir
            shutil.rmtree(self.output)  # delete dir
        os.makedirs(self.output)  # make new dir

        # Load model
        device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size

        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im.unsqueeze(0)  # expand for batch dim

            # Inference
            pred = model(im, augment=self.augment)

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det)

            # List to store bounding coordinates of people
            people_coordinates = []

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                if webcam:
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    p = Path(p)  # to Path
                    s += f'{i}: '
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path

                save_path = self.output + '/' + str(p.name)

                annotator = Annotator(im0, line_width=2)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]} {'s' * (n > 1)}"  # add to string
                        print(s)

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if names[int(cls)] == 'person':
                            # List to store bounding coordinates of people
                            people_coordinates.append(xyxy)

                # distance
                HighRiskPeople, LowRiskPeople = getHighRiskPeople(people_coordinates, 45)

                for bboxes in HighRiskPeople:
                    label = f'High Risk'
                    colors = (0, 0, 255)
                    annotator.box_label(bboxes, label, color=colors)

                for bboxes in LowRiskPeople:
                    label = f'Low Risk'
                    colors = (0, 255, 0)
                    annotator.box_label(bboxes, label, color=colors)

            # Stream results
            im0 = annotator.result()
            if self.show_vid:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Save results (image with detections)
            if self.save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

        if self.save_vid:
            print('Results saved to %s' % Path(save_path))
