
# Social-Distancing-Detection


### Classifying people as high risk and low risk based on their distance to other people


The original [YOLOv5](https://github.com/ultralytics/yolov5) model was used to detect people. Thereafter the people were classified as "*High Risk*" and "*Low Risk*" based on their distance from one another.

## How to Run
Step-1. Create a new conda environment using 
```bash
$ conda create -n <environment name> python=3.7 -y
```
Step-2. Activate the environment
```bash
$ conda activate <environment name>
```
Step-3 Clone / Download my repository
```bash
$ git clone https://github.com/Sumegh20/Social-Distancing-Detection.git 
```
OR 

Download the repository then unzip it.

Step-4 install Requirements

```bash
$ pip install -r requirements.txt
OR
$ pip install -U -r requirements.txt
```

Step-5 Run on a single video file
```bash
$ python app.py --source <path to the video file>
```

## Options

*Add using '--'*

```
weights='yolov5s.pt' imgsz=[H, W] max-det = <Number> device=<cpu/gpu> show-vid
```

---


## Output

![](Top_Angle.gif)

