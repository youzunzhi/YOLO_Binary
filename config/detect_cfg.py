from yacs.config import CfgNode as CN
import torch
_C = CN()
_C.OUTPUT_DIR = 'runs/'
_C.EXPERIMENT_NAME = 'default'
_C.CUDA_ID = '0'
_C.MODEL_CFG_FNAME = "pjreddie_files/yolov2.cfg"
_C.WEIGHTS_FNAME = "weights/yolov2.weights"
_C.IMG_PATH = "data/samples/dog.jpg"
if torch.cuda.is_available():
    _C.DATA_CFG_FNAME = "pjreddie_files/coco.data"
else:
    _C.DATA_CFG_FNAME = "pjreddie_files/coco-mac.data"

_C.IMG_SIZE = 608
_C.CONF_THRESH = 0.2
_C.NMS_THRESH = 0.45
