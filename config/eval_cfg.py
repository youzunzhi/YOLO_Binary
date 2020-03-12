from yacs.config import CfgNode as CN
import torch
_C = CN()
_C.OUTPUT_DIR = 'runs/'
_C.EXPERIMENT_NAME = 'default'
_C.CUDA_ID = '0'
eval_on = 'voc'
if eval_on == 'coco':
    _C.MODEL_CFG_FNAME = "pjreddie_files/yolov2.cfg"
    _C.WEIGHTS_FNAME = "weights/yolov2.weights"
elif eval_on == 'voc':
    # _C.MODEL_CFG_FNAME = "pjreddie_files/yolov2-voc.cfg"
    # _C.WEIGHTS_FNAME = "weights/yolov2-voc.weights"
    _C.MODEL_CFG_FNAME = "pjreddie_files/yolov2-tiny-voc.cfg"
    _C.WEIGHTS_FNAME = "weights/yolov2-tiny-voc.weights"
_C.CONF_THRESH = 0.005
_C.NMS_THRESH = 0.45

_C.DATA = CN()
if eval_on == 'coco':
    if torch.cuda.is_available():
        _C.DATA.DATA_CFG_FNAME = "pjreddie_files/coco.data"
    else:
        _C.DATA.DATA_CFG_FNAME = "pjreddie_files/coco-mac.data"

elif eval_on == 'voc':
    _C.DATA.DATA_CFG_FNAME = "pjreddie_files/voc.data"
_C.DATA.IMG_SIZE = 608
_C.DATA.BATCH_SIZE = 1
_C.DATA.MULTISCALE = False
_C.DATA.N_CPU = 0
