import argparse, time, os, torch
from config import detect_cfg
from utils.utils import prepare_experiment, handle_keyboard_interruption, handle_other_exception
from model.model import YOLOv2Model
os.environ["CUDA_VISIBLE_DEVICES"] = detect_cfg.CUDA_ID


def main():
    try:
        cfg = prepare_experiment(detect_cfg, 'e')
        model = YOLOv2Model(cfg, training=False)
        model.detect(detect_cfg.IMG_PATH)
    except KeyboardInterrupt:
        handle_keyboard_interruption(cfg)
    except:
        handle_other_exception(cfg)


if __name__ == '__main__':
    main()