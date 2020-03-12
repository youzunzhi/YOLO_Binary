import argparse, time, os
from config import eval_cfg
from utils.utils import prepare_experiment, handle_keyboard_interruption, handle_other_exception
from model.model import YOLOv2Model
from data import make_dataloader
os.environ["CUDA_VISIBLE_DEVICES"] = eval_cfg.CUDA_ID


def main():
    try:
        cfg = prepare_experiment(eval_cfg, 'e')
        model = YOLOv2Model(cfg, training=False)
        eval_dataloader = make_dataloader(cfg, training=False)
        model.eval(eval_dataloader)
    except KeyboardInterrupt:
        handle_keyboard_interruption(cfg)
    except:
        handle_other_exception(cfg)


if __name__ == '__main__':
    main()