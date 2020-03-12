import argparse, time, os
from config import train_cfg
from utils.utils import prepare_experiment, handle_keyboard_interruption, handle_other_exception
from model.model import YOLOv2Model
from data import make_dataloader
os.environ["CUDA_VISIBLE_DEVICES"] = train_cfg.CUDA_ID


def main():
    try:
        cfg = prepare_experiment(train_cfg, 't')
        model = YOLOv2Model(cfg, training=True)
        train_dataloader, eval_dataloader = make_dataloader(cfg, training=True)
        model.train(train_dataloader, eval_dataloader)
    except KeyboardInterrupt:
        handle_keyboard_interruption(cfg)
    except:
        handle_other_exception(cfg)


if __name__ == '__main__':
    main()