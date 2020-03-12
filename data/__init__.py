from torch.utils.data import DataLoader
from .dataset import YOLOv2Dataset

def make_dataloader(cfg, training):
    eval_dataset = YOLOv2Dataset(cfg.DATA, cfg.MODEL_CFG_FNAME, training=False)
    eval_dataloader =  DataLoader(
        eval_dataset,
        batch_size=eval_dataset.batch_size,
        shuffle=False,
        num_workers=eval_dataset.n_cpu,
        collate_fn=eval_dataset.collate_fn
    )

    if not training:
        return eval_dataloader

    train_dataset = YOLOv2Dataset(cfg.DATA, cfg.MODEL_CFG_FNAME, training=True)
    train_dataloader =  DataLoader(
        train_dataset,
        batch_size=train_dataset.batch_size,
        shuffle=True,
        num_workers=train_dataset.n_cpu,
        collate_fn=train_dataset.collate_fn
    )
    return train_dataloader, eval_dataloader