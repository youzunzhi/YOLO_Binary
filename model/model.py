from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
import os, time, logging

from model.networks import YOLOv2Network
from utils.computation import *
from utils.utils import parse_data_cfg, draw_detect_box, log_train_progress, show_eval_result
from data.dataset import get_imgs_size
torch.manual_seed(0)


class YOLOv2Model(object):
    def __init__(self, cfg, training=False):
        self.cfg = cfg
        self.training = training
        self.use_cuda = torch.cuda.is_available()
        self.network = YOLOv2Network(cfg.MODEL_CFG_FNAME, cfg.WEIGHTS_FNAME, self.use_cuda)
        if training:
            self.save_weights_fname_prefix = os.path.join(self.cfg.OUTPUT_DIR,
                                                          cfg.MODEL_CFG_FNAME.split('/')[-1].split('.')[0])
            self.seen = 0
            self.learning_rate = cfg.TRAIN.LEARNING_RATE
            self.optimizer = optim.SGD(self.network.parameters(),
                                       lr=self.learning_rate,
                                       momentum=0.9,
                                       weight_decay=0.0005)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [150, 300], 0.1)

    def detect(self, img_path):
        self.network.eval()

        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.cfg.IMG_SIZE, self.cfg.IMG_SIZE))
        img = transforms.ToTensor()(img)
        img = torch.stack([img])

        with torch.no_grad():
            output = self.network(img)
        predictions = non_max_suppression(output, self.cfg.CONF_THRESH, self.cfg.NMS_THRESH)
        draw_detect_box(img_path, predictions[0], parse_data_cfg(self.cfg.DATA_CFG_FNAME)['names'])

    def eval(self, eval_dataloader):
        self.network.eval()
        metrics = []
        labels = []
        for batch_i, (imgs, targets, imgs_path) in enumerate(tqdm.tqdm(eval_dataloader, desc="Detecting objects")):
            labels += targets[:, 1].tolist()
            if self.use_cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                outputs = self.network(imgs)
            predictions = non_max_suppression(outputs, self.cfg.CONF_THRESH, self.cfg.NMS_THRESH)
            metrics += get_batch_metrics(predictions, targets)
            # if batch_i > 1:
            #     break
        show_eval_result(metrics, labels)

    def train(self, train_dataloader, eval_dataloader):
        total_epochs = self.cfg.TRAIN.TOTAL_EPOCHS
        self.network.train()

        for epoch in range(1, total_epochs+1):
            start_time = time.time()
            for batch_i, (imgs, targets, img_paths) in enumerate(train_dataloader):
                if self.use_cuda:
                    imgs = imgs.cuda()
                    targets = targets.cuda().detach()

                loss = self.network(imgs, targets, img_paths)

                self.optimizer.zero_grad()
                loss.backward()
                for p in list(self.network.parameters()):
                    if hasattr(p, 'org'):
                        p.data.copy_(p.org)
                self.optimizer.step()
                for p in list(self.network.parameters()):
                    if hasattr(p, 'org'):
                        p.org.copy_(p.data.clamp_(-1, 1))
                log_train_progress(epoch, total_epochs, batch_i, len(train_dataloader), self.learning_rate, start_time,
                                   self.network.module_list[-1].metrics)

            self.scheduler.step()
            if epoch % self.cfg.SAVE_INTERVAL == 0 or epoch == 1:
                epoch_save_weights_fname = f'{self.save_weights_fname_prefix}-{epoch}.weights'
                self.network.save_weights(epoch_save_weights_fname)
            if epoch % self.cfg.EVAL_INTERVAL == 0 or epoch == 1:
                self.eval(eval_dataloader)

        if total_epochs % self.cfg.SAVE_INTERVAL != 0:
            epoch_save_weights_fname = self.save_weights_fname_prefix + str(total_epochs) + '.weights'
            self.network.save_weights(epoch_save_weights_fname)
        if total_epochs % self.cfg.EVAL_INTERVAL != 0:
            self.eval(eval_dataloader)