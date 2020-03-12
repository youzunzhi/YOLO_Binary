import argparse
import os, sys, time, datetime, math, shutil
import logging
import numpy as np
import cv2, torch
from terminaltables import AsciiTable
from utils.computation import ap_per_class


def prepare_experiment(cfg, log_prefix):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("opts", help="Modify configs using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_list(args.opts)

    # each experiment's output is in the dir named after the time when it starts to run
    if torch.cuda.is_available():
        log_dir_name = log_prefix + '-[{}]'.format(
            (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%m%d%H%M%S'))
    else:
        log_dir_name = log_prefix + '-[{}]'.format((datetime.datetime.now()).strftime('%m%d%H%M%S'))
    log_dir_name += cfg.EXPERIMENT_NAME
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, log_dir_name)
    os.mkdir(cfg.OUTPUT_DIR)
    cfg.freeze()

    logger = setup_logger("YOLOv2", cfg.OUTPUT_DIR, 0)
    logger.info("Running with config:\n{}".format(cfg))
    return cfg


def handle_keyboard_interruption(cfg):
    assert cfg.OUTPUT_DIR != 'runs/'
    assert cfg.OUTPUT_DIR.find('[') != -1
    save = input('save the log files(%s)?[y|n]' % cfg.OUTPUT_DIR)
    if save == 'y':
        print('log files may be saved in', cfg.OUTPUT_DIR)
    elif save == 'n':
        shutil.rmtree(cfg.OUTPUT_DIR)
        print('log directory removed:', cfg.OUTPUT_DIR)
    else:
        print('unknown input, saved by default')


def handle_other_exception(cfg):
    import traceback
    print(traceback.format_exc())
    assert cfg.OUTPUT_DIR != 'runs/'
    assert cfg.OUTPUT_DIR.find('[') != -1
    print('log directory removed:', cfg.OUTPUT_DIR)
    shutil.rmtree(cfg.OUTPUT_DIR)


def setup_logger(name, log_path, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_path:
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        # ISOTIMEFORMAT = '%Y-%m-%d-%X'
        # txt_name = '{}.txt'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
        txt_name = 'log.txt'
        fh = logging.FileHandler(os.path.join(log_path, txt_name), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def load_class_names(names_file):
    class_names = []
    with open(names_file, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def parse_data_cfg(path):
    """Parses the data configuration file"""
    data_cfg = dict()
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        data_cfg[key.strip()] = value.strip()
    return data_cfg


def draw_detect_box(img_path, predictions, names_file):
    def get_color(channel, offset, class_num):
        colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
        ratio = float(offset) / class_num * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][channel] + ratio * colors[j][channel]
        return int(r * 255)

    class_names = load_class_names(names_file)

    def get_rgb(class_id):
        class_num = len(class_names)
        offset = class_id * 123457 % class_num
        red = get_color(2, offset, class_num)
        green = get_color(1, offset, class_num)
        blue = get_color(0, offset, class_num)
        rgb = (red, green, blue)
        return rgb

    img = cv2.imread(img_path)
    save_path = 'detect-' + img_path.split('/')[-1]
    for prediction in predictions:
        left_top_x, left_top_y, right_bottom_x, right_bottom_y, class_id = prediction[0], prediction[1], prediction[2], \
                                                                           prediction[3], prediction[-1]
        color_tuple = get_rgb(class_id)
        cv2.rectangle(img, (left_top_x, left_top_y), (right_bottom_x, right_bottom_y), color_tuple, 4)
        cv2.putText(img, class_names[int(class_id)], (left_top_x, left_top_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color_tuple)
    cv2.imwrite(save_path, img)
    print('save detection to', save_path)


def log_train_progress(epoch, total_epochs, batch_i, total_batch, lr, start_time, metrics):
    log_str = "\n---- [Epoch %d/%d, Batch %d/%d, LR " % (epoch, total_epochs, batch_i, total_batch) + str(
        lr) + "] ----\n"
    metric_table = [["Metrics", "Region Layer"]]
    formats = {m: "%.6f" for m in metrics}
    formats["grid_size"] = "%2d"
    formats["cls_acc"] = "%.2f%%"
    for i, metric in enumerate(metrics):
        row_metrics = formats[metric] % metrics.get(metric, 0)
        metric_table += [[metric, row_metrics]]
    log_str += AsciiTable(metric_table).table

    # Determine approximate time left for epoch
    epoch_batches_left = total_batch - (batch_i + 1)
    time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
    log_str += f"\n---- ETA {time_left}"

    logger = logging.getLogger('YOLOv2.Train')
    logger.info(log_str)


def show_eval_result(metrics, labels):
    true_positives, pred_conf, pred_labels = [np.concatenate(x, 0) for x in list(zip(*metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_conf, pred_labels, labels)
    logger = logging.getLogger('YOLOv2.Eval')
    logger.info(f"mAP: {AP.mean()}")
