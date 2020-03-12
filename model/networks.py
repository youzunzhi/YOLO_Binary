import os
import numpy as np
import torch
import torch.nn as nn
from model.modules import ReorgLayer, RegionLayer, MaxPoolStride1


class YOLOv2Network(nn.Module):
    def __init__(self, model_cfg_fname, weights_fname, use_cuda):
        super(YOLOv2Network, self).__init__()
        self.use_cuda = use_cuda
        self.hyper_parameters, self.module_list = self.get_module_list(model_cfg_fname)
        if not os.path.exists(weights_fname):
            print(f"{weights_fname} not exists")
            self.seen = 0
            self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        else:
            self.load_weights(weights_fname)

    def forward(self, x, targets=None, img_paths=''):
        layer_outputs = []
        output = None
        for i, (layer_def, layer) in enumerate(zip(self.layer_defs, self.module_list)):
            if layer_def['type'] in ['convolutional', 'maxpool', 'reorg']:
                x = layer(x)
            elif layer_def['type'] == 'route':
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in layer_def["layers"].split(",")], 1)
            elif layer_def['type'] == 'region':
                output = layer(x, targets, seen=self.seen, img_paths=img_paths)
                self.seen += x.shape[0]
            layer_outputs.append(x)
        return output

    def load_weights(self, weights_fname):
        with open(weights_fname, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=4)  # First four are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet19_448.conv.23" in weights_fname:
            self.seen = 0
            cutoff = 23

        ptr = 0
        for i, (layer_def, layer) in enumerate(zip(self.layer_defs, self.module_list)):
            if i == cutoff:
                break
            if layer_def['type'] == 'convolutional':
                conv_layer = layer[0]
                bn = int(layer_def["batch_normalize"])
                if bn:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = layer[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_weights(self, weights_fname):
        fp = open(weights_fname, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layersK
        for i, module in enumerate(self.module_list):
            if isinstance(module, nn.Conv2d):
                conv_layer = module
                if not isinstance(self.module_list[i + 1], nn.BatchNorm2d):
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                    conv_layer.weight.data.cpu().numpy().tofile(fp)

            elif isinstance(module, nn.BatchNorm2d):
                bn_layer = module
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

    def get_module_list(self, model_cfg_path):
        self.layer_defs = self.parse_model_cfg(model_cfg_path)
        hyper_parameters = self.layer_defs.pop(0)
        module_list = nn.ModuleList()
        filters = int(hyper_parameters["channels"])
        filter_num_list = [filters]
        for layer_i, layer_def in enumerate(self.layer_defs):
            if layer_def["type"] == "convolutional":
                layer = nn.Sequential()
                is_batch_normalize = int(layer_def["batch_normalize"])
                kernel_size = int(layer_def["size"])
                stride = int(layer_def["stride"])
                # pad = int(layer_def["pad"])
                pad = (kernel_size - 1) // 2
                filters = int(layer_def["filters"])
                activation_type = layer_def["activation"]
                layer.add_module(
                    f"conv_{layer_i}",
                    nn.Conv2d(
                        in_channels=filter_num_list[-1],
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=pad,
                        bias=not is_batch_normalize,
                    ),
                )
                if is_batch_normalize:
                    layer.add_module(
                        f"batchnorm_{layer_i}",
                        nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5)
                    )
                if layer_def["activation"] == "leaky":
                    layer.add_module(
                        f"leaky_{layer_i}",
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                elif layer_def["activation"] != "linear":
                    assert False, "unknown activation type"
            elif layer_def["type"] == "maxpool":
                pool_size = int(layer_def['size'])
                stride = int(layer_def['stride'])
                if stride > 1:
                    layer = nn.MaxPool2d(pool_size, stride)
                elif stride == 1:
                    layer = MaxPoolStride1(pool_size)
                else:
                    raise NotImplementedError
            elif layer_def["type"] == "route":
                layer_idx = [int(x) for x in layer_def["layers"].split(",")]
                filters = sum([filter_num_list[1:][i] for i in layer_idx])
                layer = None
            elif layer_def["type"] == "reorg":
                stride = int(layer_def['stride'])
                filters = stride * stride * filter_num_list[-1]
                layer = ReorgLayer(stride)
            elif layer_def["type"] == "region":
                layer = RegionLayer(layer_def, use_cuda=self.use_cuda)
            else:
                assert False, 'unknown type %s' % (layer_def['type'])
            if self.use_cuda and layer is not None:
                layer = layer.cuda()
            module_list.append(layer)
            filter_num_list.append(filters)
        return hyper_parameters, module_list

    def parse_model_cfg(self, model_cfg_path):
        """
        Parses the yolov2 layer configuration file and returns module definitions(list of dicts)
        """
        model_cfg_file = open(model_cfg_path, 'r')
        lines = model_cfg_file.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]  # get rid of comments
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        layer_defs = []
        for line in lines:
            if line.startswith('['):  # This marks the start of a new block
                layer_defs.append({})
                layer_defs[-1]['type'] = line[1:-1].rstrip()
                if layer_defs[-1]['type'] == 'convolutional':
                    layer_defs[-1]['batch_normalize'] = 0
            else:
                key, value = line.split("=")
                value = value.strip()
                layer_defs[-1][key.rstrip()] = value.strip()

        return layer_defs
