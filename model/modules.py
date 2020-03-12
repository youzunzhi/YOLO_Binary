import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.computation import bbox_iou, bbox_wh_iou


class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, [0, 1, 0, 1], mode='replicate'), self.kernel_size, stride=1)
        return x


# class EmptyLayer(nn.Module):
#     """Placeholder for 'route' and 'shortcut' layers"""
#
#     def __init__(self):
#         super(EmptyLayer, self).__init__()

class ReorgLayer(nn.Module):
    def __init__(self, stride):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H // hs, W // ws)
        return x


class RegionLayer(nn.Module):
    def __init__(self, module_def, use_cuda=False):
        super(RegionLayer, self).__init__()
        self.num_anchors = int(module_def['num'])
        self.num_classes = int(module_def['classes'])
        self.use_cuda = use_cuda
        # get anchors
        anchors = module_def['anchors'].split(',')
        self.anchors = [float(i) for i in anchors]
        self.anchors = [(self.anchors[i], self.anchors[i + 1]) for i in
                        range(0, len(anchors), 2)]  # list of tuple (anchor_w, anchor_h)
        self.anchors = torch.FloatTensor(self.anchors)
        if use_cuda:
            self.anchors = self.anchors.cuda()

        self.object_scale = float(module_def['object_scale'])
        self.noobject_scale = float(module_def['noobject_scale'])
        self.class_scale = float(module_def['class_scale'])
        self.coord_scale = float(module_def['coord_scale'])
        self.thresh = float(module_def['thresh'])
        self.rescore = int(module_def['rescore'])

    def forward(self, x, targets, seen=0, img_paths=''):
        """
        :param x: the output of last layer. Shape: (B, A*(C+5), H, W).
        :param targets: the target bboxes. Each row has 6 num: sample_index, cls, x, y, w, h (0~1). shape: (num_target_box, 6)
        :param seen:
        :return:
        """
        num_samples = x.size(0)     # B == num_sample
        grid_size = x.size(2)       # H == W == grid_size

        torch_byte_tensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
        torch_float_tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        # make x to a (B, A, H, W, (C+5)) shaped x_permute
        x_permute = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )

        # the first 4 float of the last dimension of x_permute are `tx, ty, tw, th`
        pred_tx, pred_ty, pred_tw, pred_th = x_permute[..., 0], x_permute[..., 1], x_permute[..., 2], x_permute[..., 3]
        # the 5th float of the last dimension of x_permute are confidence of having object in this pred_box
        pred_conf = torch.sigmoid(x_permute[..., 4])  # use sigmoid to make it 0~1
        # the rest of the last dimension of x_permute are P(cls|object)
        pred_cls = torch.sigmoid(x_permute[..., 5:])  # use sigmoid to make it 0~1
        # make a (B, A, H, W, 4) shaped tensor to contain 4 coords of the pred_box, related to grid size
        pred_coords_grid_size = torch_float_tensor(num_samples, self.num_anchors, grid_size, grid_size, 4)
        pred_coords_grid_size[..., 0] = torch.sigmoid(pred_tx) + torch.arange(grid_size).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size]).float().cuda() if self.use_cuda else torch.sigmoid(pred_tx) + torch.arange(grid_size).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size]).float()
        pred_coords_grid_size[..., 1] = torch.sigmoid(pred_ty) + torch.arange(grid_size).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size]).float().cuda() if self.use_cuda else torch.sigmoid(pred_ty) + torch.arange(grid_size).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size]).float()
        pred_coords_grid_size[..., 2] = torch.exp(pred_tw) * self.anchors[:, 0].view(1, self.num_anchors, 1, 1)
        pred_coords_grid_size[..., 3] = torch.exp(pred_th) * self.anchors[:, 1].view(1, self.num_anchors, 1, 1)

        # if it's testing, just return the prediction
        if targets is None:
            predictions = torch.cat(
                (
                    pred_coords_grid_size.view(num_samples, -1, 4) / grid_size,
                    pred_conf.view(num_samples, -1, 1),
                    pred_cls.view(num_samples, -1, self.num_classes),
                ),
                -1,
            )  # B, (H*W*A), (C+5)
            return predictions

        # -------------- training -------------- #
        # ---- preparation ---- #

        # get 4 coords of target boxes, related to grid_size. Shape: (num_num_target_box, 4).
        target_coords_grid_size = targets[:, 2:6] * grid_size
        # get each target_box's corresponding anchor_index, sample_index, grid_index_x, grid_index_y. They are all (num_target_box,) shaped.
        _, anchor_index = torch.stack([bbox_wh_iou(anchor, target_coords_grid_size[:, 2:]) for anchor in self.anchors]).max(0)
        grid_index_x, grid_index_y = target_coords_grid_size[:, 0].long(), target_coords_grid_size[:, 1].long()
        sample_index = targets[:, 0].long()
        # while multi_target_to_one_anchor:
        #     change the second correct pred_box to the second similar anchor

        # make a mask of all target-corresponded pred_box. Note that x[target_mask] is not x[sample_index, anchor_index, grid_index_y, grid_index_x] (the order is different and maybe the number as well)
        target_mask = torch_byte_tensor(num_samples, self.num_anchors, grid_size, grid_size).fill_(0).bool()
        target_mask[sample_index, anchor_index, grid_index_y, grid_index_x] = True

        # 这个情况太难了，先算了，但是绝对应该要解决。
        # assert target_mask.sum() == len(target_coords_grid_size), img_paths    # There's a chance that multiple target_boxes corresponds to the same anchor box. I don't know how to deal with that yet.
        # while target_mask.sum() != len(target_coords_grid_size):   # There's a chance that multiple target_boxes corresponds to the same anchor box. I don't know how to deal with that yet.
        #     conflicted_index_list = []
        #     index_combination = torch.stack([sample_index, anchor_index, grid_index_y, grid_index_x]).t()
        #
        #     for i in range(index_combination.shape[0]-1):
        #         for j in range(i+1, index_combination.shape[0]):
        #             if (index_combination[i] == index_combination[j]).sum() == 4:
        #                 appended_flag = False
        #                 for k in range(len(conflicted_index_list)):
        #                     if (index_combination[j] == index_combination[conflicted_index_list[k][0]]).sum() == 4:
        #                         conflicted_index_list[k].append(j)
        #                         assert len(conflicted_index_list[k]) <= self.num_anchors,
        #                         appended_flag = True
        #                 if not appended_flag:
        #                     conflicted_index_list.append([i, j])
        #                 # anchor_index[i] = torch.stack([bbox_wh_iou(anchor, target_coords_grid_size[:, 2:]) for anchor in self.anchors]).argsort(0)[-2][i]


        # compute the IoU between each target_box and its corresponding pred_box
        iou_target_pred = torch_float_tensor(num_samples, self.num_anchors, grid_size, grid_size).fill_(0)
        iou_target_pred[sample_index, anchor_index, grid_index_y, grid_index_x] = bbox_iou(pred_coords_grid_size[sample_index, anchor_index, grid_index_y, grid_index_x], target_coords_grid_size, x1y1x2y2=False)

        # ---- classification loss ---- #
        target_cls = torch_float_tensor(num_samples, self.num_anchors, grid_size, grid_size, self.num_classes).fill_(0)
        target_cls_index = targets[:, 1].long()
        target_cls[sample_index, anchor_index, grid_index_y, grid_index_x, target_cls_index] = 1.
        loss_cls = nn.MSELoss()(pred_cls[target_mask], target_cls[target_mask]) * self.class_scale

        # ---- confidence loss ---- #
        target_conf = torch_float_tensor(num_samples, self.num_anchors, grid_size, grid_size).fill_(0)
        target_conf[target_mask] = 1.
        if self.rescore:
            target_conf *= iou_target_pred
        loss_obj_conf = nn.MSELoss()(pred_conf[target_mask], target_conf[target_mask]) * self.object_scale

        # ---- coordinates loss ---- #
        target_coords_for_loss = torch_float_tensor(num_samples, self.num_anchors, grid_size, grid_size, 4).fill_(0)
        target_coords_for_loss[sample_index, anchor_index, grid_index_y, grid_index_x, 0] = target_coords_grid_size[:, 0] - grid_index_x
        target_coords_for_loss[sample_index, anchor_index, grid_index_y, grid_index_x, 1] = target_coords_grid_size[:, 1] - grid_index_y
        target_coords_for_loss[sample_index, anchor_index, grid_index_y, grid_index_x, 2] = torch.log(target_coords_grid_size[:, 2] / self.anchors[anchor_index, 0])
        target_coords_for_loss[sample_index, anchor_index, grid_index_y, grid_index_x, 3] = torch.log(target_coords_grid_size[:, 3] / self.anchors[anchor_index, 1])

        pred_coords_for_loss = torch_float_tensor(num_samples, self.num_anchors, grid_size, grid_size, 4)
        pred_coords_for_loss[..., 0] = torch.sigmoid(pred_tx)
        pred_coords_for_loss[..., 1] = torch.sigmoid(pred_ty)
        pred_coords_for_loss[..., 2] = pred_tw
        pred_coords_for_loss[..., 3] = pred_th
        # the loss of coordinates is scaled upon the corresponded target_box's w and h.
        loss_coords_wh_scale = torch_float_tensor(num_samples, self.num_anchors, grid_size, grid_size)
        loss_coords_wh_scale[sample_index, anchor_index, grid_index_y, grid_index_x] = 2 - target_coords_grid_size[:, 2] * target_coords_grid_size[:, 3] / (grid_size**2)
        loss_coords = nn.MSELoss()(pred_coords_for_loss[target_mask] * loss_coords_wh_scale[target_mask].unsqueeze(-1) ** 0.5, target_coords_for_loss[target_mask] * loss_coords_wh_scale[target_mask].unsqueeze(-1) ** 0.5) * self.coord_scale # 1/n*s*(a-b)^2 == 1/n*(a*s^(0.5)-b*s^(0.5))^2

        # ---- noobject loss ---- #
        noobject_mask = ~target_mask
        # Set noobject_mask to zero where iou of pred_box and any target box exceeds threshold
        for target_coords_grid_size_i in target_coords_grid_size:
            target_coords_grid_size_i_for_iou = target_coords_grid_size_i.repeat(num_samples, self.num_anchors, grid_size, grid_size, 1)
            pred_coords_ious = bbox_iou(pred_coords_grid_size, target_coords_grid_size_i_for_iou, x1y1x2y2=False)
            noobject_mask[pred_coords_ious > self.thresh] = False
        loss_noobject = nn.MSELoss()(pred_conf[noobject_mask], torch_float_tensor(pred_conf[noobject_mask].shape).fill_(0)) * self.noobject_scale

        # ---- prior loss ---- #
        if seen < 12800:
            prior_coords_for_loss = torch_float_tensor(num_samples, self.num_anchors, grid_size, grid_size, 4).fill_(0)
            prior_coords_for_loss[..., 0] = 0.5
            prior_coords_for_loss[..., 1] = 0.5
            # prior_coords_for_loss[..., 2] == prior_coords_for_loss[..., 3] == 0
            loss_prior = nn.MSELoss()(pred_coords_for_loss[~target_mask], prior_coords_for_loss[~target_mask]) * 0.01
        else:
            loss_prior = 0.

        total_loss = loss_cls + loss_obj_conf + loss_coords + loss_noobject + loss_prior
        # Metrics
        correct_cls = pred_cls[sample_index, anchor_index, grid_index_y, grid_index_x].argmax(1) == target_cls_index
        cls_acc = 100 * correct_cls.float().mean()
        obj_conf = pred_conf[target_mask].mean()
        conf_noobj = pred_conf[noobject_mask].mean()
        conf_over_50 = pred_conf > 0.5
        correct_detect_over_50_iou = correct_cls & (iou_target_pred[sample_index, anchor_index, grid_index_y, grid_index_x] > 0.5) & (pred_conf[sample_index, anchor_index, grid_index_y, grid_index_x] > 0.5)
        correct_detect_over_75_iou = correct_cls & (iou_target_pred[sample_index, anchor_index, grid_index_y, grid_index_x] > 0.75) & (pred_conf[sample_index, anchor_index, grid_index_y, grid_index_x] > 0.5)
        precision = torch.sum(correct_detect_over_50_iou) / (conf_over_50.sum() + 1e-16)
        recall50 = torch.sum(correct_detect_over_50_iou) / (target_mask.sum() + 1e-16)
        recall75 = torch.sum(correct_detect_over_75_iou) / (target_mask.sum() + 1e-16)

        self.metrics = {
            "loss": total_loss.item(),
            "loss_cls": loss_cls.item(),
            "loss_obj_conf": loss_obj_conf.item(),
            "loss_coords": loss_coords.item(),
            "loss_noobject": loss_noobject.item(),
            "cls_acc": cls_acc.item(),
            "obj_conf": obj_conf.item(),
            "obj_iou": iou_target_pred[target_mask].mean(),
            "noobj_conf": conf_noobj.item(),
            "recall50": recall50.item(),
            "recall75": recall75.item(),
            "precision": precision.item(),
            "grid_size": grid_size,
        }
        if seen < 12800:
            self.metrics['loss_prior'] = loss_prior.item()
        return total_loss


class RegionLayer_deprecated(nn.Module):
    def __init__(self, module_def):
        super(RegionLayer_deprecated, self).__init__()

        anchors = module_def['anchors'].split(',')
        self.anchors = [float(i) for i in anchors]
        self.anchors = [(self.anchors[i], self.anchors[i + 1]) for i in
                        range(0, len(anchors), 2)]  # list of tuple (anchor_w, anchor_h)
        self.num_classes = int(module_def['classes'])
        self.num_anchors = int(module_def['num'])

        self.object_scale = float(module_def['object_scale'])
        self.noobject_scale = float(module_def['noobject_scale'])
        self.class_scale = float(module_def['class_scale'])
        self.coord_scale = float(module_def['coord_scale'])
        self.thresh = float(module_def['thresh'])
        self.rescore = int(module_def['rescore'])

        self.metrics = {}

    def forward(self, x, targets, seen=0, use_cuda=False):

        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )  # B,A,H,W,25

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x # B,A,H,W
        y = torch.sigmoid(prediction[..., 1])  # Center y # B,A,H,W
        w = prediction[..., 2]  # Width # B,A,H,W
        h = prediction[..., 3]  # Height # B,A,H,W
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf # B,A,H,W
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred. # B,A,H,W,20

        # Add offset and scale with anchors
        pred_boxes = torch.FloatTensor(prediction[..., :4].shape)  # B,A,H,W,4
        g = grid_size
        grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(torch.FloatTensor)  # 1,1,H,W
        grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(torch.FloatTensor)  # 1,1,H,W
        anchor_w = torch.FloatTensor(self.anchors).index_select(1, torch.LongTensor([0])).view(1, self.num_anchors, 1,
                                                                                               1)
        anchor_h = torch.FloatTensor(self.anchors).index_select(1, torch.LongTensor([1])).view(1, self.num_anchors, 1,
                                                                                               1)
        if use_cuda:
            pred_boxes = pred_boxes.cuda()
            grid_x = grid_x.cuda()
            grid_y = grid_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) / grid_size,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )  # B,(H*W*A),25

        if targets is None:
            return output
        else:
            iou_scores, class_mask, coord_mask_scale, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                targets=targets,
                anchors=self.anchors,
                ignore_thresh=self.thresh,
                seen=seen,
                use_cuda=use_cuda
            )

            loss_x = nn.MSELoss()(x * coord_mask_scale ** 0.5, tx * coord_mask_scale ** 0.5)
            loss_y = nn.MSELoss()(y * coord_mask_scale ** 0.5, ty * coord_mask_scale ** 0.5)
            loss_w = nn.MSELoss()(w * coord_mask_scale ** 0.5, tw * coord_mask_scale ** 0.5)
            loss_h = nn.MSELoss()(h * coord_mask_scale ** 0.5, th * coord_mask_scale ** 0.5)
            loss_coord = loss_x + loss_y + loss_w + loss_h
            self.object_scale = noobj_mask.sum() / obj_mask.sum()
            loss_conf_obj = self.object_scale * nn.MSELoss()(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.noobject_scale * nn.MSELoss()(pred_conf[noobj_mask],
                                                                                tconf[noobj_mask])
            loss_cls = self.class_scale * nn.BCELoss()(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_coord + loss_conf_obj + loss_conf_noobj + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": total_loss.item(),
                "loss_coord": loss_coord.item(),
                "loss_conf_obj": loss_conf_obj.item(),
                "loss_conf_noobj": loss_conf_noobj.item(),
                "loss_cls": loss_cls.item(),
                "avg_iou": iou_scores[obj_mask].mean(),
                "conf_obj": conf_obj.item(),
                "conf_noobj": conf_noobj.item(),
                "cls_acc": cls_acc.item(),
                "recall50": recall50.item(),
                "recall75": recall75.item(),
                "precision": precision.item(),
                "grid_size": grid_size,
            }

            return total_loss

    def build_targets(self, pred_boxes, pred_cls, targets, anchors, ignore_thresh, seen, use_cuda):

        ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        nB = pred_boxes.size(0)
        nA = pred_boxes.size(1)
        nC = pred_cls.size(-1)
        nG = pred_boxes.size(2)  # grid_size
        anchors = FloatTensor(anchors)

        # Output tensors
        coord_mask_scale = FloatTensor(nB, nA, nG, nG).fill_(0)
        obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0).bool()
        noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1).bool()
        class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
        iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
        tx = FloatTensor(nB, nA, nG, nG).fill_(0)
        ty = FloatTensor(nB, nA, nG, nG).fill_(0)
        tw = FloatTensor(nB, nA, nG, nG).fill_(0)
        th = FloatTensor(nB, nA, nG, nG).fill_(0)
        tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

        # if iter < 12800, learn anchor box
        if seen < 12800:
            tx.fill_(0.5)
            ty.fill_(0.5)
            coord_mask_scale.fill_(1)

        # Convert to position relative to box
        target_boxes = targets[:, 2:6] * nG
        gxy = target_boxes[:, :2]
        gwh = target_boxes[:, 2:]
        # Get anchors with best iou
        ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
        best_ious, best_n = ious.max(0)
        # Separate target values
        b, target_labels = targets[:, :2].long().t()
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()
        # Set masks
        coord_mask_scale[b, best_n, gj, gi] = (2 - targets[:, 4] * targets[:, 5]) * self.coord_scale
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

        # Set noobj mask to zero where iou of pred_box and any target box exceeds ignore threshold
        for target_box in target_boxes:
            target_box_repeat = target_box.repeat(nB, nA, nG, nG, 1)
            pred_ious = bbox_iou(pred_boxes, target_box_repeat, x1y1x2y2=False)
            noobj_mask[pred_ious > ignore_thresh] = 0

        # Target Coordinates
        tx[b, best_n, gj, gi] = gx - gx.floor()
        ty[b, best_n, gj, gi] = gy - gy.floor()
        # Target Width and height
        tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0])
        th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1])
        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1
        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
        iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

        if self.rescore:
            tconf = obj_mask.float() * iou_scores  # rescore
        else:
            tconf = obj_mask.float()

        return iou_scores, class_mask, coord_mask_scale, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
