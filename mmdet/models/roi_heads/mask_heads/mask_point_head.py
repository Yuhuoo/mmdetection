# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend/point_head/point_head.py  # noqa

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import point_sample, rel_roi_point_to_rel_img_point
from mmcv.runner import BaseModule

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import (get_uncertain_point_coords_with_randomness,
                                get_uncertainty)
import ipdb
import numpy as np
from PIL import Image

@HEADS.register_module()
class MaskPointHead(BaseModule):
    """A mask point head use in PointRend.

    ``MaskPointHead`` use shared multi-layer perceptron (equivalent to
    nn.Conv1d) to predict the logit of input points. The fine-grained feature
    and coarse feature will be concatenate together for predication.

    Args:
        num_fcs (int): Number of fc layers in the head. Default: 3.
        in_channels (int): Number of input channels. Default: 256.
        fc_channels (int): Number of fc channels. Default: 256.
        num_classes (int): Number of classes for logits. Default: 80.
        class_agnostic (bool): Whether use class agnostic classification.
            If so, the output channels of logits will be 1. Default: False.
        coarse_pred_each_layer (bool): Whether concatenate coarse feature with
            the output of each fc layer. Default: True.
        conv_cfg (dict | None): Dictionary to construct and config conv layer.
            Default: dict(type='Conv1d'))
        norm_cfg (dict | None): Dictionary to construct and config norm layer.
            Default: None.
        loss_point (dict): Dictionary to construct and config loss layer of
            point head. Default: dict(type='CrossEntropyLoss', use_mask=True,
            loss_weight=1.0).
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 num_fcs=3,
                 in_channels=256,
                 fc_channels=256,
                 class_agnostic=False,
                 coarse_pred_each_layer=True,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 loss_point=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal', std=0.001,
                     override=dict(name='fc_logits'))):
        super().__init__(init_cfg)
        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.fc_channels = fc_channels
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.coarse_pred_each_layer = coarse_pred_each_layer
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_point = build_loss(loss_point)

        fc_in_channels = in_channels + num_classes
        self.fcs = nn.ModuleList()
        for _ in range(num_fcs):
            fc = ConvModule(
                fc_in_channels,
                fc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.fcs.append(fc)
            fc_in_channels = fc_channels
            fc_in_channels += num_classes if self.coarse_pred_each_layer else 0

        out_channels = 1 if self.class_agnostic else self.num_classes
        self.fc_logits = nn.Conv1d(
            fc_in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, fine_grained_feats, coarse_feats):
        """Classify each point base on fine grained and coarse feats.

        Args:
            fine_grained_feats (Tensor): Fine grained feature sampled from FPN,
                shape (num_rois, in_channels, num_points).
            coarse_feats (Tensor): Coarse feature sampled from CoarseMaskHead,
                shape (num_rois, num_classes, num_points).

        Returns:
            Tensor: Point classification results,
                shape (num_rois, num_class, num_points).
        """

        x = torch.cat([fine_grained_feats, coarse_feats], dim=1)
        for fc in self.fcs:
            x = fc(x)
            if self.coarse_pred_each_layer:
                x = torch.cat((x, coarse_feats), dim=1)
        return self.fc_logits(x)

    def get_targets(self, rois, rel_roi_points, sampling_results, gt_masks,
                    cfg):
        """Get training targets of MaskPointHead for all images.

        Args:
            rois (Tensor): Region of Interest, shape (num_rois, 5).
            rel_roi_points: Points coordinates relative to RoI, shape
                (num_rois, num_points, 2).
            sampling_results (:obj:`SamplingResult`): Sampling result after
                sampling and assignment.
            gt_masks (Tensor) : Ground truth segmentation masks of
                corresponding boxes, shape (num_rois, height, width).
            cfg (dict): Training cfg.

        Returns:
            Tensor: Point target, shape (num_rois, num_points).
        """

        num_imgs = len(sampling_results)
        rois_list = []
        rel_roi_points_list = []
        for batch_ind in range(num_imgs):
            inds = (rois[:, 0] == batch_ind)
            rois_list.append(rois[inds])
            rel_roi_points_list.append(rel_roi_points[inds])
        pos_assigned_gt_inds_list = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        cfg_list = [cfg for _ in range(num_imgs)]

        point_targets = map(self._get_target_single, rois_list,
                            rel_roi_points_list, pos_assigned_gt_inds_list,
                            gt_masks, cfg_list)
        point_targets = list(point_targets)

        if len(point_targets) > 0:
            point_targets = torch.cat(point_targets)

        return point_targets

    def _get_target_single(self, rois, rel_roi_points, pos_assigned_gt_inds,
                           gt_masks, cfg):
        """Get training target of MaskPointHead for each image."""
        num_pos = rois.size(0)
        num_points = cfg.num_points
        if num_pos > 0:
            gt_masks_th = (
                gt_masks.to_tensor(rois.dtype, rois.device).index_select(
                    0, pos_assigned_gt_inds))
            gt_masks_th = gt_masks_th.unsqueeze(1)
            rel_img_points = rel_roi_point_to_rel_img_point(
                rois, rel_roi_points, gt_masks_th)
            point_targets = point_sample(gt_masks_th,
                                         rel_img_points).squeeze(1)
        else:
            point_targets = rois.new_zeros((0, num_points))
        return point_targets

    def loss(self, point_pred, point_targets, labels):
        """Calculate loss for MaskPointHead.

        Args:
            point_pred (Tensor): Point predication result, shape
                (num_rois, num_classes, num_points).
            point_targets (Tensor): Point targets, shape (num_roi, num_points).
            labels (Tensor): Class label of corresponding boxes,
                shape (num_rois, )

        Returns:
            dict[str, Tensor]: a dictionary of point loss components
        """

        loss = dict()
        if self.class_agnostic:
            loss_point = self.loss_point(point_pred, point_targets,
                                         torch.zeros_like(labels))
        else:
            loss_point = self.loss_point(point_pred, point_targets, labels)
        loss['loss_point'] = loss_point
        return loss

    def get_roi_rel_points_train(self, mask_pred, labels, cfg):
        """Get ``num_points`` most uncertain points with random points during
        train.

        Sample points in [0, 1] x [0, 1] coordinate space based on their
        uncertainty. The uncertainties are calculated for each point using
        '_get_uncertainty()' function that takes point's logit prediction as
        input.

        Args:
            mask_pred (Tensor): A tensor of shape (num_rois, num_classes,
                mask_height, mask_width) for class-specific or class-agnostic
                prediction.
            labels (list): The ground truth class for each instance.
            cfg (dict): Training config of point head.

        Returns:
            point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
                that contains the coordinates sampled points.
        """
        point_coords = get_uncertain_point_coords_with_randomness(
            mask_pred, labels, cfg.num_points, cfg.oversample_ratio,
            cfg.importance_sample_ratio)
        return point_coords

    def get_roi_rel_points_test(self, mask_pred, pred_label, cfg):
        """Get ``num_points`` most uncertain points during test.

        Args:
            mask_pred (Tensor): A tensor of shape (num_rois, num_classes,
                mask_height, mask_width) for class-specific or class-agnostic
                prediction.
            pred_label (list): The predication class for each instance.
            cfg (dict): Testing config of point head.

        Returns:
            point_indices (Tensor): A tensor of shape (num_rois, num_points)
                that contains indices from [0, mask_height x mask_width) of the
                most uncertain points.
            point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
                that contains [0, 1] x [0, 1] normalized coordinates of the
                most uncertain points from the [mask_height, mask_width] grid .
        """
        num_points = cfg.subdivision_num_points
        # ipdb.set_trace()
        uncertainty_map = get_uncertainty(mask_pred, pred_label)    # (5, 1, 56, 56)
        # ipdb.set_trace()
        num_rois, _, mask_height, mask_width = uncertainty_map.shape
        # ipdb.set_trace()

        # During ONNX exporting, the type of each elements of 'shape' is
        # `Tensor(float)`, while it is `float` during PyTorch inference.
        if isinstance(mask_height, torch.Tensor):   # false
            h_step = 1.0 / mask_height.float()
            w_step = 1.0 / mask_width.float()
        else:
            # ipdb.set_trace()
            h_step = 1.0 / mask_height
            w_step = 1.0 / mask_width
        # ipdb.set_trace()

        # cast to int to avoid dynamic K for TopK op in ONNX
        mask_size = int(mask_height * mask_width)
        uncertainty_map = uncertainty_map.view(num_rois, mask_size)   # (5, 3136)
        # ipdb.set_trace()

        num_points = min(mask_size, num_points)
        point_indices = uncertainty_map.topk(num_points, dim=1)[1]  # (5, topk)
        # ipdb.set_trace()

        xs = w_step / 2.0 + (point_indices % mask_width).float() * w_step
        ys = h_step / 2.0 + (point_indices // mask_width).float() * h_step
        point_coords = torch.stack([xs, ys], dim=2)
        # ipdb.set_trace()

        return point_indices, point_coords

    def get_boundary_rel_points_test(self, mask_instance_pred, mask_semantic_pred, pred_label):
        """Get ``num_points`` boundary points using semantic mask during test.

        Args:
            rois (Tensor): shape (num_rois, 5).
            mask_instance_pred (Tensor): A instance pred tensor of shape (num_rois, num_classes,
                mask_height, mask_width) for class-specific or class-agnostic
                prediction.
            mask_semantic_pred (Tensor): A semantic pred tensor of shape (1, 1, x[0].shape[0], x[0].shape[1])
            pred_label (list): The predication class for each instance.

        Returns:
            point_indices List[(Tensor)]: A list[tensor] of shape (num_rois, num_points)
                that contains indices from [0, mask_height x mask_width) of the
                most uncertain points.
            point_coords List[(Tensor)]: A list[tensor] of shape (num_rois, num_points, 2)
                that contains [0, 1] x [0, 1] normalized coordinates of the
                most uncertain points from the [mask_height, mask_width] grid .
        """

        # ipdb.set_trace()
        boundary_points_map = get_boundary_points(mask_instance_pred, mask_semantic_pred, pred_label)
        mask_height, mask_width = boundary_points_map.shape[-2:]
        # ipdb.set_trace()
        # During ONNX exporting, the type of each elements of 'shape' is
        # `Tensor(float)`, while it is `float` during PyTorch inference.
        if isinstance(mask_height, torch.Tensor):
            h_step = 1.0 / mask_height.float()
            w_step = 1.0 / mask_width.float()
        else:
            h_step = 1.0 / mask_height
            w_step = 1.0 / mask_width
        # ipdb.set_trace()
        # cast to int to avoid dynamic K for TopK op in ONNX
        mask_size = int(mask_height * mask_width)
        boundary_points_map = [boundary_point_map.view(1, mask_size) for boundary_point_map in boundary_points_map]
        point_indices = [torch.where(boundary_point_map != 0)[1] for boundary_point_map in boundary_points_map]
        # ipdb.set_trace()

        point_coords = []
        for point_indice in point_indices:
            xs = w_step / 2.0 + (point_indice % mask_width).float() * w_step
            ys = h_step / 2.0 + (point_indice // mask_width).float() * h_step
            point_coord = torch.stack([xs, ys], dim=1)
            point_coords.append(point_coord)
        # ipdb.set_trace()

        return point_indices, point_coords

def get_boundary_points(mask_instance_pred, mask_semantic_pred, pred_label):

    # 提取出对应labele的 instance pred
    inds = torch.arange(mask_instance_pred.shape[0], device=mask_instance_pred.device)
    prediction_instance = mask_instance_pred[inds, pred_label].unsqueeze(1)
    mask_instance_logits = np.where(prediction_instance > 0.5, 1, 0) * 255

    # ipdb.set_trace()
    # for i, mask_instance in enumerate(mask_instance_logits):
    #     filename = "/hy-tmp/mmdetection/mask_instance_{}.png".format(i)
    #     save_semantic_logits_as_Image(filename, mask_instance)

    # ipdb.set_trace()
    mask_semantic_logits = np.where(mask_semantic_pred > 0.5, 1, 0) * 255
    # for i, mask_semantic in enumerate(mask_semantic_logits):
    #     filename = "/hy-tmp/mmdetection/mask_semantic_{}.png".format(i)
    #     save_semantic_logits_as_Image(filename, mask_semantic)

    # boundary points
    # ipdb.set_trace()
    mask_boundary = mask_instance_logits + mask_semantic_logits
    boundary_points = np.where(mask_boundary != 0, 1, 0)
    boundary_points = torch.from_numpy(boundary_points)

    # ipdb.set_trace()
    return boundary_points


def save_semantic_logits_as_Image(filename, prediction):
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.numpy()

    if prediction.shape[0] == 1:
        prediction = prediction.squeeze(0)

    prediction = prediction.astype(np.uint8)
    mask = Image.fromarray(prediction)
    mask.save(filename)