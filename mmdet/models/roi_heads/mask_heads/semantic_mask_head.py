# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule, Linear
from mmcv.runner import ModuleList, auto_fp16
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss
from .fcn_mask_head import FCNMaskHead
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
import ipdb
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from warnings import warn
from PIL import Image


@HEADS.register_module()
class SemanticMaskHead(FCNMaskHead):
    """Semantic mask head.
    """

    def __init__(self,
                 *arg,
                 class_agnostic=True,
                 upsample_cfg=dict(type='deconv', scale_factor=1),
                 loss_seg=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 **kwarg):
        super(SemanticMaskHead, self).__init__(
            *arg,
            class_agnostic=class_agnostic,
            upsample_cfg=upsample_cfg,
            **kwarg)
        self.loss_seg = build_loss(loss_seg)

    def init_weights(self):
        super(FCNMaskHead, self).init_weights()

    def get_targets(self, gt_masks, device, dtype):

        gt_semantic_mask = []
        for mask in gt_masks:
            semantic_mask = np.sum(mask.masks, axis=0)
            semantic_mask = np.where(semantic_mask != 0, 1, 0)
            semantic_mask = torch.tensor(semantic_mask, dtype=torch.uint8)
            gt_semantic_mask.append(semantic_mask)

        # ipdb.set_trace()
        max_h = max([target.shape[-2] for target in gt_semantic_mask])
        max_w = max([target.shape[-1] for target in gt_semantic_mask])
        # ipdb.set_trace()
        semantic_target = torch.zeros(
            (len(gt_semantic_mask), max_h, max_w),
            dtype=dtype, device=device)
        # ipdb.set_trace()
        for idx, target in enumerate(gt_semantic_mask):
            semantic_target[idx, :target.shape[-2], :target.shape[-1]] = target
        # ipdb.set_trace()

        return semantic_target

    @auto_fp16()
    def forward(self, feats):
        # ipdb.set_trace()
        assert isinstance(feats, torch.Tensor)
        mask_pred_semantic = super().forward(feats)
        # ipdb.set_trace()
        return mask_pred_semantic

    def loss(self, mask_pred_semantic, semantic_target):
        # ipdb.set_trace(context=5)
        semantic_target = F.interpolate(
            semantic_target.unsqueeze(1),
            mask_pred_semantic.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        semantic_target = (semantic_target >= 0.5).float()

        # ipdb.set_trace()
        loss = dict()
        loss_seg = F.binary_cross_entropy_with_logits(
                    mask_pred_semantic.squeeze(1), semantic_target, reduction='mean')
        loss['loss_seg'] = loss_seg
        # ipdb.set_trace()
        return loss


    def get_seg_masks(self, mask_semantic_pred, rcnn_test_cfg):
        """Get segmentation masks from semantic pred.

        Args:
            mask_pred (Tensor or ndarray): shape (1, 1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)
            scale_factor(ndarray | Tensor): If ``rescale is True``, box
                coordinates are divided by this scale factor to fit
                ``ori_shape``.
            rescale (bool): If True, the resulting masks will be rescaled to
                ``ori_shape``.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the
                i-th item in that inner list is the mask for the i-th box with
                class label c.
        """

        ipdb.set_trace()
        img_h, img_w = mask_semantic_pred.size[-2:]
        im_mask = torch.zeros([img_h, img_w], dtype=torch.uint8)

        mask_semantic_pred = mask_semantic_pred.sigmoid().squeeze()
        threshold = rcnn_test_cfg.mask_thr_binary
        mask_semantic_pred = (mask_semantic_pred >= threshold).to(dtype=torch.bool)

        im_mask[mask_semantic_pred] = 255
        ipdb.set_trace()

        filename = "/hy-tmp/mmdetection/semantic.png"
        save_semantic_logits_as_Image(filename, im_mask)

        return mask_semantic_pred

def save_semantic_logits_as_Image(filename, prediction):
    prediction = prediction.numpy().astype(np.uint8)

    mask = Image.fromarray(prediction)
    mask.save(filename)
