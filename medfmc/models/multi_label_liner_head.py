# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmcls.models.builder import HEADS
from mmcls.models.heads.multi_label_head import MultiLabelClsHead


@HEADS.register_module()
class MultiLabelLinearClsHead_merge(MultiLabelClsHead):
    """Linear classification head for multilabel task.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(MultiLabelLinearClsHead_merge, self).__init__(
            loss=loss, init_cfg=init_cfg)

        if num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        gt_label = gt_label.type_as(x)
        b,c = gt_label.size()
        if b % 4 == 0:
            gt_label_l, gt_label_r = gt_label[:b // 2, :], gt_label[b // 2:b, :]
        else:
            gt_label_l, gt_label_r = gt_label, gt_label
        gt_label_merge = gt_label_l + gt_label_r
        gt_label_merge[gt_label_merge==2] = 1

        if b % 4 == 0:
            gt_label = torch.cat([gt_label,gt_label_merge,gt_label_merge],dim=0)
        else:
            gt_label = gt_label

        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def simple_test(self, x, sigmoid=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            sigmoid (bool): Whether to sigmoid the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        b, c = x.size()
        if b == 8:
            x = x[:b//2, :]
        else:
            x = x
        cls_score = self.fc(x)

        if sigmoid:
            pred = torch.sigmoid(cls_score) if cls_score is not None else None
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred