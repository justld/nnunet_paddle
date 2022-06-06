import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg3d.cvlibs import manager


@manager.LOSSES.add_component
class DC_and_CE_loss(nn.Layer):
    def __init__(self, batch_dice=True, smooth=1e-5, do_bg=False, ce_kwargs={}, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        self.dc = SoftDiceLoss(apply_softmax=True, batch_dice=batch_dice, smooth=smooth, do_bg=do_bg)


    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        dice_coef = dc
        dc_loss = -dc.mean()     # 为了得到perchannel dice，修改了损失函数的部分逻辑，这部分原来在softdiceloss里面
        if self.log_dice:
            dc_loss = -paddle.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].astype('float32')) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()
        # print('dc: {}, ce: {}'.format(dc_loss.numpy(), ce_loss.numpy()))
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result, dice_coef.numpy()
        

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        input = input.transpose([0, *[i for i in range(2, input.ndim)], 1])
        return super().forward(input, target.astype('int64'))


class SoftDiceLoss(nn.Layer):
    def __init__(self, apply_softmax=True, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_softmax = apply_softmax
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_softmax:
            x = F.softmax(x, axis=1)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        # print("tp: {}, fp: {}, fn: {}, fg: {}. {}, {}.".format(tp.numpy(), fp.numpy(), fn.numpy(), paddle.sum(y == 1).numpy(), x.shape, y.shape))

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        # dc = dc.mean()

        # return -dc
        return dc

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    shp_x = net_output.shape
    shp_y = gt.shape

    with paddle.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.reshape((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.astype('int64')[:, 0, ...]
            y_onehot = paddle.nn.functional.one_hot(gt, num_classes=net_output.shape[1])
            y_onehot = y_onehot.transpose([0, y_onehot.ndim - 1, *[i for i in range(1, y_onehot.ndim - 1)]])

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = paddle.stack(tuple(x_i * mask[:, 0] for x_i in paddle.unbind(tp, axis=1)), axis=1)
        fp = paddle.stack(tuple(x_i * mask[:, 0] for x_i in paddle.unbind(fp, axis=1)), axis=1)
        fn = paddle.stack(tuple(x_i * mask[:, 0] for x_i in paddle.unbind(fn, axis=1)), axis=1)
        tn = paddle.stack(tuple(x_i * mask[:, 0] for x_i in paddle.unbind(tn, axis=1)), axis=1)
    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)
    return tp, fp, fn, tn


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp



