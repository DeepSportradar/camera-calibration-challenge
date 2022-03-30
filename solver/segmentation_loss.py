# encoding: utf-8
"""
@author:  davide zambrano
@contact: d.zambrano@sportradar.com
"""


class loss_fn_seg:
    def __init__(self, loss):
        self.loss = loss

    def __call__(self, ypred, y):
        return self.loss(ypred["out"], y)


class loss_fn_seg_dlv3:
    def __init__(self, loss):
        self.loss = loss

    def __call__(self, ypred, y):
        losses = {}
        for name, x in ypred.items():
            losses[name] = self.loss(x, y, ignore_index=255)
        if len(losses) == 1:
            return losses["out"]
        return losses["out"] + 0.5 * losses["aux"]
