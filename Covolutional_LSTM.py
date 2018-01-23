# -*- coding: utf-8 -*-

import sys, os, cv2 ,glob
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Variable, optimizers, Chain, datasets
from chainer import training
from itertools import chain
import cm
import cupy as cp

class Conv_LSTM(Chain):
    def __init__(self):
        self.Pi = L.Linear()
        self.Pf = L.Linear()
        self.Po = L.Linear()
        super(Conv_LSTM, self).__init__()

        with self.init_scope():
            self.Wz = L.Convolution2D()
            self.Wi = L.Convolution2D()
            self.Wf = L.Convolution2D()
            self.Wo = L.Convolution2D()
            self.Rz = L.Convolution2D()
            self.Ri = L.Convolution2D()
            self.Rf = L.Convolution2D()
            self.Ro = L.Convolution2D()

    def __call__(self, s):
        accum_loss = None
        hei = len(s[0])
        wid = len(s[0][0])
        h = Variable(cp.zeros((1, 1, hei, wid), dtype=cp.float32))
        c = Variable(cp.zeros((1, 1, hei, wid), dtype=cp.float32))


        for i in range(len(s) - 1):
            tx = Variable(cp.array(s[i + 1], dtype=cp.float32).reshape(1, 1, hei, wid))
            x_k = Variable(cp.array(s[i], dtype=cp.float32).reshape(1, 1, hei, wid))
            z0 = self.Wz(x_k) + self.Rz(h)   # LSTM-Blockへの入力+出力からのフィードバック
            z1 = F.tanh(z0)                  # input-gateを通過すれば記憶セルの入力へ
            i0 = self.Wi(x_k) + self.Ri(h)   # 新しい入力と前の時間の出力
            i1 = F.sigmoid(i0 + self.Pi(c))  # input-gate
            f0 = self.Wf(x_k) + self.Rf(h)   # forget-gateを通過すれば記憶セルの入力へ
            f1 = F.sigmoid(f0 + self.Pf(c))  # forget-gate
            c = z1 * i1 + f1 * c             # 記憶セル＝input-gateからの入力 + forget-gateからの入力 + 前の記憶
            o0 = self.Wo(x_k) + self.Ro(h)   # output-gateを通過すれば出力へ
            o1 = F.sigmoid(o0 + self.Po(c))  # output-gate
            h = o1 * F.tanh(c)               # output-gateからの入力+内部セルからの入力

            loss = F.mean_squared_error(h, tx)  # LSTMの出力と正解予測画像との誤差
            accum_loss = loss if accum_loss is None else accum_loss + loss

        return accum_loss
