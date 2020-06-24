# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import cPickle
import argparse
import pprint
import logging
import time
import os
import mxnet as mx
import numpy as np

from symbols import *
from dataset import *
from core.loader import TestLoader, TestLossLoader, TestBatchLoader, TestScaleRegLoader
from core.tester import Predictor, pred_eval, pred_scalereg, pred_eval_loss, pred_adavid_eval
from utils.load_model import load_param
from utils.load_data import load_gt_roidb, merge_roidb, filter_roidb

def test_rcnn_loss(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # setup multi-gpu
    batch_size = len(ctx) # number of gpus
    input_batch_size = cfg.TRAIN.BATCH_IMAGES * batch_size # batch size fetched from training data

    # load symbol and testing data
    sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    sym = sym_instance.get_testloss_symbol(cfg) # This is the function that computes loss!

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in image_set.split('+')]
    roidbs = []
    for iset in image_sets:
        imdb = eval(dataset)(iset, root_path, dataset_path, result_path=output_path)
        roidbs.append(imdb.gt_roidb())
    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb, cfg)

    # get test data iter
    test_data = TestLossLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)
    
    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True) # load a checkpoint

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single+test_data.provide_label_single)
    sym_instance.infer_shape(data_shape_dict)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = [k[0] for k in test_data.provide_label_single]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
    if not has_rpn:
        max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    pred_eval_loss(predictor, test_data, imdb, cfg, roidb, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)


def test_scalereg(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    sym = sym_instance.get_test_scalereg_symbol(cfg)

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in image_set.split('+')]
    roidbs = []
    for iset in image_sets:
        imdb = eval(dataset)(iset, root_path, dataset_path, result_path=output_path)
        roidbs.append(imdb.gt_roidb())
    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb, cfg)
    # imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
    # roidb = imdb.gt_roidb()

    optimal_output_path = os.path.join(imdb.result_path, cfg.symbol + '_regscale.pkl')
    # get test data iter
    test_data = TestScaleRegLoader(roidb, optimal_output_path, cfg, batch_size=cfg.TEST.BATCH_IMAGES, shuffle=shuffle, has_rpn=has_rpn)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (cfg.TEST.BATCH_IMAGES, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    # start detection
    pred_scalereg(predictor, test_data, imdb, cfg, roidb, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)


def test_adaptive_vid(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    # sym = sym_instance.get_test_symbol(cfg)
    sym = sym_instance.get_test_adaptive_scale_symbol(cfg)
    # sym = sym_instance.get_test_adaptive_scale_1x1_symbol(cfg)

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in image_set.split('+')]
    roidbs = []
    for iset in image_sets:
        imdb = eval(dataset)(iset, root_path, dataset_path, result_path=output_path)
        roidbs.append(imdb.gt_roidb())
    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb, cfg)
    # imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
    # roidb = imdb.gt_roidb()

    # get test data iter
    test_data = TestLoader(roidb, '', cfg, batch_size=cfg.TEST.BATCH_IMAGES, shuffle=shuffle, has_rpn=has_rpn)

    rfcn_model_prefix = prefix+cfg.TRAIN.rfcn_model_prefix 
    regressor_model_prefix = prefix+cfg.TRAIN.regressor_model_prefix
    # load model rfcn
    arg_params_rfcn, aux_params_rfcn = load_param(rfcn_model_prefix, epoch, process=True)
    arg_params_reg, aux_params_reg = load_param(regressor_model_prefix, cfg.TEST.reg_test_epoch, process=True)
    arg_params = arg_params_rfcn.copy()
    aux_params = aux_params_rfcn.copy()
    for key in arg_params_reg:
        if not key in arg_params:
            arg_params[key] = arg_params_reg[key]
    for key in aux_params_reg:
        if not key in aux_params:
            aux_params[key] = aux_params_reg[key]
    print('Testing with R-FCN:{}, ScaleReg:{}'.format(epoch, cfg.TEST.reg_test_epoch))

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (cfg.TEST.BATCH_IMAGES, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
    if not has_rpn:
        max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # start detection
    pred_adavid_eval(predictor, imdb, cfg, roidb, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)


    
def test_rcnn(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    sym = sym_instance.get_test_symbol(cfg)

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in image_set.split('+')]
    roidbs = []
    for iset in image_sets:
        imdb = eval(dataset)(iset, root_path, dataset_path, result_path=output_path)
        roidbs.append(imdb.gt_roidb())
    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb, cfg)
    # imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
    # roidb = imdb.gt_roidb()

    # optimal_output_path = os.path.join(imdb.result_path, cfg.symbol + '_regscale-0004.pkl')
    # optimal_output_path = os.path.join(imdb.result_path, cfg.symbol + '_regscale.pkl')
    optimal_output_path = os.path.join(imdb.result_path, cfg.symbol + '_optimal.pkl')
    # get test data iter
    if cfg.OPTIMAL:
        test_data = TestLoader(roidb, optimal_output_path, cfg, batch_size=cfg.TEST.BATCH_IMAGES, shuffle=shuffle, has_rpn=has_rpn)
    else:
        test_data = TestLoader(roidb, '', cfg, batch_size=cfg.TEST.BATCH_IMAGES, shuffle=shuffle, has_rpn=has_rpn)
    # Batch by Rudy
    # test_data = TestBatchLoader(roidb, optimal_output_path, cfg, batch_size=cfg.TEST.BATCH_IMAGES, shuffle=shuffle, has_rpn=has_rpn)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (cfg.TEST.BATCH_IMAGES, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
    if not has_rpn:
        max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # start detection
    pred_eval(predictor, test_data, imdb, cfg, roidb, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)

