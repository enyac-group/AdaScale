# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import cPickle
import os
import time
import mxnet as mx
import numpy as np

from module import MutableModule
from utils import image
from bbox.bbox_transform import bbox_pred, clip_boxes
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper, py_softnms_wrapper
from utils.PrefetchingIter import PrefetchingIter
from utils.image import get_scaled_frame


class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        # [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]
        return [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]

    def predict_loss(self, data_batch):
        self._mod.forward(data_batch, is_train=True)
        # [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]
        return [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]


def im_proposal(predictor, data_batch, data_names, scales):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    boxes_all = []

    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        # drop the batch index
        boxes = output['rois_output'].asnumpy()[:, 1:]
        scores = output['rois_score'].asnumpy()

        # transform to original scale
        boxes = boxes / scale
        scores_all.append(scores)
        boxes_all.append(boxes)

    return scores_all, boxes_all, data_dict_all


def generate_proposals(predictor, test_data, imdb, cfg, vis=False, thresh=0.):
    """
    Generate detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: thresh for valid detections
    :return: list of detected boxes
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    idx = 0
    t = time.clock()
    imdb_boxes = list()
    original_boxes = list()
    for im_info, data_batch in test_data:
        t1 = time.clock() - t
        t = time.clock()

        scales = [iim_info[0, 2] for iim_info in im_info]
        scores_all, boxes_all, data_dict_all = im_proposal(predictor, data_batch, data_names, scales)
        t2 = time.clock() - t
        t = time.clock()
        for delta, (scores, boxes, data_dict, scale) in enumerate(zip(scores_all, boxes_all, data_dict_all, scales)):
            # assemble proposals
            dets = np.hstack((boxes, scores))
            original_boxes.append(dets)

            # filter proposals
            keep = np.where(dets[:, 4:] > thresh)[0]
            dets = dets[keep, :]
            imdb_boxes.append(dets)

            if vis:
                vis_all_detection(data_dict['data'].asnumpy(), [dets], ['obj'], scale, cfg)

            print 'generating %d/%d' % (idx + 1, imdb.num_images), 'proposal %d' % (dets.shape[0]), \
                'data %.4fs net %.4fs' % (t1, t2 / test_data.batch_size)
            idx += 1


    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'

    # save results
    rpn_folder = os.path.join(imdb.result_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)

    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if thresh > 0:
        full_rpn_file = os.path.join(rpn_folder, imdb.name + '_full_rpn.pkl')
        with open(full_rpn_file, 'wb') as f:
            cPickle.dump(original_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'wrote rpn proposals to {}'.format(rpn_file)
    return imdb_boxes


def im_detectScale(predictor, data_batch, data_names, cfg):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, idata)) for idata in data_batch.data]
    reg_scale_all = []
    for output, data_dict in zip(output_all, data_dict_all):
        im_shape = data_dict['data'].shape

        # save output
        shortest = np.min(im_shape[-2:])
        reg_scale = output['scale_output'].asnumpy()[0]
        reg_scale = ((reg_scale+1)/float(2)*(float(600)/128 - float(128)/600) + float(128)/600) * shortest
        reg_scale = int(np.clip(reg_scale, 128, 600))

        reg_scale_all.append(reg_scale)
    return reg_scale_all, data_dict_all

def im_detectWithScale(predictor, data_batch, scales, cfg):
    output_all = predictor.predict(data_batch)
    data_names = ['data', 'im_info']
    data_dict_all = [dict(zip(data_names, idata)) for idata in data_batch.data]

    scores_all = []
    pred_boxes_all = []
    reg_scale_all = []
    for output, scale, data_dict in zip(output_all, scales, data_dict_all):
        rois = output['rois_output'].asnumpy()[:, 1:]
        im_shape = data_dict['data'].shape

        # save output
        shortest = np.min(im_shape[-2:])
        reg_scale = output['scale_output'].asnumpy()[0]
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]

        # post processing
        reg_scale = ((reg_scale+1)/float(2)*(float(cfg.MAX_REG_SCALE)/cfg.MIN_REG_SCALE - float(cfg.MIN_REG_SCALE)/cfg.MAX_REG_SCALE) + float(cfg.MIN_REG_SCALE)/cfg.MAX_REG_SCALE) * shortest
        reg_scale = int(np.clip(reg_scale, cfg.MIN_REG_SCALE, cfg.MAX_REG_SCALE))
        pred_boxes = bbox_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

        # we used scaled image & roi to train, so it is necessary to transform them back
        pred_boxes = pred_boxes / scale

        scores_all.append(scores)
        pred_boxes_all.append(pred_boxes)
        reg_scale_all.append(reg_scale)
    return scores_all, pred_boxes_all, reg_scale_all

def im_detectLoss(predictor, data_batch, data_names, scales, cfg):
    output_all = predictor.predict_loss(data_batch)

    data_dict_all = [dict(zip(data_names, idata)) for idata in data_batch.data]
    scores_all = []
    pred_boxes_all = []
    cls_loss_all = []
    bbox_loss_all = []
    label_all = []
    bbox_pred_all = []
    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        if cfg.TEST.HAS_RPN:
            rois = output['rois_output'].asnumpy()[:, 1:]
        else:
            rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
        im_shape = data_dict['data'].shape

        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]

        cls_loss_all.append(output['cls_loss_output'].asnumpy())
        bbox_loss_all.append(output['bbox_loss_output'].asnumpy())
        label_all.append(output['label_output'].asnumpy())
        bbox_pred_all.append(bbox_deltas)

        # post processing
        pred_boxes = bbox_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

        # we used scaled image & roi to train, so it is necessary to transform them back
        pred_boxes = pred_boxes / scale

        scores_all.append(scores)
        pred_boxes_all.append(pred_boxes)
    return scores_all, pred_boxes_all, data_dict_all, cls_loss_all, bbox_loss_all, label_all, bbox_pred_all

def im_detect(predictor, data_batch, data_names, scales, cfg):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, idata)) for idata in data_batch.data]
    scores_all = []
    pred_boxes_all = []
    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        rois = output['rois_output'].asnumpy()[:, 1:]
        im_shape = data_dict['data'].shape

        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]

        # post processing
        pred_boxes = bbox_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

        # we used scaled image & roi to train, so it is necessary to transform them back
        pred_boxes = pred_boxes / scale

        scores_all.append(scores)
        pred_boxes_all.append(pred_boxes)
    return scores_all, pred_boxes_all, data_dict_all

def im_batch_detect(predictor, data_batch, data_names, scales, cfg):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    pred_boxes_all = []
    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        im_infos = data_dict['im_info'].asnumpy()
        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
        rois = output['rois_output'].asnumpy()
        for im_idx in xrange(im_infos.shape[0]):
            bb_idxs = np.where(rois[:,0] == im_idx)[0]
            im_shape = im_infos[im_idx, :2].astype(np.int)

            # post processing
            pred_boxes = bbox_pred(rois[bb_idxs, 1:], bbox_deltas[bb_idxs, :])
            pred_boxes = clip_boxes(pred_boxes, im_shape)

            # we used scaled image & roi to train, so it is necessary to transform them back
            pred_boxes = pred_boxes / scale[im_idx]

            scores_all.append(scores[bb_idxs, :])
            pred_boxes_all.append(pred_boxes)

    return scores_all, pred_boxes_all, data_dict_all

def pred_scalereg(predictor, test_data, imdb, cfg, roidb, vis=False, thresh=1e-3, logger=None, ignore_cache=True):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """

    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]

    num_images = test_data.size
    out_name = test_data.name
    print('Output optimal sizes: {}'.format(out_name))
    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    nms = py_nms_wrapper(cfg.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = cfg.TEST.max_per_image

    reg_scales = {}

    idx = 0
    data_time, net_time, post_time = 0.0, 0.0, 0.0
    t = time.clock()
    for data_batch in test_data:
        t1 = time.clock() - t
        t = time.clock()

        reg_scale_all , data_dict_all = im_detectScale(predictor, data_batch, data_names, cfg)

        t2 = time.clock() - t
        t = time.clock()
        for delta, (reg_scale, data_dict) in enumerate(zip(reg_scale_all, data_dict_all)):
            reg_scales[int(roidb[idx+delta]['frame_id'])] = (reg_scale, 2000)
            # print(reg_scales[roidb[idx+delta]['frame_id']])

        idx += test_data.batch_size
        t3 = time.clock() - t
        t = time.clock()
        data_time += t1
        net_time += t2
        post_time += t3
        if idx % 1000 == 0:
            print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, num_images, data_time / idx * test_data.batch_size, net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size)

    with open(out_name, 'wb') as f:
        cPickle.dump(reg_scales, f, protocol=cPickle.HIGHEST_PROTOCOL)

def detect_at_single_scale_withloss(predictor, roidb, data_names, imdb, test_data, cfg, thresh, vis, all_boxes_single_scale, logger):
    loss_file = os.path.join(imdb.result_path, cfg.symbol + '_loss_{}_{}.pkl'.format(cfg.SCALES[0][0], cfg.SCALES[0][1]))
    idx = 0
    data_time, net_time, post_time = 0.0, 0.0, 0.0
    t = time.time()

    output = []
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()
        scales = [iim_info[2] for iim_info in im_info]
        scores_all, boxes_all, data_dict_all, cls_loss_all, bbox_loss_all, label_all, bbox_pred_all = im_detectLoss(predictor, data_batch, data_names, scales, cfg)

        t2 = time.time() - t
        t = time.time()
        for delta, (scores, boxes, data_dict, cls_loss, bbox_loss, label, bbox_pred_t) in enumerate(zip(scores_all, boxes_all, data_dict_all, cls_loss_all, bbox_loss_all, label_all, bbox_pred_all)):
            output.append({'frame_id': int(roidb[idx+delta]['frame_id']), 'cls_loss': cls_loss, 'bbox_loss': bbox_loss, 'label': label, 'bbox_pred': bbox_pred_t})
            for j in range(1, imdb.num_classes):
                indexes = np.where(scores[:, j] > thresh)[0]
                cls_scores = scores[indexes, j, np.newaxis]
                cls_boxes = boxes[indexes, 4:8] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores)).copy()
                all_boxes_single_scale[j][idx + delta] = cls_dets
            if vis:
                boxes_this_image = [[]] + [all_boxes_single_scale[j][idx + delta] for j in range(1, imdb.num_classes)]
                data_for_vis = data_dict['data'].asnumpy().copy()
                vis_all_detection(data_for_vis, boxes_this_image, imdb.classes, scales[delta], cfg)

        idx += test_data.batch_size
        t3 = time.time() - t
        t = time.time()
        data_time += t1
        net_time += t2
        post_time += t3

        if idx % 1000 == 0:
            print 'testing {}/{} with scale {}: data {:.4f}s net {:.4f}s post {:.4f}s' \
                .format(idx, imdb.num_images, cfg.SCALES, data_time / idx * test_data.batch_size,
                        net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size)
            if logger:
                logger.info('testing {}/{} with scale {}: data {:.4f}s net {:.4f}s post {:.4f}s'
                            .format(idx, imdb.num_images, cfg.SCALES, data_time / idx * test_data.batch_size,
                                    net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size))
    with open(loss_file, 'wb') as f:
        cPickle.dump(output, f, protocol=cPickle.HIGHEST_PROTOCOL)


def pred_eval_loss(predictor, test_data, imdb, cfg, roidb, vis=False, thresh=1e-3, logger=None, ignore_cache=True):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """

    if cfg.OPTIMAL:
        det_file = os.path.join(imdb.result_path, imdb.name + '_detections_optim.pkl')
    else:
        det_file = os.path.join(imdb.result_path, imdb.name + '_detections.pkl')
    # if os.path.exists(det_file) and not ignore_cache:
    #     with open(det_file, 'rb') as fid:
    #         all_boxes = cPickle.load(fid)
    #     # Should calculate NMS here
    #     info_str = imdb.evaluate_detections(all_boxes, roidb)
    #     if logger:
    #         logger.info('evaluate detections: \n{}'.format(info_str))

    #     # Rudy: Below are used to generate images with bounding boxes
    #     # optimal_output_path = os.path.join(imdb.result_path, cfg.symbol + '_optimal.pkl')
    #     # imdb.annotate_detections(optimal_output_path, all_boxes, roidb)
    #     return

    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]

    num_images = test_data.size
    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    # limit detections to max_per_image over all classes
    max_per_image = cfg.TEST.max_per_image

    for test_scale in cfg.TEST_SCALES:
        det_file_single_scale = os.path.join(imdb.result_path, imdb.name + '_detections_prenms_{}.pkl'.format(test_scale[0]))
        if os.path.exists(det_file_single_scale):
           continue
        cfg.SCALES = [test_scale]
        test_data.reset()

        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes_single_scale = [[[] for _ in range(num_images)]
                                    for _ in range(imdb.num_classes)]
        detect_at_single_scale_withloss(predictor, roidb, data_names, imdb, test_data, cfg, thresh, vis, all_boxes_single_scale, logger)

        with open(det_file_single_scale, 'wb') as f:
            cPickle.dump(all_boxes_single_scale, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]
    
    for test_scale in cfg.TEST_SCALES:
        print('Loading prenms results of scale {}'.format(test_scale))
        det_file_single_scale = os.path.join(imdb.result_path, imdb.name + '_detections_prenms_{}.pkl'.format(test_scale[0]))
        if os.path.exists(det_file_single_scale):
            with open(det_file_single_scale, 'rb') as fid:
                all_boxes_single_scale = cPickle.load(fid)
            for idx_class in range(1, imdb.num_classes):
                for idx_im in range(0, num_images):
                    if len(all_boxes[idx_class][idx_im]) == 0:
                        all_boxes[idx_class][idx_im] = all_boxes_single_scale[idx_class][idx_im]
                    else:
                        all_boxes[idx_class][idx_im] = np.vstack((all_boxes[idx_class][idx_im], all_boxes_single_scale[idx_class][idx_im]))

    for idx_class in range(1, imdb.num_classes):
        for idx_im in range(0, num_images):
            if cfg.TEST.USE_SOFTNMS:
                soft_nms = py_softnms_wrapper(cfg.TEST.SOFTNMS_THRESH, max_dets=max_per_image)
                all_boxes[idx_class][idx_im] = soft_nms(all_boxes[idx_class][idx_im])
            else:
                nms = py_nms_wrapper(cfg.TEST.NMS)
                keep = nms(all_boxes[idx_class][idx_im])
                all_boxes[idx_class][idx_im] = all_boxes[idx_class][idx_im][keep, :]

    if max_per_image > 0:
        for idx_im in range(0, num_images):
            image_scores = np.hstack([all_boxes[j][idx_im][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][idx_im][:, -1] >= image_thresh)[0]
                    all_boxes[j][idx_im] = all_boxes[j][idx_im][keep, :]

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)

    info_str = imdb.evaluate_detections(all_boxes, roidb)
    if logger:
        logger.info('evaluate detections: \n{}'.format(info_str))


def pred_adavid_eval(predictor, imdb, cfg, roidb, vis=False, thresh=1e-3, logger=None, ignore_cache=True):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """

    det_file = os.path.join(imdb.result_path, imdb.name + '_detections_vidscale_{}_{}.pkl'.format(cfg.SCALES[0][0], cfg.SCALES[0][1]))
    scale_file = os.path.join(imdb.result_path, '{}_scalePerImage.pkl'.format(cfg.TRAIN.regressor_model_prefix))
    if os.path.exists(det_file) and not ignore_cache:
        with open(det_file, 'rb') as fid:
            all_boxes = cPickle.load(fid)

        info_str = imdb.evaluate_detections(all_boxes, roidb)
        if logger:
            logger.info('evaluate detections: \n{}'.format(info_str))
        return

    num_images = len(roidb)

    nms = py_nms_wrapper(cfg.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = cfg.TEST.max_per_image

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    # Init scale (scale, frame_id)
    regscale_from_lastframe = (cfg.SCALES[0][0], int(roidb[0]['frame_id']))

    all_scales = {}
    scale_list = []
    data_time, net_time, post_time = 0.0, 0.0, 0.0
    t = time.clock()
    for idx in range(num_images):
        # if np.abs(regscale_from_lastframe[1] - int(roidb[idx]['frame_id'])) < 2:
        data, scales = get_scaled_frame(roidb[idx], cfg, regscale_from_lastframe[0])
        all_scales[int(roidb[idx]['frame_id'])] = (regscale_from_lastframe[0], 2000)

        # else:
            # data, scales = get_scaled_frame(roidb[idx], cfg, cfg.SCALES[0][0])
            # all_scales[int(roidb[idx]['frame_id'])] = (cfg.SCALES[0][0], 2000)
            # scale_list.append(cfg.SCALES[0][0])

        provide_data = [[(k, v.shape) for k, v in zip(['data', 'im_info'], idata)] for idata in data]
        # print(provide_data)
        provide_label = [None]

        data_batch = mx.io.DataBatch(data=data, label=[], provide_data=provide_data, provide_label=provide_label)
        t1 = time.clock() - t
        t = time.clock()

        scores_all, boxes_all, reg_scale_all = im_detectWithScale(predictor, data_batch, scales, cfg)

        t2 = time.clock() - t
        t = time.clock()
        for delta, (scores, boxes, reg_scale) in enumerate(zip(scores_all, boxes_all, reg_scale_all)):
            for j in range(1, imdb.num_classes):
                indexes = np.where(scores[:, j] > thresh)[0]
                cls_scores = scores[indexes, j, np.newaxis]
                cls_boxes = boxes[indexes, 4:8] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                all_boxes[j][idx+delta] = cls_dets[keep, :]
                
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][idx+delta][:, -1]
                                          for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][idx+delta][:, -1] >= image_thresh)[0]
                        all_boxes[j][idx+delta] = all_boxes[j][idx+delta][keep, :]
            scale_list.append(regscale_from_lastframe[0])
            regscale_from_lastframe = (reg_scale, int(roidb[idx+delta]['frame_id']))

        t3 = time.clock() - t
        data_time += t1
        net_time += t2
        post_time += t3
        if idx % 1000 == 0:
            print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s avg_scale: {:.2f}'.format(idx, num_images, data_time / (idx+1), net_time / (idx+1), post_time / (idx+1), np.mean(scale_list))
            if logger:
                logger.info('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s avg_scale: {:.2f}'.format(idx, num_images, data_time / (idx+1), net_time / (idx+1), post_time / (idx+1), np.mean(scale_list)))
        t = time.clock()

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)

    with open(scale_file, 'wb') as f:
        cPickle.dump(all_scales, f, protocol=cPickle.HIGHEST_PROTOCOL)

    info_str = imdb.evaluate_detections(all_boxes, roidb)
    if logger:
        logger.info('evaluate detections: \n{}'.format(info_str))


def detect_at_single_scale(predictor, data_names, imdb, test_data, cfg, thresh, vis, all_boxes_single_scale, logger):
    idx = 0
    data_time, net_time, post_time = 0.0, 0.0, 0.0
    t = time.time()

    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()
        scales = [iim_info[0, 2] for iim_info in im_info]
        scores_all, boxes_all, data_dict_all = im_detect(predictor, data_batch, data_names, scales, cfg)

        t2 = time.time() - t
        t = time.time()
        for delta, (scores, boxes, data_dict) in enumerate(zip(scores_all, boxes_all, data_dict_all)):
            for j in range(1, imdb.num_classes):
                indexes = np.where(scores[:, j] > thresh)[0]
                cls_scores = scores[indexes, j, np.newaxis]
                cls_boxes = boxes[indexes, 4:8] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores)).copy()
                all_boxes_single_scale[j][idx + delta] = cls_dets
            if vis:
                boxes_this_image = [[]] + [all_boxes_single_scale[j][idx + delta] for j in range(1, imdb.num_classes)]
                data_for_vis = data_dict['data'].asnumpy().copy()
                vis_all_detection(data_for_vis, boxes_this_image, imdb.classes, scales[delta], cfg)

        idx += test_data.batch_size
        t3 = time.time() - t
        t = time.time()
        data_time += t1
        net_time += t2
        post_time += t3

        if idx % 1000 == 0:
            print 'testing {}/{} with scale {}: data {:.4f}s net {:.4f}s post {:.4f}s' \
                .format(idx, imdb.num_images, cfg.SCALES, data_time / idx * test_data.batch_size,
                        net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size)
            if logger:
                logger.info('testing {}/{} with scale {}: data {:.4f}s net {:.4f}s post {:.4f}s'
                            .format(idx, imdb.num_images, cfg.SCALES, data_time / idx * test_data.batch_size,
                                    net_time / idx * test_data.batch_size, post_time / idx * test_data.batch_size))



def pred_eval(predictor, test_data, imdb, cfg, roidb, vis=False, thresh=1e-3, logger=None, ignore_cache=True):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """

    if cfg.OPTIMAL:
        det_file = os.path.join(imdb.result_path, imdb.name + '_detections_optim.pkl')
    else:
        det_file = os.path.join(imdb.result_path, imdb.name + '_detections.pkl')
    if os.path.exists(det_file) and not ignore_cache:
        with open(det_file, 'rb') as fid:
            all_boxes = cPickle.load(fid)

        info_str = imdb.evaluate_detections(all_boxes, roidb)
        if logger:
            logger.info('evaluate detections: \n{}'.format(info_str))

        # Rudy: Below are used to generate images with bounding boxes
        # optimal_output_path = os.path.join(imdb.result_path, cfg.symbol + '_optimal.pkl')
        # imdb.annotate_detections(optimal_output_path, all_boxes, roidb)
        return

    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]

    num_images = test_data.size
    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    nms = py_nms_wrapper(cfg.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = cfg.TEST.max_per_image

    for test_scale in cfg.TEST_SCALES:
        det_file_single_scale = os.path.join(imdb.result_path, imdb.name + '_detections_prenms_{}.pkl'.format(test_scale[0]))
        if os.path.exists(det_file_single_scale):
           continue
        cfg.SCALES = [test_scale]
        test_data.reset()

        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes_single_scale = [[[] for _ in range(num_images)]
                                    for _ in range(imdb.num_classes)]
        detect_at_single_scale(predictor, data_names, imdb, test_data, cfg, thresh, vis, all_boxes_single_scale, logger)

        with open(det_file_single_scale, 'wb') as f:
            cPickle.dump(all_boxes_single_scale, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]
    
    for test_scale in cfg.TEST_SCALES:
        det_file_single_scale = os.path.join(imdb.result_path, imdb.name + '_detections_prenms_{}.pkl'.format(test_scale[0]))
        if os.path.exists(det_file_single_scale):
            with open(det_file_single_scale, 'rb') as fid:
                all_boxes_single_scale = cPickle.load(fid)
            for idx_class in range(1, imdb.num_classes):
                for idx_im in range(0, num_images):
                    if len(all_boxes[idx_class][idx_im]) == 0:
                        all_boxes[idx_class][idx_im] = all_boxes_single_scale[idx_class][idx_im]
                    else:
                        all_boxes[idx_class][idx_im] = np.vstack((all_boxes[idx_class][idx_im], all_boxes_single_scale[idx_class][idx_im]))

    for idx_class in range(1, imdb.num_classes):
        for idx_im in range(0, num_images):
            if cfg.TEST.USE_SOFTNMS:
                soft_nms = py_softnms_wrapper(cfg.TEST.SOFTNMS_THRESH, max_dets=max_per_image)
                all_boxes[idx_class][idx_im] = soft_nms(all_boxes[idx_class][idx_im])
            else:
                nms = py_nms_wrapper(cfg.TEST.NMS)
                keep = nms(all_boxes[idx_class][idx_im])
                all_boxes[idx_class][idx_im] = all_boxes[idx_class][idx_im][keep, :]

    if max_per_image > 0:
        for idx_im in range(0, num_images):
            image_scores = np.hstack([all_boxes[j][idx_im][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][idx_im][:, -1] >= image_thresh)[0]
                    all_boxes[j][idx_im] = all_boxes[j][idx_im][keep, :]

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)

    info_str = imdb.evaluate_detections(all_boxes, roidb)
    if logger:
        logger.info('evaluate detections: \n{}'.format(info_str))


def vis_all_detection(im_array, detections, class_names, scale, cfg, threshold=1e-3):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()


def draw_all_detection(im_array, detections, class_names, scale, cfg, threshold=1e-1):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2
    import random
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im
