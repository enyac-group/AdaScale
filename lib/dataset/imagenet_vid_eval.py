# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Xizhou Zhu
# --------------------------------------------------------

"""
given a imagenet vid imdb, compute mAP
"""

import numpy as np
import os
import cPickle

CLASS_TEXT = ['__background__',  # always index 0
                'airplane', 'antelope', 'bear', 'bicycle',
                'bird', 'bus', 'car', 'cattle',
                'dog', 'domestic_cat', 'elephant', 'fox',
                'giant_panda', 'hamster', 'horse', 'lion',
                'lizard', 'monkey', 'motorcycle', 'rabbit',
                'red_panda', 'sheep', 'snake', 'squirrel',
                'tiger', 'train', 'turtle', 'watercraft',
                'whale', 'zebra']

def parse_vid_rec(filename, classhash, img_ids, defaultIOUthr=0.5, pixelTolerance=10):
    """
    parse imagenet vid record into a dictionary
    :param filename: xml file path
    :return: list of dict
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        if not classhash.has_key(obj.find('name').text):
            continue
        obj_dict['label'] = classhash[obj.find('name').text]
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [float(bbox.find('xmin').text),
                            float(bbox.find('ymin').text),
                            float(bbox.find('xmax').text),
                            float(bbox.find('ymax').text)]
        gt_w = obj_dict['bbox'][2] - obj_dict['bbox'][0] + 1
        gt_h = obj_dict['bbox'][3] - obj_dict['bbox'][1] + 1
        thr = (gt_w*gt_h)/((gt_w+pixelTolerance)*(gt_h+pixelTolerance))
        obj_dict['thr'] = np.min([thr, defaultIOUthr])
        objects.append(obj_dict)
    return {'bbox' : np.array([x['bbox'] for x in objects]),
             'label': np.array([x['label'] for x in objects]),
             'thr'  : np.array([x['thr'] for x in objects]),
             'img_ids': img_ids}


def vid_ap(rec, prec):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :return: average precision
    """

    # append sentinel values at both ends
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute precision integration ladder
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # look for recall value changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # sum (\delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def vid_eval(detpath, annopath, imageset_file, classname_map, annocache, ovthresh=0.5, optimal_path=''):
    """
    imagenet vid evaluation
    :param detpath: detection results detpath.format(classname)
    :param annopath: annotations annopath.format(classname)
    :param imageset_file: text file containing list of images
    :param annocache: caching annotations
    :param ovthresh: overlap threshold
    :return: rec, prec, ap
    """
    img_basenames = []
    gt_img_ids = []
    with open(imageset_file, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            if len(line) == 2:
                img_basenames.append(line[0])
            else:
                img_basenames.append('{}/{:06d}'.format(line[0], int(line[2])))
            gt_img_ids.append(int(line[1]))
    classhash = dict(zip(classname_map, range(0,len(classname_map))))

    # load annotations from cache
    if not os.path.isfile(annocache):
        recs = []
        for ind, image_filename in enumerate(img_basenames):
            if os.path.isfile(annopath.format('VID/' + image_filename)):
                ann = annopath.format('VID/' + image_filename)
            else:
                ann = annopath.format('DET/' + image_filename)
            recs.append(parse_vid_rec(ann, classhash, gt_img_ids[ind]))
            if ind % 100 == 0:
                print 'reading annotations for {:d}/{:d}'.format(ind + 1, len(img_basenames))
        print 'saving annotations cache to {:s}'.format(annocache)
        with open(annocache, 'wb') as f:
            cPickle.dump(recs, f, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        with open(annocache, 'rb') as f:
            recs = cPickle.load(f)

    # extract objects in :param classname:
    npos = np.zeros(len(classname_map))
    for rec in recs:
        rec_labels = rec['label']
        for x in rec_labels:
            npos[x] += 1

    # read detections
    with open(detpath, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    img_ids = np.array([int(x[0]) for x in splitlines])
    obj_labels = np.array([int(x[1]) for x in splitlines])
    obj_confs = np.array([float(x[2]) for x in splitlines])
    obj_bboxes = np.array([[float(z) for z in x[3:]] for x in splitlines])

    # Rudy: match downsample images and re-normalize confidence's scale (for mAP)
    # if optimal_path != '':
    #     print optimal_path
    #     assert os.path.exists(optimal_path), '%s does not exist'.format(optimal_path)
    #     with open(optimal_path) as f:
    #         optimals = np.array(cPickle.load(f))
        
    # # numberOfResolutions x 2
    # resolutions = np.unique(optimals, axis=0)
    # num_of_res = resolutions.shape[0]

    # # resolutions per bbox
    # res_per_bbox = optimals[img_ids]

    # means = np.zeros(num_of_res)
    # stds = np.zeros(num_of_res)

    # for i in range(num_of_res):
    #     ind = res_per_bbox == resolutions[i]
    #     ind = np.logical_and.reduce(ind, axis=1)
    #     conf = obj_confs[ind]
    #     conf.sort()
    #     means[i] = np.mean(conf[-int(0.1*conf.shape[0]):])
    #     stds[i] = np.std(conf[-int(0.1*conf.shape[0]):])
    #     print 'For Res{}, Mean: {}, Std: {}'.format(i, means[i], stds[i])
    #     obj_confs[ind] = (obj_confs[ind] - means[i]) / stds[i]

    # sort by image_id
    if obj_bboxes.shape[0] > 0:
        sorted_inds = np.argsort(img_ids)
        img_ids = img_ids[sorted_inds]
        obj_labels = obj_labels[sorted_inds]
        obj_confs = obj_confs[sorted_inds]
        obj_bboxes = obj_bboxes[sorted_inds, :]

    num_imgs = max(max(gt_img_ids),max(img_ids)) + 1
    obj_labels_cell = [None] * num_imgs
    obj_confs_cell = [None] * num_imgs
    obj_bboxes_cell = [None] * num_imgs
    start_i = 0
    id = img_ids[0]
    for i in range(0, len(img_ids)):
        if i == len(img_ids)-1 or img_ids[i+1] != id:
            conf = obj_confs[start_i:i+1]
            label = obj_labels[start_i:i+1]
            bbox = obj_bboxes[start_i:i+1, :]
            sorted_inds = np.argsort(-conf)

            obj_labels_cell[id] = label[sorted_inds]
            obj_confs_cell[id] = conf[sorted_inds]
            obj_bboxes_cell[id] = bbox[sorted_inds, :]
            if i < len(img_ids)-1:
                id = img_ids[i+1]
                start_i = i+1


    # go down detections and mark true positives and false positives
    tp_cell = [None] * num_imgs
    fp_cell = [None] * num_imgs

    for rec in recs:
        id = rec['img_ids']
        gt_labels = rec['label']
        gt_bboxes = rec['bbox']
        gt_thr = rec['thr']
        num_gt_obj = len(gt_labels)
        gt_detected = np.zeros(num_gt_obj)

        labels = obj_labels_cell[id]
        bboxes = obj_bboxes_cell[id]

        num_obj = 0 if labels is None else len(labels)
        tp = np.zeros(num_obj)
        fp = np.zeros(num_obj)

        for j in range(0,num_obj):
            bb = bboxes[j, :]
            ovmax = -1
            kmax = -1
            for k in range(0,num_gt_obj):
                if labels[j] != gt_labels[k]:
                    continue
                if gt_detected[k] > 0:
                    continue
                bbgt = gt_bboxes[k, :]
                bi=[np.max((bb[0],bbgt[0])), np.max((bb[1],bbgt[1])), np.min((bb[2],bbgt[2])), np.min((bb[3],bbgt[3]))]
                iw=bi[2]-bi[0]+1
                ih=bi[3]-bi[1]+1
                if iw>0 and ih>0:            
                    # compute overlap as area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                           (bbgt[2] - bbgt[0] + 1.) * \
                           (bbgt[3] - bbgt[1] + 1.) - iw*ih
                    ov=iw*ih/ua
                    # makes sure that this object is detected according
                    # to its individual threshold
                    if ov >= ovthresh and ov > ovmax:
                        ovmax=ov
                        kmax=k
            if kmax >= 0:
                tp[j] = 1
                gt_detected[kmax] = 1
            else:
                fp[j] = 1

        tp_cell[id] = tp
        fp_cell[id] = fp

    tp_all = np.concatenate([x for x in np.array(tp_cell)[gt_img_ids] if x is not None])
    fp_all = np.concatenate([x for x in np.array(fp_cell)[gt_img_ids] if x is not None])
    obj_labels = np.concatenate([x for x in np.array(obj_labels_cell)[gt_img_ids] if x is not None])
    confs = np.concatenate([x for x in np.array(obj_confs_cell)[gt_img_ids] if x is not None])

    print('Mean/Std for Confidence of TPs: {}/{}'.format(np.mean(confs[tp_all==1]), np.std(confs[tp_all==1])))
    print('Mean/Std for Confidence of FPs: {}/{}'.format(np.mean(confs[fp_all==1]), np.std(confs[fp_all==1])))

    sorted_inds = np.argsort(-confs)
    tp_all = tp_all[sorted_inds]
    fp_all = fp_all[sorted_inds]
    obj_labels = obj_labels[sorted_inds]

    ap = np.zeros(len(classname_map))
    for c in range(1, len(classname_map)):
        # compute precision recall
        fp = np.cumsum(fp_all[obj_labels == c])
        tp = np.cumsum(tp_all[obj_labels == c])
        rec = tp / float(npos[c])
        # avoid division by zero in case first detection matches a difficult ground ruth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        # save data for PR curve
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        AUC = {'mrec': mrec, 'mpre': mpre, 'fp': fp[-1], 'tp': tp[-1]}
        with open('./prcurve/{}.pkl'.format(CLASS_TEXT[c]), 'wb') as f:
            cPickle.dump(AUC, f, protocol=cPickle.HIGHEST_PROTOCOL)

        ap[c] = vid_ap(rec, prec)
    ap = ap[1:]
    return ap
