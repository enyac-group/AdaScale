import cPickle
import numpy as np
import matplotlib.pyplot as plt


root = '/data/adascale_output/rfcn/imagenet_vid/rfcn_testloss/'
t_root = '/data/adascale_output/rfcn/imagenet_vid/resnet_v1_101_scalereg_1x1/'
# t_root = '/data/adascale_output/rfcn/imagenet_vid/rfcn_vid_demo/'
# datasets = ['DET_train_30classes', 'VID_val_frames', 'VID_train_15frames']
net = 'resnet_v1_101_rfcn'
# datasets = ['DET_train_30classes_VID_train_15frames']
datasets = ['MIX_train']
# datasets = ['VID_val_frames']
# resolutions = [(864, 2000), (688, 2000), (576, 2000), (480, 2000), (304, 2000), (128, 2000)]
resolutions = [(600, 2000), (480, 2000), (360, 2000), (240, 2000), (128, 2000)]
# resolutions = [(600, 2000), (540, 2000), (480, 2000), (420, 2000), (360, 2000), (300, 2000), (240, 2000), (180, 2000), (128, 2000)]

def dedup(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

for dataset in datasets:
    output = []
    loss_mat = []
    ind_to_frameid = []
    for res in resolutions:
        filename = '{}/{}/{}_loss_{}_{}.pkl'.format(root, dataset, net, res[0], res[1])
        with open(filename) as f:
            tmp = cPickle.load(f)
            output.append(tmp)
    
    N = len(output[0]) # Size of the dataset
    m = len(output) # Number of resolutions

    optimals = {} # A dictionary from frame_id to optimal resolutions
    optimal_hist = np.zeros(m) # For each kappa, check the size distribution

    fg = np.zeros((N, m)) # Use to analyze the number of foreground for each image in each resolution
    bg = np.zeros((N, m)) # Use to analyze the number of background for each image in each resolution
    t = np.zeros((4, N, m)) # The regressed axis of bounding boxes
    fg_cls_loss = np.zeros((N, m))
    bg_cls_loss = np.zeros((N, m))
    reg_loss = np.zeros((N, m))

    for i in range(N):
        # gathering stats
        for j in np.arange(m-1, -1, -1):
            fg_idx = (output[j][i]['label'] != 0)
            fg[i][j] = np.sum(fg_idx)
            bg[i][j] = output[j][i]['label'].shape[0] - fg[i][j]
            if fg[i][j] > 0:
                fg_cls_loss[i][j] = np.mean(output[j][i]['cls_loss'][fg_idx])
            if bg[i][j] > 0:
                bg_cls_loss[i][j] = np.mean(output[j][i]['cls_loss'][~fg_idx])
            reg_loss[i][j] = np.mean(output[j][i]['bbox_loss'])
            for k in range(4):
                t[k][i][j] = np.mean(np.abs(output[j][i]['bbox_pred'][:, 4+k]))
        # choose the smallest foreground number that is not 0
        total_loss = np.zeros(m)
        cur_fg = fg[i].copy()
        total_loss[cur_fg == 0] = np.inf
        cur_fg[cur_fg == 0] = np.inf
        fg_num = np.min(cur_fg)

        # Sort loss for each image, pick the smallest `fg_num` loss for all resolutions
        if fg_num < np.inf:
            for j in range(m):
                # Only resolutions with foreground will be computed
                if total_loss[j] == 0:
                    fg_idx = (output[j][i]['label'] != 0)
                    fg_bbox_loss = np.sum(output[j][i]['bbox_loss'][fg_idx], axis=1)
                    fg_all_loss = fg_bbox_loss + output[j][i]['cls_loss'][fg_idx]
                    # fg_all_loss = dedup(fg_all_loss)
                    inds = np.argsort(fg_all_loss)
                    total_loss[j] = np.sum(fg_all_loss[inds[:int(fg_num)]])

        # Policy: Pick the smallest loss
        idx = np.argmin(total_loss)

        # if idx == len(resolutions)-1:
        #     print('cur_fg: {}'.format(cur_fg))
        #     for j in range(m):
        #         fg_idx = (output[j][i]['label'] != 0)
        #         fg_loss = np.sum(output[j][i]['bbox_loss'][fg_idx], axis=1) + output[j][i]['cls_loss'][fg_idx]
        #         fg_loss.sort()
        #         print('fg_loss (m={}): {}'.format(j, fg_loss)) 
        #     break

        optimals[int(output[0][i]['frame_id'])] = resolutions[idx]
        optimal_hist[idx]+=1

    print('Resolutions {}'.format([res[0] for res in resolutions]))
    print('Avg. fg number: {}'.format(np.mean(fg, axis=0)))
    print('Avg. bg number: {}'.format(np.mean(bg, axis=0)))
    print('Avg. t_x, t_y, t_w, t_h: {}'.format(np.mean(t, axis=1)))
    print('Avg. fg_cls_loss: {}'.format(np.mean(fg_cls_loss, axis=0)))
    print('Avg. bg_cls_loss: {}'.format(np.mean(bg_cls_loss, axis=0)))
    print('Avg. reg_loss: {}'.format(np.mean(reg_loss, axis=0)))

    dist = optimal_hist / float(N)
    print('Hist: {}'.format(dist))
    normalized_latency = np.array([(float(res[0])/resolutions[0][0])**2 for res in resolutions])
    print('Latency: {}'.format(np.matmul(dist, normalized_latency)))
    target = '{}/{}/{}_optimal.pkl'.format(t_root, dataset, net)
    with open(target, 'wb') as fid:
        cPickle.dump(optimals, fid, cPickle.HIGHEST_PROTOCOL)

    # xticks_label = ['({},{})'.format(resolutions[i][0], resolutions[i][1]) for i in range(m)]
    #plt.figure(dpi=300)
    #plt.bar(np.arange(m), dist)
    #plt.xticks(np.arange(m), xticks_label)
    #plt.xlabel('Resolution')
    #plt.ylabel('Percentage of Images')
    #plt.ylim([0, 1])
    #plt.savefig('{}.png'.format(dataset))