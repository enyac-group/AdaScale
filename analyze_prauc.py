import sys
import cPickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# candidates = ['singlescale_600', 'multiscale_4scales', 'adaptive_5scales', 'random', 'optimal']
candidates = ['singlescale_600', 'multiscale_600', 'multiscale_4scales', 'adaptive_5scales', 'random']
candidate_name = ['$M_a$', '$M_b$', '$M_c$', 'AdaScale', 'Random']
colors = ['b', 'orange', 'g', 'r', '#885EAD', 'c', 'r']

CLASS_TEXT = ['airplane', 'antelope', 'bear', 'bicycle',
            'bird', 'bus', 'car', 'cattle',
            'dog', 'domestic_cat', 'elephant', 'fox',
            'giant_panda', 'hamster', 'horse', 'lion',
            'lizard', 'monkey', 'motorcycle', 'rabbit',
            'red_panda', 'sheep', 'snake', 'squirrel',
            'tiger', 'train', 'turtle', 'watercraft',
            'whale', 'zebra']

font = {'family' : 'Times New Roman',
        'size'   : 17}
matplotlib.rc('font', **font)

def draw_auc(prfile):
    """
    Take an auc.pkl as input plot precision recall curve
    Notice that the data structure in auc.pkl should be a dictionary
    dict({'mrec': np.array, 'mpre': np.array})
    Args:
        prfile (str): The path to find input pickle file
    Returns:
        None
    """
    with open(prfile) as f:
        d = cPickle.load(f)
    plt.plot(d['mrec'], d['mpre'], '-o', markevery=[-2])
    return d['tp'], d['fp']

for c in CLASS_TEXT:
    method_bars = []
    # ROC curve
    plt.Figure(dpi=600)
    for i, method in enumerate(candidates):
        tp, fp = draw_auc('{}/prcurve_{}/{}.pkl'.format(sys.argv[1], method, c))
        if i == 0:
            basetp, basefp = tp, fp
        method_bars.append([tp/float(basetp), fp/float(basefp)])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(candidate_name, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
    plt.axis([0, 1, 0, 1])
    # plt.tight_layout()
    plt.savefig('{}/prcurve/{}.png'.format(sys.argv[2], c), bbox_inches='tight')
    plt.close()

    # TP, FP
    plt.Figure(dpi=600)
    for i, method in enumerate(method_bars):
        ind = np.array([i, i+len(candidates)+1])
        plt.bar(ind+0.2, method, label=candidate_name[i], color=colors[i])
    ind = np.array([len(candidates)/2, len(candidates)/2+len(candidates)+1])
    plt.xticks(ind+0.2, ['TP', 'FP'])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
    # plt.tight_layout()
    plt.savefig('{}/tpfp/{}.png'.format(sys.argv[2], c), bbox_inches='tight')
    plt.close()