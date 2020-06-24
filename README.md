# Adaptive Object Detection

Adaptive approximation for video object detection.

This is the codebase for the paper [AdaScale: Towards Real-time Video Object Detection using Adaptive Scaling](https://proceedings.mlsys.org/book/275.pdf)

## Usage

To use the current working version of inference loss.

    python experiment/rfcn/rfcn_testloss.py --cfg experiment/rfcn/cfg/rfcn_vid_demo.yaml

Please remember to put ImageNet dataset under `/data/imagenet/ILSVRC`
Also, please put the trained weights under `./output/rfcn/imagenet_vid/rfcn_vid_demo/DET_train_30classes_VID_train_15frames/rfcn_vid-0000.params`
(What I did is symlink `rfcn_vid-0000.params` from `/data/models/dff_mxnet/rfcn_vid-0000.params`)

# Citation

If you find this repository useful for your research, please consider citing

    @incollection{mlsys2019_209,
    author = {Chin, Ting-Wu and Ding, Ruizhou and Marculescu, Diana},
    booktitle = {Proceedings of Machine Learning and Systems 2019},
    pages = {431--441},
    title = {AdaScale: Towards Real-time Video Object Detection using Adaptive Scaling},
    year = {2019}
    }


# Disclaimer

This repository is based on [Deep Feature Flow for Video Recognition](https://github.com/msracver/Deep-Feature-Flow). We do not own those code written by previous contributers.

The code is not tested and is used for references. The code for AdaScale is under `rfcn/`
