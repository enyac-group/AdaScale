# Adaptive Object Detection

Adaptive approximation for video object detection.

## Usage

To use the current working version of inference loss.

    python experiment/rfcn/rfcn_testloss.py --cfg experiment/rfcn/cfg/rfcn_vid_demo.yaml

Please remember to put ImageNet dataset under `/data/imagenet/ILSVRC`
Also, please put the trained weights under `./output/rfcn/imagenet_vid/rfcn_vid_demo/DET_train_30classes_VID_train_15frames/rfcn_vid-0000.params`
(What I did is symlink `rfcn_vid-0000.params` from `/data/models/dff_mxnet/rfcn_vid-0000.params`)

# Disclaimer

This repository is based on [Deep Feature Flow for Video Recognition](https://github.com/msracver/Deep-Feature-Flow). We do not own those code written by previous contributers.

The code is not tested and is used for references. The code for AdaScale is under `rfcn/`
