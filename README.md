
# Overview

**Please find the latest version at [WDR-Pose](https://github.com/cvlab-epfl/wide-depth-range-pose).**

This repository contains the code for the paper **Segmentation-driven 6D Object Pose Estimation**. [Yinlin Hu](http://yinlinhu.github.io), [Joachim Hugonot](https://people.epfl.ch/joachim.hugonot), [Pascal Fua](http://people.epfl.ch/pascal.fua), [Mathieu Salzmann](http://people.epfl.ch/mathieu.salzmann). CVPR. 2019. [\[Paper\]](https://arxiv.org/abs/1812.02541)

The most recent trend in estimating the 6D pose of rigid objects has been to train deep networks to either directly regress the pose from the image or to predict the 2D locations of 3D keypoints, from which the pose can be obtained using a PnP algorithm. In both cases, the object is treated as a global entity, and a single pose estimate is computed. As a consequence, the resulting techniques can be vulnerable to large occlusions. 

In this paper, we introduce a segmentation-driven 6D pose estimation framework where each visible part of the objects contributes a local pose prediction in the form of 2D keypoint locations. We then use a predicted measure of confidence to combine these pose candidates into a robust set of 3D-to-2D correspondences, from which a reliable pose estimate can be obtained. We outperform the state-of-the-art on the challenging Occluded-LINEMOD and YCB-Video datasets, which is evidence that our approach deals well with multiple poorly-textured objects occluding each other. Furthermore, it relies on a simple enough architecture to achieve real-time performance.

![](./images/fig1.jpg)

Figure 1: Overall workflow of our method. Our architecture has two streams: One for object segmentation and the other to regress 2D keypoint locations. These two streams share a common encoder, but the decoders are separate. Each one produces a tensor of a spatial resolution that defines an SxS grid over the image. The segmentation stream predicts the label of the object observed at each grid location. The regression stream predicts the 2D keypoint locations for that object. 

![](./images/fig2.jpg)

Figure 2: Occluded-LINEMOD results. In each column, we show, from top to bottom: the foreground segmentation mask, all 2D reprojection candidates, the selected 2D reprojections, and the final pose results. Our method generates accurate pose estimates, even in the presence of large occlusions. Furthermore, it can process multiple objects in real time.


# How to Use

## Step 1

Download the datasets.

Occluded-LINEMOD: https://hci.iwr.uni-heidelberg.de/vislearn/iccv2015-occlusion-challenge/

YCB-Video: https://rse-lab.cs.washington.edu/projects/posecnn/


## Step 2

Download the pretrained model.

Occluded-LINEMOD: https://1drv.ms/u/s!ApOY_gOHw8hLbbdmVZgnqk30I5A 

YCB-Video: https://1drv.ms/u/s!ApOY_gOHw8hLbLl4i8CAXD6LGuU

Download and put them into **./model** directory.

*Due to commercial problem, we can only provide the code for inference. However, it is straightforward to implement the training part according to our [paper](https://arxiv.org/abs/1812.02541) and this repository.*

## Step 3

Prepare the input file list using **gen_filelist.py**.


## Step 4

Run **test.py** and explore it.

# Citing

```
@inproceedings{hu2019segpose,
  title={Segmentation-driven 6D Object Pose Estimation},
  author={Yinlin Hu and Joachim Hugonot and Pascal Fua and Mathieu Salzmann},
  booktitle={CVPR},
  year={2019}
}
```
