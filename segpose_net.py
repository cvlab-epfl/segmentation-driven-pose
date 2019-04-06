import torch
import torch.nn as nn
from darknet import Darknet
from pose_2d_layer import Pose2DLayer
from pose_seg_layer import PoseSegLayer

class SegPoseNet(nn.Module):
    def __init__(self, data_options):
        super(SegPoseNet, self).__init__()

        pose_arch_cfg = data_options['pose_arch_cfg']
        self.width = int(data_options['width'])
        self.height = int(data_options['height'])
        self.channels = int(data_options['channels'])

        self.coreModel = Darknet(pose_arch_cfg, self.width, self.height, self.channels)
        self.segLayer = PoseSegLayer(data_options)
        self.regLayer = Pose2DLayer(data_options)

    def forward(self, x, y = None):
        if self.training:
            pass
        else:
            outlayers = self.coreModel(x)
            out1 = self.segLayer(outlayers[0])
            out2 = self.regLayer(outlayers[1])
            out_preds = [out1, out2]
            return out_preds

    def train(self):
        self.coreModel.train()
        self.segLayer.train()
        self.regLayer.train()
        self.training = True

    def eval(self):
        self.coreModel.eval()
        self.segLayer.eval()
        self.regLayer.eval()
        self.training = False

    def print_network(self):
        self.coreModel.print_network()

    def load_weights(self, weightfile):
        self.coreModel.load_state_dict(torch.load(weightfile))

    def save_weights(self, weightfile):
        torch.save(self.coreModel.state_dict(), weightfile)