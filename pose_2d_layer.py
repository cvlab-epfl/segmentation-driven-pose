
import torch.nn as nn
from utils import *

class Pose2DLayer(nn.Module):
    def __init__(self, options):
        super(Pose2DLayer, self).__init__()
        self.coord_norm_factor = 10
        self.keypoints = torch.from_numpy(np.load(options['keypointsfile'])).float()
        self.num_keypoints = int(options['num_keypoints'])
        self.keypoints = self.keypoints[:,:self.num_keypoints,:]

    def forward(self, output, target=None, param = None):
        seen = 0
        if param:
            seen = param[0]

        # output : BxAs*(1+2*num_vpoints+num_classes)*H*W
        t0 = time.time()
        nB = output.data.size(0)
        nA = 1
        nV = self.num_keypoints
        nH = output.data.size(2)
        nW = output.data.size(3)

        output = output.view(nB * nA, (3 * nV), nH * nW).transpose(0, 1). \
            contiguous().view((3 * nV), nB * nA * nH * nW)

        conf = torch.sigmoid(output[0:nV].transpose(0, 1).view(nB, nA, nH, nW, nV))
        x = output[nV:2*nV].transpose(0, 1).view(nB, nA, nH, nW, nV)
        y = output[2*nV:3*nV].transpose(0, 1).view(nB, nA, nH, nW, nV)
        t1 = time.time()

        grid_x = ((torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA * nV, 1, 1). \
            view(nB, nA, nV, nH, nW).type_as(output) + 0.5) / nW ) * self.coord_norm_factor
        grid_y = ((torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA * nV, 1, 1). \
            view(nB, nA, nV, nH, nW).type_as(output) + 0.5) / nH) * self.coord_norm_factor
        grid_x = grid_x.permute(0, 1, 3, 4, 2).contiguous()
        grid_y = grid_y.permute(0, 1, 3, 4, 2).contiguous()

        predx = x + grid_x
        predy = y + grid_y

        if self.training:
            pass
        else:
            predx = predx.view(nB, nH, nW, nV) / self.coord_norm_factor
            predy = predy.view(nB, nH, nW, nV) / self.coord_norm_factor

            # copy to CPU
            conf = convert2cpu(conf.view(nB,nH,nW,nV)).detach().numpy()
            px = convert2cpu(predx).detach().numpy()
            py = convert2cpu(predy).detach().numpy()
            keypoints = convert2cpu(self.keypoints).detach().numpy()

            t2 = time.time()

            if False:
                print('---------------------------------')
                print('matrix computation : %f' % (t1 - t0))
                print('        gpu to cpu : %f' % (t2 - t1))
                print('---------------------------------')

            out_preds = [px, py, conf, keypoints]
            return out_preds