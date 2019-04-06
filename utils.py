
import os
import time
import torch
import numpy as np
from torch.autograd import Variable

import cv2


def get_class_colors(class_id):
    colordict = {'gray': [128, 128, 128], 'silver': [192, 192, 192], 'black': [0, 0, 0],
                 'maroon': [128, 0, 0], 'red': [255, 0, 0], 'purple': [128, 0, 128], 'fuchsia': [255, 0, 255],
                 'green': [0, 128, 0],
                 'lime': [0, 255, 0], 'olive': [128, 128, 0], 'yellow': [255, 255, 0], 'navy': [0, 0, 128],
                 'blue': [0, 0, 255],
                 'teal': [0, 128, 128], 'aqua': [0, 255, 255], 'orange': [255, 165, 0], 'indianred': [205, 92, 92],
                 'lightcoral': [240, 128, 128], 'salmon': [250, 128, 114], 'darksalmon': [233, 150, 122],
                 'lightsalmon': [255, 160, 122], 'crimson': [220, 20, 60], 'firebrick': [178, 34, 34],
                 'darkred': [139, 0, 0],
                 'pink': [255, 192, 203], 'lightpink': [255, 182, 193], 'hotpink': [255, 105, 180],
                 'deeppink': [255, 20, 147],
                 'mediumvioletred': [199, 21, 133], 'palevioletred': [219, 112, 147], 'coral': [255, 127, 80],
                 'tomato': [255, 99, 71], 'orangered': [255, 69, 0], 'darkorange': [255, 140, 0], 'gold': [255, 215, 0],
                 'lightyellow': [255, 255, 224], 'lemonchiffon': [255, 250, 205],
                 'lightgoldenrodyellow': [250, 250, 210],
                 'papayawhip': [255, 239, 213], 'moccasin': [255, 228, 181], 'peachpuff': [255, 218, 185],
                 'palegoldenrod': [238, 232, 170], 'khaki': [240, 230, 140], 'darkkhaki': [189, 183, 107],
                 'lavender': [230, 230, 250], 'thistle': [216, 191, 216], 'plum': [221, 160, 221],
                 'violet': [238, 130, 238],
                 'orchid': [218, 112, 214], 'magenta': [255, 0, 255], 'mediumorchid': [186, 85, 211],
                 'mediumpurple': [147, 112, 219], 'blueviolet': [138, 43, 226], 'darkviolet': [148, 0, 211],
                 'darkorchid': [153, 50, 204], 'darkmagenta': [139, 0, 139], 'indigo': [75, 0, 130],
                 'slateblue': [106, 90, 205],
                 'darkslateblue': [72, 61, 139], 'mediumslateblue': [123, 104, 238], 'greenyellow': [173, 255, 47],
                 'chartreuse': [127, 255, 0], 'lawngreen': [124, 252, 0], 'limegreen': [50, 205, 50],
                 'palegreen': [152, 251, 152],
                 'lightgreen': [144, 238, 144], 'mediumspringgreen': [0, 250, 154], 'springgreen': [0, 255, 127],
                 'mediumseagreen': [60, 179, 113], 'seagreen': [46, 139, 87], 'forestgreen': [34, 139, 34],
                 'darkgreen': [0, 100, 0], 'yellowgreen': [154, 205, 50], 'olivedrab': [107, 142, 35],
                 'darkolivegreen': [85, 107, 47], 'mediumaquamarine': [102, 205, 170], 'darkseagreen': [143, 188, 143],
                 'lightseagreen': [32, 178, 170], 'darkcyan': [0, 139, 139], 'cyan': [0, 255, 255],
                 'lightcyan': [224, 255, 255],
                 'paleturquoise': [175, 238, 238], 'aquamarine': [127, 255, 212], 'turquoise': [64, 224, 208],
                 'mediumturquoise': [72, 209, 204], 'darkturquoise': [0, 206, 209], 'cadetblue': [95, 158, 160],
                 'steelblue': [70, 130, 180], 'lightsteelblue': [176, 196, 222], 'powderblue': [176, 224, 230],
                 'lightblue': [173, 216, 230], 'skyblue': [135, 206, 235], 'lightskyblue': [135, 206, 250],
                 'deepskyblue': [0, 191, 255], 'dodgerblue': [30, 144, 255], 'cornflowerblue': [100, 149, 237],
                 'royalblue': [65, 105, 225], 'mediumblue': [0, 0, 205], 'darkblue': [0, 0, 139],
                 'midnightblue': [25, 25, 112],
                 'cornsilk': [255, 248, 220], 'blanchedalmond': [255, 235, 205], 'bisque': [255, 228, 196],
                 'navajowhite': [255, 222, 173], 'wheat': [245, 222, 179], 'burlywood': [222, 184, 135],
                 'tan': [210, 180, 140],
                 'rosybrown': [188, 143, 143], 'sandybrown': [244, 164, 96], 'goldenrod': [218, 165, 32],
                 'darkgoldenrod': [184, 134, 11], 'peru': [205, 133, 63], 'chocolate': [210, 105, 30],
                 'saddlebrown': [139, 69, 19],
                 'sienna': [160, 82, 45], 'brown': [165, 42, 42], 'snow': [255, 250, 250], 'honeydew': [240, 255, 240],
                 'mintcream': [245, 255, 250], 'azure': [240, 255, 255], 'aliceblue': [240, 248, 255],
                 'ghostwhite': [248, 248, 255], 'whitesmoke': [245, 245, 245], 'seashell': [255, 245, 238],
                 'beige': [245, 245, 220], 'oldlace': [253, 245, 230], 'floralwhite': [255, 250, 240],
                 'ivory': [255, 255, 240],
                 'antiquewhite': [250, 235, 215], 'linen': [250, 240, 230], 'lavenderblush': [255, 240, 245],
                 'mistyrose': [255, 228, 225], 'gainsboro': [220, 220, 220], 'lightgrey': [211, 211, 211],
                 'darkgray': [169, 169, 169], 'dimgray': [105, 105, 105], 'lightslategray': [119, 136, 153],
                 'slategray': [112, 128, 144], 'darkslategray': [47, 79, 79], 'white': [255, 255, 255]}

    colornames = list(colordict.keys())
    assert (class_id < len(colornames))

    r, g, b = colordict[colornames[class_id]]

    return b, g, r  # for OpenCV

def vertices_reprojection(vertices, rt, k):
    p = np.matmul(k, np.matmul(rt[:3,0:3], vertices.T) + rt[:3,3].reshape(-1,1))
    p[0] = p[0] / (p[2] + 1e-5)
    p[1] = p[1] / (p[2] + 1e-5)
    return p[:2].T

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def do_detect(model, rawimg, intrinsics, bestCnt, conf_thresh, use_gpu=False):
    model.eval()
    t0 = time.time()

    height, width, _ = rawimg.shape

    # scale
    img = cv2.resize(rawimg, (model.width, model.height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

    t1 = time.time()
    if use_gpu:
        img = img.cuda()
    img = Variable(img)
    t2 = time.time()

    out_preds = model(img)

    t3 = time.time()

    predPose = fusion(out_preds, width, height, intrinsics, conf_thresh, 0, bestCnt)

    t4 = time.time()

    if True:
    # if False:
        print('-----------------------------------')
        print(' image to tensor : %f' % (t1 - t0))
        if use_gpu:
            print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('          fusion : %f' % (t4 - t3))
        print('           total : %f' % (t4 - t0))
        print('-----------------------------------')
    return predPose

def fusion(output, width, height, intrinsics, conf_thresh, batchIdx, bestCnt):
    layerCnt = len(output)
    assert(layerCnt == 2)

    cls_confs = output[0][0][batchIdx]
    cls_ids = output[0][1][batchIdx]
    predx = output[1][0][batchIdx]
    predy = output[1][1][batchIdx]
    det_confs = output[1][2][batchIdx]
    keypoints = output[1][3]

    nH, nW, nV = predx.shape
    nC = cls_ids.max() + 1

    outPred = []

    mx = predx.mean(axis=2) # average x positions
    my = predy.mean(axis=2) # average y positions
    mdConf = det_confs.mean(axis=2) # average 2D confidences
    for cidx in range(nC):
        # skip background
        if cidx == 0:
            continue
        foremask = (cls_ids == cidx)
        cidx -= 1

        foreCnt = foremask.sum()
        if foreCnt < 1:
            continue

        xs = predx[foremask]
        ys = predy[foremask]
        ds = det_confs[foremask]
        cs = cls_confs[foremask]
        centerxys = np.concatenate((mx[foremask].reshape(-1,1), my[foremask].reshape(-1,1)), 1)

        # choose the item with maximum detection confidence
        # actually, this will choose only one object instance for each type, this is true for OccludedLINEMOD and YCB-Video dataset
        maxIdx = np.argmax(mdConf[foremask])
        refxys = centerxys[maxIdx].reshape(1,-1).repeat(foreCnt, axis=0)
        selected = (np.linalg.norm(centerxys - refxys, axis=1) < 0.2)

        xsi = xs[selected] * width
        ysi = ys[selected] * height
        dsi = ds[selected]
        csi = cs[selected]

        if csi.mean() < conf_thresh: # valid classification propabilities
            continue

        gridCnt = len(xsi)
        assert(gridCnt > 0)

        # choose best N count
        p2d = None
        p3d = None
        candiBestCnt = min(gridCnt, bestCnt)
        for i in range(candiBestCnt):
            bestGrids = dsi.argmax(axis=0)
            validmask = (dsi[bestGrids, list(range(nV))] > 0.5)
            xsb = xsi[bestGrids, list(range(nV))][validmask]
            ysb = ysi[bestGrids, list(range(nV))][validmask]
            t2d = np.concatenate((xsb.reshape(-1, 1), ysb.reshape(-1, 1)), 1)
            t3d = keypoints[cidx][validmask]
            if p2d is None:
                p2d = t2d
                p3d = t3d
            else:
                p2d = np.concatenate((p2d, t2d), 0)
                p3d = np.concatenate((p3d, t3d), 0)
            dsi[bestGrids, list(range(nV))] = 0

        if len(p3d) < 6:
            continue

        retval, rot, trans, inliers = cv2.solvePnPRansac(p3d, p2d, intrinsics, None, flags=cv2.SOLVEPNP_EPNP)

        if not retval:
            continue

        R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
        T = trans.reshape(-1, 1)
        rt = np.concatenate((R, T), 1)

        outPred.append([cidx, rt, 1, None, None, None, [cidx], -1, [0], [0], None])

    return outPred

def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if len(line) > 0 and line[0] != '#' and '=' in line:
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            options[key] = value
    return options

def save_predictions(imgBaseName, predPose, object_names, outpath):
    for p in predPose:
        id, rt, conf, puv, pxyz, opoint, clsid, partid, cx, cy, layerId = p
        path = outpath + '/' + object_names[int(id)] + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        np.savetxt(path + imgBaseName + '.txt', rt)

def visualize_predictions(predPose, image, vertex, intrinsics):
    height, width, _ = image.shape
    confImg = np.copy(image)
    maskImg = np.zeros((height,width), np.uint8)
    contourImg = np.copy(image)
    for p in predPose:
        outid, rt, conf, puv, pxyz, opoint, clsid, partid, cx, cy, layerId = p

        # show surface reprojection
        maskImg.fill(0)
        if True:
            # if False:
            vp = vertices_reprojection(vertex[outid][:], rt, intrinsics)
            for p in vp:
                if p[0] != p[0] or p[1] != p[1]:  # check nan
                    continue
                maskImg = cv2.circle(maskImg, (int(p[0]), int(p[1])), 1, 255, -1)
                confImg = cv2.circle(confImg, (int(p[0]), int(p[1])), 1, get_class_colors(outid), -1, cv2.LINE_AA)

        # fill the holes
        kernel = np.ones((5,5), np.uint8)
        maskImg = cv2.morphologyEx(maskImg, cv2.MORPH_CLOSE, kernel)
        # find contour
        contours, _ = cv2.findContours(maskImg, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contourImg = cv2.drawContours(contourImg, contours, -1, (255, 255, 255), 4, cv2.LINE_AA) # border
        contourImg = cv2.drawContours(contourImg, contours, -1, get_class_colors(outid), 2, cv2.LINE_AA)

    return contourImg

def transform_pred_pose(pred_dir, object_names, transformations):
    objNameList = [f for f in os.listdir(pred_dir) if os.path.isdir(pred_dir + '/' + f)]
    objNameList.sort()
    for objName in objNameList:
        objId = object_names.index(objName.lower())
        obj_dir = pred_dir + '/' + objName
        filelist = [f for f in os.listdir(obj_dir) if f.endswith('.txt')]
        for f in filelist:
            f = obj_dir + '/' + f
            pred_rt = np.loadtxt(f)
            pred_rt = np.matmul(pred_rt, transformations[objId])
            np.savetxt(f, pred_rt)
    return