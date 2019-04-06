from utils import *
from segpose_net import SegPoseNet
import cv2

def evaluate(data_cfg, weightfile, listfile, outdir, object_names, intrinsics, vertex,
                         bestCnt, conf_thresh, linemod_index=False, use_gpu=False, gpu_id='0'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data_options = read_data_cfg(data_cfg)
    m = SegPoseNet(data_options)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        m.cuda()

    with open(listfile, 'r') as file:
        imglines = file.readlines()

    for idx in range(len(imglines)):
        imgfile = imglines[idx].rstrip()
        img = cv2.imread(imgfile)

        dirname, filename = os.path.split(imgfile)
        baseName, _ = os.path.splitext(filename)
        if linemod_index:
            outFileName = baseName[-4:]
        else:
            dirname = os.path.splitext(dirname[dirname.rfind('/') + 1:])[0]
            outFileName = dirname+'_'+baseName

        start = time.time()
        predPose = do_detect(m, img, intrinsics, bestCnt, conf_thresh, use_gpu)
        finish = time.time()

        arch = 'CPU'
        if use_gpu:
            arch = 'GPU'
        print('%s: Predict %d objects in %f seconds (on %s).' % (imgfile, len(predPose), (finish - start), arch))
        save_predictions(outFileName, predPose, object_names, outdir)

        # visualize predictions
        vis_start = time.time()
        visImg = visualize_predictions(predPose, img, vertex, intrinsics)
        cv2.imwrite(outdir + '/' + outFileName + '.jpg', visImg)
        vis_finish = time.time()
        print('%s: Visualization in %f seconds.' % (imgfile, (vis_finish - vis_start)))

if __name__ == '__main__':
    # use_gpu = True
    use_gpu = False
    #
    dataset = 'Occluded-LINEMOD'
    outdir = './Occluded-LINEMOD-Out'
    #
    # dataset = 'YCB-Video'
    # outdir = './YCB-Video-Out'
    #
    if dataset == 'Occluded-LINEMOD':
        # intrinsics of LINEMOD dataset
        k_linemod = np.array([[572.41140, 0.0, 325.26110],
                              [0.0, 573.57043, 242.04899],
                              [0.0, 0.0, 1.0]])
        # 8 objects for LINEMOD dataset
        object_names_occlinemod = ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']
        vertex_linemod = np.load('./data/Occluded-LINEMOD/LINEMOD_vertex.npy')
        evaluate('./data/data-LINEMOD.cfg',
                             './occluded-linemod.pth',
                             '/data/OcclusionChallengeICCV2015/test.txt',
                             outdir, object_names_occlinemod, k_linemod, vertex_linemod,
                             bestCnt=10, conf_thresh=0.3, linemod_index=True, use_gpu=use_gpu)
        #
        rt_transforms = np.load('./data/Occluded-LINEMOD/Transform_RT_to_OccLINEMOD_meshes.npy')
        transform_pred_pose(outdir, object_names_occlinemod, rt_transforms)
    elif dataset == 'YCB-Video':
        # intrinsics of YCB-VIDEO dataset
        k_ycbvideo = np.array([[1.06677800e+03, 0.00000000e+00, 3.12986900e+02],
                               [0.00000000e+00, 1.06748700e+03, 2.41310900e+02],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # 21 objects for YCB-Video dataset
        object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                                 '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                                 '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                                 '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
        vertex_ycbvideo = np.load('./data/YCB-Video/YCB_vertex.npy')
        evaluate('./data/data-YCB.cfg',
                             './ycb-video.pth',
                             '/data/YCB_Video_Dataset/test.txt',
                             outdir, object_names_ycbvideo, k_ycbvideo, vertex_ycbvideo,
                             bestCnt=10, conf_thresh=0.3, use_gpu=use_gpu)
    else:
        print('unsupported dataset \'%s\'.' % dataset)