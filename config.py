# Convolution layers analyzed for similarity analysis.
CONV_LAYERS = sorted(
    ['conv11', 'conv21', 'conv31', 'conv41', 'conv51',
     'conv61', 'conv71', 'conv81', 'conv91', 'conv12',
     'conv22', 'conv32', 'conv42', 'conv52', 'conv62',
     'conv72', 'conv82', 'conv92'])

# Segmentation class considered for predictions.
PREDICT_LAYER_CLASS = 'conv100'

# General.
INPUT_RANGE = [0, 15000]  # limit the number of x within this range of indices
IMG_SIZE = 128  # size of x image to network
CLASSES = 1  # number of segmentation classes
BATCH_NORM = False  # batch normalization (set to True if network was trained with it)
OPERATIONS = ['/Relu:0']  # layer operations after which activation values are extracted (used in feature extraction)
OBJ_OPERATION = "/Conv2D:0"  # layer operation for segmentation considered (used to get binary masks)
CONV_SEGMENTS = ['conv100']  # segmentation class considered in feature value extraction
STAT_TYPES = ['mean', 'std', 'cv']  # stat_types of statistics calculated per feature
GPU_MEM_FRACTION = 0.9  # fraction of memory used when processing with the network
MODEL_PATH = "checkpoints\\(shoes)\\train_rsp0.5"  # model folder to restore
MODEL = "segmentation.ckpt-1900"  # weights to restore (in model folder)

# Data paths.
IMG_PATH = "./datasets/shoes_selected"
PREDICT_PATH = "./predictions"
OBJ_PATH = "./objects"
OBJ_ATTR_SUFFIX = "/attributes"
OBJ_GRAPH_SUFFIX = "/graphs"
OBJ_FUZZY_SUFFIX = "/fuzzy_similarity"
OUT_STAT_SUFFIX = "/statistics"

# Cluster of data manually defined given dataset indices.
clusters = {}
clusters['1'] = [list(range(0, 15))]
clusters['2'] = [list(range(60, 75))]
clusters['3'] = [list(range(140, 155))]
clusters['4'] = [list(range(200, 215))]
clusters['5'] = [list(range(270, 285))]
clusters['6'] = [list(range(320, 335))]
clusters['7'] = [list(range(380, 395))]
clusters['8'] = [list(range(420, 435))]
clusters['9'] = [list(range(480, 495))]

# Colors per group of clusters.
# Note: if e.g., clusters['1'] has 2 groups (2 lists), then
# ['color1', 'color2'] should be considered.
cluster_colors = [['royalblue'],
          ['crimson'],  # write ['crimson', 'red'] instead if clusters['2'] = [[...], [...]]
          ['mediumseagreen'],
          ['black'],
          ['lightsteelblue'],
          ['lightcoral'],
          ['mediumaquamarine'],
          ['lightsalmon'],
          ['slategrey']]
