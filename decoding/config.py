import os
import numpy as np

gpu_device = '0,1'
print("GPU DEVICE NO: ", gpu_device)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device

# paths

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_LM_DIR = os.path.join("/home/anna/semantic-decoding", "data_lm")
DATA_TRAIN_DIR = os.path.join(REPO_DIR, "data_train")
DATA_TEST_DIR = os.path.join(REPO_DIR, "data_test")
MODEL_DIR = os.path.join(REPO_DIR, "models")
RESULT_DIR = '/home/anna/semantic-decoding/results'
SCORE_DIR = os.path.join(REPO_DIR, "scores")

# GPT encoding model parameters

TRIM = 5
STIM_DELAYS = [1, 2, 3, 4]
RESP_DELAYS = [-4, -3, -2, -1]
ALPHAS = np.logspace(4, 8, 10)
NBOOTS = 15
VOXELS = 10000
CHUNKLEN = 40
GPT_LAYER = 9
GPT_WORDS = 20

# decoder parameters

RANKED = True
WIDTH = 100
NM_ALPHA = 2/3
LM_TIME = 8
LM_MASS = 0.9
LM_RATIO = 0.1
EXTENSIONS = 5

# evaluation parameters

WINDOW = 20

# devices

GPT_DEVICE = "cuda"
EM_DEVICE = "cuda"
SM_DEVICE = "cuda"