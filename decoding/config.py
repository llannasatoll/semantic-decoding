import os
import numpy as np
import socket

HOSTNAME = socket.gethostname() 
# gpu_device = "0"
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
print("GPU DEVICE NO: ", os.environ['CUDA_VISIBLE_DEVICES'] )

# paths

HOME_DIR = '/home/anna/semantic-decoding'
REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if HOSTNAME in ["e7c21511f574"]:
    DATA_LM_DIR = os.path.join(REPO_DIR, "data_lm")
    DATA_TRAIN_DIR = os.path.join(REPO_DIR, "data_train")
else:
    DATA_LM_DIR = os.path.join(HOME_DIR, "data_lm")
    DATA_TRAIN_DIR = os.path.join(HOME_DIR, "data_train")
DATA_TEST_DIR = os.path.join(REPO_DIR, "data_test")
MODEL_DIR = os.path.join(REPO_DIR, "models")
RESULT_DIR = '/home/anna/semantic-decoding/results'
SCORE_DIR = os.path.join(REPO_DIR, "scores")

# GPT encoding model parameters

TRIM = 5
STIM_DELAYS = [1, 2, 3, 4]
RESP_DELAYS = [-4, -3, -2, -1]
ALPHAS = np.logspace(4, 8, 17)
NBOOTS = 15
VOXELS = 10000
CHUNKLEN = 40
GPT_LAYER = 9
GPT_WORDS = 20

# decoder parameters

RANKED = True
WIDTH = 30
NM_ALPHA = 2/3
LM_TIME = 8
LM_MASS = 0.9
LM_RATIO = 0.1
EXTENSIONS = 10

# evaluation parameters

WINDOW = 20

# devices

if HOSTNAME in ["thales", "fermat"]:
    GPT_DEVICE = "cpu"
else:
    GPT_DEVICE = "cuda"
EM_DEVICE = "cuda"
SM_DEVICE = "cuda"

MODELS = {
    'eng1000' : 'eng1000',
    'gpt2' : 'openai-community/gpt2',
    'llama3' : 'meta-llama/Meta-Llama-3-8B',
    'llama3.1' : 'meta-llama/Meta-Llama-3.1-8B',
    'embed_small' : 'text-embedding-3-small',
    'embed_large' : 'text-embedding-3-large',
    'original' : os.path.join(os.path.join(REPO_DIR, "data_lm"), "perceived", "model"),
    "deberta_xxlarge": "microsoft/deberta-v2-xxlarge",
    "e5" : "intfloat/e5-mistral-7b-instruct",
    "t5_xxlarge" : "sentence-transformers/sentence-t5-xxl",
    "roberta" : "FacebookAI/roberta-large",
    "t5_11b" : "google-t5/t5-11b",
    "opt-13b" : "facebook/opt-13b",
}