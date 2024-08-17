import os
import sys
import numpy as np
import json
import argparse
import logging

import config
from GPT import GPT
from StimulusModel import LMFeatures

from utils_stim import get_story_wordseqs
from utils_resp import get_resp
from utils_ridge.DataSequence import DataSequence
from utils_ridge.util import make_delayed
from utils_ridge.ridge import bootstrap_ridge
np.random.seed(42)

if __name__ == "__main__":

    logger = logging.getLogger("train_WR")
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--llm", type = str, required = True)
    parser.add_argument("--sessions", nargs = "+", type = int, 
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    args = parser.parse_args()

    # training stories
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f) 
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])
    test_stories = ['wheretheressmoke']

    # ROI voxels
    with open(os.path.join(config.DATA_TRAIN_DIR, "ROIs", "%s.json" % args.subject), "r") as f:
        vox = json.load(f)

    # estimate word rate model
    save_location = os.path.join(config.MODEL_DIR, args.subject)
    os.makedirs(save_location, exist_ok = True)

    gpt = GPT(path = config.MODELS[args.llm], device = config.GPT_DEVICE)
    features = LMFeatures(model = gpt, layer = 0, context_words = -1)

    wordseqs = get_story_wordseqs(stories)
    rates = {}
    for story in stories:
        ds = wordseqs[story]
        _, wordind2tokind = features.make_stim(ds.data)
        words = DataSequence(np.ones(len(wordind2tokind)), ds.split_inds, ds.data_times[wordind2tokind], ds.tr_times)
        rates[story] = words.chunksums("lanczos", window = 3)
    nz_rate = np.concatenate([rates[story][5+config.TRIM:-config.TRIM] for story in stories], axis = 0)
    nz_rate = np.nan_to_num(nz_rate).reshape([-1, 1])
    mean_rate = np.mean(nz_rate)
    rate = nz_rate - mean_rate

    # rate for test
    wordseqs = get_story_wordseqs(test_stories)
    rates = {}
    for story in test_stories:
        ds = wordseqs[story]
        _, wordind2tokind = features.make_stim(ds.data)
        words = DataSequence(np.ones(len(wordind2tokind)), ds.split_inds, ds.data_times[wordind2tokind], ds.tr_times)
        rates[story] = words.chunksums("lanczos", window = 3)
    nz_rate = np.concatenate([rates[story][5+config.TRIM:-config.TRIM] for story in test_stories], axis = 0)
    nz_rate = np.nan_to_num(nz_rate).reshape([-1, 1])
    mean_rate = np.mean(nz_rate)
    logger.info(mean_rate)
    rate_test = nz_rate - mean_rate

    for roi in ["auditory"]:
        resp = get_resp(args.subject, stories, stack = True, vox = vox[roi])
        resp_test = get_resp(args.subject, test_stories, stack = True, vox = vox[roi])
        delresp = make_delayed(resp, config.RESP_DELAYS)
        delresp_test = make_delayed(resp_test, config.RESP_DELAYS)
        nchunks = int(np.ceil(delresp.shape[0] / 5 / config.CHUNKLEN))    
        
        logger.info("len(stories) : %d", len(stories))
        logger.info("delresp shape : %s", str(delresp.shape))
        logger.info("rate shape : %s", str(rate.shape))
        logger.info("chunklen : %d", config.CHUNKLEN)
        logger.info("nchunks : %d", nchunks)

        Rcorr, valphas, timestamp = bootstrap_ridge(
            delresp, rate, delresp_test, rate_test, use_corr = False, alphas = config.ALPHAS, 
            nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks, logger = logger
        )
        save_location = os.path.join(config.RESULT_DIR, args.subject, "test", args.llm)
        os.makedirs(save_location, exist_ok = True)
        np.savez(
            os.path.join(save_location, "wordrate_model_%s" % timestamp),
            corr = Rcorr, alpha=valphas, train_stories=stories, roi=roi, model_path=config.MODELS[args.llm]
        )