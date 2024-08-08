import os
import sys
import numpy as np
import json
import argparse
import logging

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim, get_embedding
from utils_resp import get_resp
from utils_ridge.ridge import ridge, bootstrap_ridge

np.random.seed(42)
models = {
    'eng1000' : 'eng1000',
    'gpt2' : 'openai-community/gpt2',
    'llama3' : 'meta-llama/Meta-Llama-3-8B',
    'embed_small' : 'text-embedding-3-small',
    'embed_large' : 'text-embedding-3-large',
    'original' : 'original',
}

if __name__ == "__main__":

    logger = logging.getLogger("train_EM")
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--llm", type = str, required = True)
    parser.add_argument("--embedding", action='store_true')
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

    if args.embedding:
        logger.info(f"Get embedding : {models[args.llm]}")
        rstim = get_embedding(stories, models[args.llm])
        pstim = get_embedding(test_stories, models[args.llm])
        rstim = (rstim - rstim.mean(axis=0)) / rstim.std(axis=0)
        pstim = (pstim - rstim.mean(axis=0)) / rstim.std(axis=0)
    else:
        # load gpt
        gpt = GPT(path = models[args.llm], device = config.GPT_DEVICE) # もどす
        features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
        rstim, tr_stats = get_stim(stories, features)
        pstim, tr_stats = get_stim(test_stories, features)

    rresp = get_resp(args.subject, stories, stack = True)
    nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))
    presp = get_resp(args.subject, test_stories, stack = True)

    logger.info("len(stories) : %d", len(stories))
    logger.info("rstim shape : %s", str(rstim.shape))
    logger.info("rresp shape : %s", str(rresp.shape))
    logger.info("pstim shape : %s", str(pstim.shape))
    logger.info("presp shape : %s", str(presp.shape))
    logger.info("chunklen : %d", config.CHUNKLEN)
    logger.info("nchunks : %d", nchunks)
    # estimate encoding model
    Rcorr, valphas, timestamp = bootstrap_ridge(rstim, rresp, pstim, presp, use_corr = False, alphas = config.ALPHAS,
        nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks, logger = logger)

    save_location = os.path.join(config.RESULT_DIR, args.subject, "test", args.llm)
    os.makedirs(save_location, exist_ok = True)
    np.savez(os.path.join(save_location, "encoding_model_%s" % timestamp),
        corr = Rcorr, alpha=valphas, model_path=models[args.llm],
        layer=config.GPT_LAYER, train_stories=stories)