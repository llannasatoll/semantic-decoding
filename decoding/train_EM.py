import os
import sys
import numpy as np
import json
import argparse
import logging

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp
from utils_ridge.ridge import ridge, bootstrap_ridge

import datetime
from scipy.stats import pearsonr

np.random.seed(42)

extracted_stories = [
    'alternateithicatom',
    'souls',
    'legacy',
    'avatar',
    'odetostepfather',
    'undertheinfluence',
    'howtodraw',
    'myfirstdaywiththeyankees',
    'naked',
    'life',
    'stagefright',
    'tildeath',
    'fromboyhoodtofatherhood',
    'exorcism',
    'sloth',
    'haveyoumethimyet',
    'adollshouse',
    'inamoment',
    'adventuresinsayingyes',
    'theclosetthatateeverything',
    'buck',
    'swimmingwithastronauts',
    'eyespy',
    'thatthingonmyarm',
    'itsabox',
    'hangtime',
]

if __name__ == "__main__":

    logger = logging.getLogger("train_EM")
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--gpt", type = str, default = "perceived")
    parser.add_argument("--sessions", nargs = "+", type = int, 
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    args = parser.parse_args()

    # training stories
    stories = []
    stories = extracted_stories
    test_stories = ['wheretheressmoke']
    # with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
    #     sess_to_story = json.load(f)
    # for sess in args.sessions:
    #     stories.extend(sess_to_story[str(sess)])

    # load gpt
    model_path = 'eng1000'
    # model_path = 'openai-community/gpt2'
    # model_path = 'meta-llama/Meta-Llama-3-8B'
    gpt = GPT(path = model_path, device = config.GPT_DEVICE)
    features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)

    # estimate encoding model
    rstim, tr_stats = get_stim(stories, features)
    rresp = get_resp(args.subject, stories, stack = True)
    nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))
    pstim, tr_stats = get_stim(test_stories, features)
    presp = get_resp(args.subject, test_stories, stack = True)

    logger.info("len(stories) : %d", len(stories))
    logger.info("rstim shape : %s", str(rstim.shape))
    logger.info("rresp shape : %s", str(rresp.shape))
    logger.info("pstim shape : %s", str(pstim.shape))
    logger.info("presp shape : %s", str(presp.shape))
    logger.info("nchunks shape : %d", nchunks)

    Rcorr, valphas, timestamp = bootstrap_ridge(rstim, rresp, pstim, presp, use_corr = False, alphas = config.ALPHAS,
        nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks, logger = logger)

    save_location = os.path.join(config.RESULT_DIR, args.subject)
    os.makedirs(save_location, exist_ok = True)
    np.savez(os.path.join(save_location, "encoding_model_result_%s" % timestamp),
        corr = Rcorr, stories = stories, alpha=valphas, model_path=model_path,
        layer=config.GPT_LAYER, train_stories=stories, test_stories=test_stories)
    exit()
    '''
    bscorrs = bscorrs.mean(2).max(0)
    vox = np.sort(np.argsort(bscorrs)[-config.VOXELS:])
    del rstim, rresp
    
    # estimate noise model
    stim_dict = {story : get_stim([story], features, tr_stats = tr_stats) for story in stories}
    resp_dict = get_resp(args.subject, stories, stack = False, vox = vox)
    noise_model = np.zeros([len(vox), len(vox)])
    for hstory in stories:
        tstim, hstim = np.vstack([stim_dict[tstory] for tstory in stories if tstory != hstory]), stim_dict[hstory]
        tresp, hresp = np.vstack([resp_dict[tstory] for tstory in stories if tstory != hstory]), resp_dict[hstory]
        bs_weights = ridge(tstim, tresp, alphas[vox])
        resids = hresp - hstim.dot(bs_weights)
        bs_noise_model = resids.T.dot(resids)
        noise_model += bs_noise_model / np.diag(bs_noise_model).mean() / len(stories)
    del stim_dict, resp_dict
    
    # save
    save_location = os.path.join(config.MODEL_DIR, args.subject)
    os.makedirs(save_location, exist_ok = True)
    np.savez(os.path.join(save_location, "encoding_model_%s" % args.gpt), 
        weights = weights, noise_model = noise_model, alphas = alphas, voxels = vox, stories = stories,
        tr_stats = np.array(tr_stats), word_stats = np.array(word_stats))
    '''