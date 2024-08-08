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

# import datetime
# from scipy.stats import pearsonr

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
    # parser.add_argument("--gpt", type = str, default = "perceived")
    parser.add_argument("--new_tokeni", action='store_true')
    parser.add_argument("--sessions", nargs = "+", type = int, 
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    args = parser.parse_args()

    # training stories
    stories = []
    if False:
        stories = extracted_stories
    else:
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
        with open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json"), "r") as f:
            gpt_vocab = json.load(f)
        gpt = GPT(path = os.path.join(config.DATA_LM_DIR, "perceived", "model"), vocab = gpt_vocab, device = config.GPT_DEVICE)
        # gpt = GPT(path = models[args.llm], device = config.GPT_DEVICE) # もどす
        features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
        old_tokeni = not args.new_tokeni
        rstim, tr_stats = get_stim(stories, features, old_tokeni=old_tokeni)
        pstim, tr_stats = get_stim(test_stories, features, old_tokeni=old_tokeni)

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