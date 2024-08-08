import os
import numpy as np
import json
import random
import itertools as itools

import config

from utils_ridge.DataSequence import DataSequence
from StimulusModel import LMFeatures

from utils_stim import get_stim, get_story_wordseqs, get_punc_script
from utils_resp import get_resp
from utils_ridge.util import make_delayed
from utils_ridge.interpdata import lanczosinterp2D

def get_wr_feature_pair(subject, stories, features, roi):
    with open(os.path.join(config.DATA_TRAIN_DIR, "ROIs", "%s.json" % subject), "r") as f:
        vox = json.load(f)
    wordseqs = get_story_wordseqs(stories)
    rates = {}
    for story in stories:
        ds = wordseqs[story]
        # wordind2tokind = features.model.get_wordind2tokind(get_punc_script(story))
        # words = DataSequence(np.ones(len(wordind2tokind)), ds.split_inds, ds.data_times[wordind2tokind], ds.tr_times)
        # rates[story] = words.chunksums("lanczos", window = 3)
        _, wordind2tokind = features.make_stim(ds.data)
        words = DataSequence(np.ones(len(wordind2tokind)), ds.split_inds, ds.data_times[wordind2tokind], ds.tr_times)
        rates[story] = words.chunksums("lanczos", window = 3)
    nz_rate = np.concatenate([rates[story][5+config.TRIM:-config.TRIM] for story in stories], axis = 0)
    nz_rate = np.nan_to_num(nz_rate).reshape([-1, 1])
    mean_rate = np.mean(nz_rate)
    rate = nz_rate - mean_rate

    resp = get_resp(subject, stories, stack = True, vox = vox[roi])
    delresp = make_delayed(resp, config.RESP_DELAYS)
    ind = get_shuffled_ind(rate.shape[0], config.CHUNKLEN)

    return delresp[ind], rate[ind], mean_rate

def get_em_feature_pair(subject, stories, features, vox):
    rstim, tr_stats = get_stim(stories, features)
    rresp = get_resp(subject, stories, stack = True, vox = vox)
    ind = get_shuffled_ind(rresp.shape[0], config.CHUNKLEN)
    
    return rstim[ind], rresp[ind]

def get_shuffled_ind(resplen, chunklen):
    allinds = range(resplen)
    indchunks = list(zip(*[iter(allinds)]*chunklen))
    random.shuffle(indchunks)
    shuffled_inds = list(itools.chain(*indchunks))
    shuffled_inds.extend(list(range(len(shuffled_inds), resplen)))

    assert len(shuffled_inds) == resplen, f'{len(shuffled_inds)} != {resplen}'
    return shuffled_inds

def get_stim_from_wordslist(wordslist, features, data_times, tr_time, tr_stats = None):
    word_mat, wordind2tokind = features.make_stim(wordslist, old_tokeni=True, mark = '')
    # print(wordslist)
    # print("len(data_times): ", len(data_times))
    # print(wordind2tokind)
    word_mean, word_std = word_mat.mean(0), word_mat.std(0)
    try:
        ds_mat = lanczosinterp2D(word_mat, data_times[wordind2tokind], tr_time)
    except:
        print("wordslist :", wordslist)
        print("len(data_times) :", len(data_times))
        print("wordind2tokind :", wordind2tokind)
        raise

    r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
    r_std[r_std == 0] = 1
    ds_mat = np.nan_to_num(np.dot((ds_mat - r_mean), np.linalg.inv(np.diag(r_std))))
    del_mat = make_delayed(ds_mat, config.STIM_DELAYS)

    return del_mat