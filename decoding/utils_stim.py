import os
import numpy as np
import json
import pickle

import config
from utils_ridge.stimulus_utils import TRFile, load_textgrids, load_simulated_trfiles
from utils_ridge.dsutils import make_word_ds
from utils_ridge.interpdata import lanczosinterp2D
from utils_ridge.util import make_delayed
from feature_spaces import get_feature_space

def get_story_wordseqs(stories):
    """loads words and word times of stimulus stories
    """
    grids = load_textgrids(stories, config.DATA_TRAIN_DIR)
    with open(os.path.join(config.DATA_TRAIN_DIR, "respdict.json"), "r") as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict)
    wordseqs = make_word_ds(grids, trfiles)
    return wordseqs

def get_punc_script(story):
    return pickle.load(open(os.path.join(config.DATA_TRAIN_DIR, "train_stimulus", "script_with_punctuation", story+".pickle"), 'rb'))

def get_embedding(stories, embedmodel, tr_stats = False):
    ds_mat = np.vstack([
        np.load(os.path.join(config.DATA_TRAIN_DIR, "train_stimulus", "embedding_-3to-1", embedmodel, story) + '.npy') 
        for story in stories
    ])
    ds_mat_late = np.vstack([
        np.load(os.path.join(config.DATA_TRAIN_DIR, "train_stimulus", "embedding_-9to-1", embedmodel, story) + '.npy') 
        for story in stories
    ])
    del_mat = np.hstack([ds_mat, np.roll(ds_mat, 1, axis=0), np.roll(ds_mat, 2, axis=0), np.roll(ds_mat, 3, axis=0), ds_mat_late])
    # del_mat = np.hstack([ds_mat, np.roll(ds_mat, 1, axis=0), np.roll(ds_mat, 2, axis=0), np.roll(ds_mat, 3, axis=0)])
    return del_mat
    # return  ds_mat
    # return  np.roll(del_mat, 1, axis=0)

def get_stim(stories, features, tr_stats = None, old_tokeni=True):
    """extract quantitative features of stimulus stories
    """
    if features.model.path == 'eng1000':
        ds_vecs = get_feature_space('eng1000', stories) # add sentence
    else:
        word_seqs = get_story_wordseqs(stories)
        if old_tokeni:
            word_vecs, wordind2tokind = {}, {}
            for story in stories:
                word_vecs[story], wordind2tokind[story] =  features.make_stim(word_seqs[story].data, old_tokeni=old_tokeni)
        else:
            wordind2tokind = {story: features.model.get_wordind2tokind(get_punc_script(story)) for story in stories}
            word_vecs = {story : features.make_stim(get_punc_script(story), mark = ' ') for story in stories}
        ds_vecs = {
            story : lanczosinterp2D(word_vecs[story], word_seqs[story].data_times[wordind2tokind[story]], word_seqs[story].tr_times)
            for story in stories
        }

    ds_mat = np.vstack([ds_vecs[story][5+config.TRIM:-config.TRIM] for story in stories])
    if tr_stats is None:
        r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
        r_std[r_std == 0] = 1
    else:
        r_mean, r_std = tr_stats
    ds_mat = np.nan_to_num(np.dot((ds_mat - r_mean), np.linalg.inv(np.diag(r_std))))
    del_mat = make_delayed(ds_mat, config.STIM_DELAYS)
    if tr_stats is None: return del_mat, (r_mean, r_std)
    else: return del_mat

def predict_word_rate(resp, wt, vox, mean_rate):
    """predict word rate at each acquisition time
    """
    delresp = make_delayed(resp[:, vox], config.RESP_DELAYS)
    rate = ((delresp.dot(wt) + mean_rate)).reshape(-1).clip(min = 0)
    return np.round(rate).astype(int)

def predict_word_times(word_rate, resp, starttime = 0, tr = 2):
    """predict evenly spaced word times from word rate
    """
    half = tr / 2
    trf = TRFile(None, tr)
    trf.soundstarttime = starttime
    trf.simulate(resp.shape[0])
    tr_times = trf.get_reltriggertimes() + half

    word_times = []
    for mid, num in zip(tr_times, word_rate):  
        if num < 1: continue
        word_times.extend(np.linspace(mid - half, mid + half, num, endpoint = False) + half / num)
    return np.array(word_times), tr_times