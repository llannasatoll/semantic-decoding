#!/usr/bin/env python
import os
import sys
import numpy as np
import argparse
import logging
import torch

from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_story_wordseqs
from utils_decode import get_wr_feature_pair, get_em_feature_pair, get_stim_from_wordslist

def get_beginning_info(model_path, fixed):
    if model_path == 'openai-community/gpt2':
        raise()
        if fixed == 5:
            begin_words =  ["i"," reached"," over"," and"," secretly"," und","id"," my"," seat","belt"," and"," when"," his"]
            begin_time = [0,1,2,3,4,5,5,6,7,7,8,9,10]
        elif fixed == 3:
            begin_words = ["i"," reached"," over"," and"," secretly"," und","id"]
            begin_time = [0,1,2,3,4,5,5]
    elif model_path == 'meta-llama/Meta-Llama-3-8B':
        if fixed == 15:
            # begin_words = ['<|begin_of_text|>', 'i', ' reached', ' over', ' and', ' secretly', ' und', 'id', ' my', ' seat', 'belt', ' and', ' when', ' his']
            begin_words = ['<|begin_of_text|>', 'i', ' reached', ' over', ' and', ' secretly', ' und', 'id', ' my', ' seat', 'belt', ' and', ' when', ' his', ' foot', ' hit', ' the', ' brake', ' at', ' the', ' red', ' light', ' i', ' fl', 'ung', ' open', ' the', ' door', ' and', ' i', ' ran', ' i', ' had', ' no', ' shoes', ' on', ' i', ' was', ' crying', ' i', ' had', ' no']
            begin_time = [0,0,1,2,3,4,5,5,6,7,7,8,9,10,11,12,13,14,15,16,17,18,19,20,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
        elif fixed == 3:
            raise()
            begin_words = ['<|begin_of_text|>', 'i', ' reached', ' over', ' and', ' secretly', ' und', 'id']
            begin_time = [0,0,1,2,3,4,5,5,6,7,7,8,9,10]
    return begin_words, begin_time
    
if __name__ == "__main__":

    logger = logging.getLogger("decode_words")
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--llm", type = str, required = True)
    parser.add_argument("--em_id", type = str, required = True)
    parser.add_argument("--wr_id", type = str, required = True)
    parser.add_argument("--num_chance", type = int, default = 0)
    args = parser.parse_args()

    test_stories = ['wheretheressmoke']
    em_data = np.load(config.RESULT_DIR+'/%s/test/%s/encoding_model_%s.npz' % (args.subject, args.llm, args.em_id))
    wr_data = np.load(config.RESULT_DIR+'/%s/test/%s/wordrate_model_%s.npz' % (args.subject, args.llm, args.wr_id))
    assert em_data['model_path'] == wr_data['model_path']

    blank = " " if args.llm == "original" else ""
    is_orig = (args.llm == "original")

    gpt = GPT(path = str(em_data['model_path']), device = config.GPT_DEVICE)
    if args.llm != "original":
        gpt.model.generation_config.pad_token_id = gpt.tokenizer.pad_token_id  
        gpt.tokenizer.pad_token_id = gpt.tokenizer.eos_token_id
    features = LMFeatures(model = gpt, layer = em_data['layer'], context_words = -1)

    # Load and construct word rate model
    logger.info("Creating word rate model")
    resp, rate, mean_wordrate = get_wr_feature_pair(args.subject, wr_data['train_stories'], features, str(wr_data['roi']))
    resp_test, rate_test, _ = get_wr_feature_pair(args.subject, test_stories, features, str(wr_data['roi']))
    clf = Ridge(alpha=wr_data['alpha'])
    clf.fit(resp, rate)
    word_rates = clf.predict(resp_test)
    word_rates = word_rates[:, 0] + mean_wordrate
    logger.info("mean corr=%0.5f" % (pearsonr(word_rates, rate_test[:, 0])[0]))

    # Load and construct encoding  model
    logger.info("Creating encoding model")
    tmp = em_data['corr']
    tmp[tmp < tmp[tmp.argsort()[-config.VOXELS]]] = 0
    vox = [i for i in range(tmp.shape[0]) if tmp[i] != 0]
    features.context_words = config.GPT_WORDS
    wordvec, resp = get_em_feature_pair(args.subject, em_data['train_stories'], features, vox)
    wordvec_test, resp_test = get_em_feature_pair(args.subject, test_stories, features, vox)
    # clf = Ridge(alpha=em_data['alpha'][vox])
    clf = Ridge(alpha=464158.8833612772)
    clf.fit(wordvec, resp)
    assert resp_test.shape[0] == len(word_rates), f'{resp_test.shape[0]} != {len(word_rates)}'
    del wordvec, resp
    torch.cuda.empty_cache()

    fixed = 11 # sec
    unfixed_tr = int((17-fixed)/2) + 1
    word_seqs = get_story_wordseqs(test_stories)
    word_rates = np.insert(word_rates, 0, [mean_wordrate for _ in range(unfixed_tr)])
    word_rates = np.array(list(map(round, word_rates)))

    for story in test_stories:
        current_sec = fixed
        for i in range(len(word_seqs[story].data_times)):
            if word_seqs[story].data_times[i] > fixed: break
        if is_orig:
            begin_words = word_seqs[story].data[:i]
            begin_time = list(range(len(begin_words)))
        else:
            begin_words, begin_time = get_beginning_info(str(em_data['model_path']), fixed)
        data_times = np.concatenate([word_seqs[story].data_times[:i][begin_time], np.linspace(fixed, 17, sum(word_rates[:unfixed_tr-1])+1)[:-1]])
        
        candidate = [[(begin_words, 1)]]
        reference = [(word_seqs[story].data[:i], 1)]
        chances = [[(begin_words, 1)] for _ in range(args.num_chance)]
        for i in range(len(word_rates)-unfixed_tr):
            logger.info(f'{i}/{len(word_rates)-unfixed_tr}')
            current_sec += 2
            data_times = np.concatenate([
                data_times, 
                np.linspace(current_sec, current_sec+2, word_rates[i+unfixed_tr-1]+1)[:-1]
            ])
            num_words = sum(word_rates[i:i+unfixed_tr])

            if num_words <= 0: continue
            res = []
            for words_corr_list in candidate:
                words = []
                for ws, _ in words_corr_list:
                    words.extend(ws)
                outputs = gpt.generate(blank.join(words[-1000:]), num_words, config.WIDTH)
                num_oldtok = len(words[-1000:]) if is_orig else len(gpt.tokenizer.encode(blank.join(words[-1000:])))
                for ii, sample_output in enumerate(outputs):
                    tmp = [gpt.vocab[x] if x < len(gpt.vocab) else '<unk>' for x in sample_output[num_oldtok:]] \
                        if is_orig else [gpt.tokenizer.decode(t, skip_special_tokens=True) for t in sample_output[num_oldtok:]]
                    if len(tmp) != num_words:
                        logger.warning(f'lack of num words: {len(tmp)} != {num_words}')
                    del_mat = get_stim_from_wordslist(words+tmp, features, data_times, list(range(current_sec-(fixed-11), current_sec+(19-fixed)+1, 2)))
                    pred = clf.predict(del_mat[-1].reshape(1, -1))[0]
                    res.append(words_corr_list + [(tmp[:word_rates[i]], pearsonr(pred, resp_test[i])[0])])
            
            for chance in chances:
                words = []
                for ws, _ in chance:
                    words.extend(ws)
                outputs = gpt.generate(blank.join(words[-1000:]), num_words, 1, do_sample=True)[0]
                num_oldtok = len(words[-1000:]) if is_orig else len(gpt.tokenizer.encode(blank.join(words[-1000:])))
                tmp = [gpt.vocab[x] if x < len(gpt.vocab) else '<unk>' for x in outputs[num_oldtok:]] \
                    if is_orig else [gpt.tokenizer.decode(t) for t in outputs[num_oldtok:]]
                if len(tmp) != num_words:
                    logger.warning(f'lack of num words (chance): {len(tmp)} != {num_words}')
                    logger.warning(tmp)
                del_mat = get_stim_from_wordslist(words+tmp, features, data_times, list(range(current_sec-(fixed-11), current_sec+(19-fixed)+1, 2)))
                pred = clf.predict(del_mat[-1].reshape(1, -1))[0]
                chance.append((tmp[:word_rates[i]], pearsonr(pred, resp_test[i])[0]))

            candidate = sorted(res, key=lambda x: x[-1][1])[::-1][:config.EXTENSIONS]
            ref_corr = pearsonr(clf.predict(wordvec_test[i].reshape(1, -1))[0], resp_test[i])[0]
            reference.append((
                list(np.array(word_seqs[story].data)[(word_seqs[story].data_times < current_sec) & (word_seqs[story].data_times >= current_sec-(17-fixed))]),
                ref_corr
                ))
            logger.info([can[-1] for can in candidate])
            logger.info(reference[-1])

            save_location = os.path.join(config.RESULT_DIR, args.subject, "decoding", "em%s_wr%s" % (args.em_id, args.wr_id))
            os.makedirs(save_location, exist_ok = True)
            np.savez(os.path.join(save_location, "%s_result_11" % story),
                can_stcs=np.array([list(map(lambda x: blank.join(x[0]).encode(), candidate[i])) for i in range(config.EXTENSIONS)]),
                can_corr=np.array([list(map(lambda x: x[1], candidate[i])) for i in range(config.EXTENSIONS)]),
                ref_stcs=np.array(list(map(lambda x: ' '.join(x[0]).encode(), reference))),
                ref_corr=np.array(list(map(lambda x: x[1], reference))),
                chance_em=np.array([[chances[i][j][1] for i in range(len(chances))] for j in range(len(reference))]) if args.num_chance else None,
                chance_stcs=np.array([list(map(lambda x: blank.join(x[0]).encode(), chances[i])) for i in range(len(chances))]),
                fixed=fixed, k=config.EXTENSIONS, samples=config.WIDTH
            )