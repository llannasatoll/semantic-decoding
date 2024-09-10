#!/usr/bin/env python
import os
import sys
import numpy as np
import argparse
import logging
import torch
import copy

from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_story_wordseqs
from utils_decode import get_wr_feature_pair, get_em_feature_pair, get_stim_from_wordslist

# def get_beginning_info(model_path, fixed):
#     raise()
#     if model_path == 'openai-community/gpt2':
#         raise()
#         if fixed == 5:
#             begin_words =  ["i"," reached"," over"," and"," secretly"," und","id"," my"," seat","belt"," and"," when"," his"]
#             begin_time = [0,1,2,3,4,5,5,6,7,7,8,9,10]
#         elif fixed == 3:
#             begin_words = ["i"," reached"," over"," and"," secretly"," und","id"]
#             begin_time = [0,1,2,3,4,5,5]
#     elif model_path == 'meta-llama/Meta-Llama-3-8B':
#         if fixed == 15:
#             begin_words = ['<|begin_of_text|>', 'i', ' reached', ' over', ' and', ' secretly', ' und', 'id', ' my', ' seat', 'belt', ' and', ' when', ' his', ' foot', ' hit', ' the', ' brake', ' at', ' the', ' red', ' light', ' i', ' fl', 'ung', ' open', ' the', ' door', ' and', ' i', ' ran', ' i', ' had', ' no', ' shoes', ' on', ' i', ' was', ' crying', ' i', ' had', ' no']
#             begin_time = [0,0,1,2,3,4,5,5,6,7,7,8,9,10,11,12,13,14,15,16,17,18,19,20,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
#         elif fixed == 11:
#             begin_words = ['<|begin_of_text|>', 'i', ' reached', ' over', ' and', ' secretly', ' und', 'id', ' my', ' seat', 'belt', ' and', ' when', ' his', ' foot', ' hit', ' the', ' brake', ' at', ' the', ' red', ' light', ' i', ' fl', 'ung', ' open', ' the', ' door', ' and', ' i', ' ran', ' i']
#             begin_time = [0,0,1,2,3,4,5,5,6,7,7,8,9,10,11,12,13,14,15,16,17,18,19,20,20,21,22,23,24,25,26,27]
#         elif fixed == 3:
#             raise()
#             begin_words = ['<|begin_of_text|>', 'i', ' reached', ' over', ' and', ' secretly', ' und', 'id']
#             begin_time = [0,0,1,2,3,4,5,5,6,7,7,8,9,10]
#     return begin_words, begin_time

if __name__ == "__main__":

    logger = logging.getLogger("decode_words")
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--em_id", type = str, required = True)
    parser.add_argument("--wr_id", type = str, required = True)
    parser.add_argument("--generate_llm", type = str, required = True)
    parser.add_argument("--em_llm", type = str, required = True)
    args = parser.parse_args()

    test_stories = ['wheretheressmoke']
    em_data = np.load(config.RESULT_DIR+'/%s/test/%s/encoding_model_%s.npz' % (args.subject, args.em_llm, args.em_id))
    wr_data = np.load(config.RESULT_DIR+'/%s/test/%s/wordrate_model_%s.npz' % (args.subject, args.generate_llm, args.wr_id))
    logger.info(f"Word rate model: {wr_data['model_path']}")
    logger.info(f"Encoding model: {em_data['model_path']}")

    blank = " " if args.generate_llm == "original" else ""
    is_orig = (args.generate_llm == "original")

    generate_gpt = GPT(path = str(wr_data['model_path']), device = config.GPT_DEVICE)
    if args.generate_llm != "original":
        generate_gpt.model.generation_config.pad_token_id = generate_gpt.tokenizer.pad_token_id  
        generate_gpt.tokenizer.pad_token_id = generate_gpt.tokenizer.eos_token_id
    generate_features = LMFeatures(model = generate_gpt, layer = 1, context_words = -1)

    # Load and construct word rate model
    logger.info("Creating word rate model")
    resp, rate, mean_wordrate = get_wr_feature_pair(args.subject, wr_data['train_stories'], generate_features, str(wr_data['roi']))
    resp_test, rate_test, _ = get_wr_feature_pair(args.subject, test_stories, generate_features, str(wr_data['roi']), is_shuffle=False)
    clf = Ridge(alpha=wr_data['alpha'])
    clf.fit(resp, rate)
    word_rates = clf.predict(resp_test)
    word_rates = word_rates[:, 0] + mean_wordrate
    logger.info("mean corr=%0.5f" % (pearsonr(word_rates, rate_test[:, 0])[0]))

    # Load and construct encoding  model
    logger.info("Creating encoding model")
    em_gpt = GPT(path = str(em_data['model_path']), device = config.GPT_DEVICE)
    em_features = LMFeatures(model = em_gpt, layer = em_data['layer'], context_words = config.GPT_WORDS)
    corrs = copy.deepcopy(em_data['corr'])
    corrs[corrs < corrs[corrs.argsort()[-config.VOXELS]]] = 0
    vox = [i for i in range(corrs.shape[0]) if corrs[i] != 0]
    wordvec, resp = get_em_feature_pair(args.subject, em_data['train_stories'], em_features, vox)
    wordvec_test, resp_test = get_em_feature_pair(args.subject, test_stories, em_features, vox)
    logger.warning("Using mean weight!!!!")
    clf = Ridge(alpha=em_data['alpha'][vox].mean())
    clf.fit(wordvec, resp)
    assert resp_test.shape[0] == len(word_rates), f'{resp_test.shape[0]} != {len(word_rates)}'
    del wordvec, resp
    torch.cuda.empty_cache()

    fixed = 11 # sec
    word_seqs = get_story_wordseqs(test_stories)
    word_rates = np.array(list(map(round, word_rates)))
    if is_orig:
        max_length = 100
    else:
        max_length = min(80, generate_gpt.tokenizer.model_max_length - 50)

    for story in test_stories:
        for i in range(len(word_seqs[story].data_times)):
            if word_seqs[story].data_times[i] > fixed: break
        begin_words, begin_time = [], []
        data_times = [0]

        save_location = os.path.join(config.RESULT_DIR, args.subject, "decoding", "em%s_wr%s" % (args.em_id, args.wr_id))
        file_path = os.path.join(save_location, "%s_result_by_words" % (story)) + ".npz"
        if os.path.exists(file_path):
            raise("NotImplementedError")
            data = np.load(file_path)
            start = len(data["chance_em"])-1
            logger.info(f"Restart at {start}.")
            if "Llama" in str(em_data['model_path']):
                strs = [[[em_gpt.tokenizer.decode(ii) for ii in em_gpt.tokenizer.encode(s.decode()) if ii != em_gpt.tokenizer.bos_token_id] for s in data["can_stcs"][jj]] for jj in range(len(data["can_stcs"]))]
                chance_strs = [[[em_gpt.tokenizer.decode(ii) for ii in em_gpt.tokenizer.encode(s.decode()) if ii != em_gpt.tokenizer.bos_token_id] for s in data["chance_stcs"][jj]] for jj in range(len(data["chance_stcs"]))]
            elif "perceived" in str(em_data['model_path']): # which means original
                strs = [[s.decode().split(" ") for s in data["can_stcs"][jj]] for jj in range(len(data["can_stcs"]))] if not args.only_null else []
                chance_strs = [[s.decode().split(" ") for s in data["chance_stcs"][jj]] for jj in range(len(data["chance_stcs"]))]
            else: 
                NotImplementedError()
            for i in range(start):
                current_sec += 2
                # `current_sec` is right after generation time.
                # If fixed is 11, the `current_sec` at the first generation step is 13`
                data_times = np.concatenate([data_times, np.linspace(current_sec+4, current_sec+6, word_rates[i+unfixed_tr-1]+1)[:-1]])
            corr = data["can_corr"] if not args.only_null else []
            candidate = [[(wordlist, r) for wordlist, r in zip(strs[ii], corr[ii])] for ii in range(len(data["can_corr"]))] if not args.only_null else []
            ref_strs = [s.decode().split(" ") for s in data["ref_stcs"]] if not args.only_null else []
            ref_corr = data["ref_corr"] if not args.only_null else []
            reference = [(wordlist, r) for wordlist, r in zip(ref_strs, ref_corr)]
            chance_corr = [data["chance_em"][:, ii] for ii in range(data["chance_em"].shape[-1])]
            chances = [[(wordlist, r) for wordlist, r in zip(chance_strs[ii], chance_corr[ii])] for ii in range(len(chance_corr))]
        else:
            candidate = [[("", 1)]]# if not args.only_null else []
            start = 0
        for i in range(start, len(word_rates)-4):
            logger.info(f'{i}/{len(word_rates)-4}')
            for data_time in np.linspace(fixed+i*2, fixed+i*2+2, word_rates[i]+1)[:-1]: # repeat num of word time 
                res = []
                data_times = np.concatenate([data_times,[data_time]])
                for words_corr_list in candidate:
                    words = []
                    for ws, _ in words_corr_list:
                        words.append(ws)
                    for sample_output in generate_gpt.generate(blank.join(words[-max_length:]), 1, config.WIDTH):
                        if is_orig:
                            new_word = generate_gpt.vocab[sample_output[-1]] if sample_output[-1] < len(generate_gpt.vocab) else '<unk>'
                        else:
                            new_word = generate_gpt.tokenizer.decode(sample_output[-1], skip_special_tokens=True)
                        corrs = []
                        for i_tr in range(4):
                            start_tr, end_tr = 5+(i+i_tr)*2, 11+(i+i_tr)*2 # the first value is start_tr=5, end_tr=11
                            del_mat = get_stim_from_wordslist(
                                np.array(words+[new_word])[(start_tr-3 < data_times) & (end_tr+3 > data_times)],
                                em_features, 
                                data_times[(start_tr-3 < data_times).flatten() & (end_tr+3 > data_times.flatten())], 
                                list(range(start_tr, end_tr+3, 2)), 
                                mark=blank
                            )
                            # if i_tr==0:
                                # logger.info(f"start_tr, end_tr = {start_tr}, {end_tr}")
                                # logger.info(f"old_time : {data_times[(start_tr-3 < data_times) & (end_tr+3 > data_times)]}")
                                # logger.info(f"new_time : {list(range(start_tr, end_tr+3, 2))}")
                                # logger.info(f"num of zeros in del_mat : {del_mat[-1].reshape(1, -1)[del_mat[-1].reshape(1, -1)==0].shape}")
                                # logger.info(words+[new_word])
                            pred = clf.predict(del_mat[-1].reshape(1, -1))[0]
                            corrs.append(pearsonr(pred, resp_test[i+i_tr+1])[0])
                        # `res[i]` is list of tuple of words list and correlation. len(res) is config.EXTENSIONS*config.WIDTH.
                        res.append(words_corr_list + [(new_word, np.mean(corrs))])

                candidate = sorted(res, key=lambda x: x[-1][1])[::-1][:config.EXTENSIONS] # sort by pearson where is [1]. [::-1] means DECENT.
                logger.info(candidate[0][-10:])

                os.makedirs(save_location, exist_ok = True)
                np.savez(file_path,
                    can_stcs=np.array([list(map(lambda x: x[0].encode(), candidate[i])) for i in range(config.EXTENSIONS)]),
                    can_corr=np.array([list(map(lambda x: x[1], candidate[i])) for i in range(config.EXTENSIONS)]),
                    data_times=data_times, fixed=fixed, k=config.EXTENSIONS, samples=config.WIDTH
                )