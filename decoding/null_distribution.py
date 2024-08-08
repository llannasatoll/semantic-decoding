import os
import sys
import numpy as np
import argparse
import logging
import torch
import pickle

from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_ridge.util import make_delayed
from utils_ridge.interpdata import lanczosinterp2D
from utils_stim import get_story_wordseqs
from utils_decode import get_wr_feature_pair, get_em_feature_pair

MODELS = {
    'eng1000' : 'eng1000',
    'gpt2' : 'openai-community/gpt2',
    'llama3' : 'meta-llama/Meta-Llama-3-8B',
}

if __name__ == "__main__":

    logger = logging.getLogger("decode_words")
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--llm", type = str, required = True)
    parser.add_argument("--scores", nargs = "+", type = str, default = ['bert', 'em'])
    parser.add_argument("--wr_id", type = str, required = True)
    parser.add_argument("--em_id", type = str, required = True)
    args = parser.parse_args()

    test_stories = ['wheretheressmoke']

    # Load and construct word rate model
    wr_data = np.load('/home/anna/semantic-decoding/results/%s/test/%s/wordrate_model_%s.npz' % (args.subject, args.llm, args.wr_id))
    assert wr_data['model_path'] == MODELS[args.llm]
    
    gpt = GPT(path = MODELS[args.llm], device = config.GPT_DEVICE)
    features = LMFeatures(model = gpt, layer = 0, context_words = config.GPT_WORDS)

    logger.info("Creating word rate model")
    train_stories = wr_data['train_stories']
    resp, rate, mean_wordrate = get_wr_feature_pair(args.subject, train_stories, gpt, features, str(wr_data['roi']))
    resp_test, rate_test, _ = get_wr_feature_pair(args.subject, test_stories, gpt, features, str(wr_data['roi']))
    clf = Ridge(alpha=wr_data['alpha'])
    clf.fit(resp, rate)
    word_rates = clf.predict(resp_test)
    word_rates = word_rates[:, 0] + mean_wordrate

    # check the correlation
    log_template = "mean corr=%0.5f"
    log_msg = log_template % (pearsonr(word_rates, rate_test[:, 0])[0])
    logger.info(log_msg)

    fixed = 5 # sec
    unfixed_tr = int((7-fixed)/2) + 1
    samples = 100
    if fixed == 5:
        word_rates = np.insert(word_rates, 0, [7,6] + [mean_wordrate for _ in range(unfixed_tr)])
        candidate = ["i"," reached"," over"," and"," secretly"," und","id"," my"," seat","belt"," and"," when"," his"]
    elif fixed == 3:
        word_rates = np.insert(word_rates, 0, [7] + [mean_wordrate for _ in range(unfixed_tr)])
        candidate = ["i"," reached"," over"," and"," secretly"," und","id"]
    else:
        raise
    word_rates = np.array(list(map(round, word_rates)))
    word_rates_sum = [sum(word_rates[:i]) for i in range(len(word_rates))]
    print(len(word_rates_sum))
    # assert len(word_rates) == 292 or len(word_rates) == 291 ####!!!!!消す

    tok = gpt.tokenizer(''.join(candidate), return_tensors="pt")
    outputs = gpt.model.generate(
        input_ids = tok['input_ids'].to(config.GPT_DEVICE),
        attention_mask = tok['attention_mask'].to(config.GPT_DEVICE),
        max_new_tokens = sum(word_rates[2:]),
        do_sample = True,
        num_return_sequences = samples
        )

    with open(os.path.join(config.RESULT_DIR, args.subject, 'decoding', 'reference', story, "em%s_wr%s.pickle" % (args.em_id, args.wr_id)), 'rb') as f:
        reference = list(map(lambda x: x[0], pickle.load(f)))

    res = [[] for ]
    if 'bert' in args.scores:
        for sample_output in enumerate(outputs):
            for i in range(len(word_rates)):
                can_tmp = gpt.tokenizer.decode(sample_output[word_rates_sum[max(0,i-5)]:word_rates_sum[i+5]])
                ref_tmp = ''.join(reference[max(0,i-5):i+5]).replace('\n', '')

                with open('tmp1.txt', 'w') as f:
                    f.write(can_tmp+'\n')
                with open('tmp2.txt', 'w') as f:
                    f.write(ref_tmp+'\n')
                cmd = ['bert-score', '--lang', 'en', '--rescale_with_baseline', '--model', 'microsoft/deberta-xlarge-mnli', \
                '--num_layers', '40', '-r', 'tmp1.txt', '-c', 'tmp2.txt']
                result = subprocess.run(cmd, capture_output=True, text=True)
                tmp = result.stdout.split('rescaled_fast-tokenizer ')[-1].replace('P: ','').replace('R: ','').replace('F1: ','').split()
                assert len(tmp) == 3
                res.append(tmp)

        save_location = os.path.join(config.RESULT_DIR, args.subject, "decoding", "bertscore")
        os.makedirs(save_location, exist_ok = True)
        np.savez(os.path.join(save_location, args.decode_res_id),
            precision = list(map(lambda x:float(x[0]), res)), \
            recall = list(map(lambda x:float(x[1]), res)), \
            f1 = list(map(lambda x:float(x[2]), res)))

    data_times = np.concatenate(
        [word_seqs[story].data_times[:i][[0,1,2,3,4,5,5,6,7,7,8,9,10]], np.linspace(fixed, 7, word_rates[0]+1)[:-1]]
        )
    data_times = np.concatenate(
        [word_seqs[story].data_times[:i][[0,1,2,3,4,5,5]], np.linspace(fixed, 7, sum(word_rates[:unfixed_tr-1])+1)[:-1]]
        )
    if 'em' in  args.scores:
        em_data = np.load('/home/anna/semantic-decoding/results/%s/test/%s/encoding_model_%s.npz' % (args.subject, args.llm, args.em_id))

        assert em_data['model_path'] == wr_data['model_path']
        assert em_data['model_path'] == MODELS[args.llm]
        features = LMFeatures(model = gpt, layer = em_data['layer'], context_words = config.GPT_WORDS)
    
        # Load and construct encoding  model
        logger.info("Creating encoding model")
        
        train_stories = em_data['train_stories']
        tmp = em_data['corr']
        tmp[tmp < tmp[tmp.argsort()[-config.VOXELS]]] = 0
        vox = [i for i in range(tmp.shape[0]) if tmp[i] != 0]
        wordvec, resp = get_em_feature_pair(args.subject, train_stories, features, vox)
        wordvec_test, resp_test = get_em_feature_pair(args.subject, test_stories, features, vox)
        clf = Ridge(alpha=em_data['alpha'][vox])
        clf.fit(wordvec, resp)

        assert resp_test.shape[0] == len(word_rates), f'{resp_test.shape[0]} != {len(word_rates)}'

    samples = 100
    k = 3
    fixed = 5 # sec
    unfixed_tr = int((7-fixed)/2) + 1
    word_seqs = get_story_wordseqs(test_stories)
    word_rates = np.insert(word_rates, 0, [mean_wordrate for _ in range(unfixed_tr)])
    word_rates = np.array(list(map(round, word_rates)))

    for story in test_stories:
        for i in range(len(word_seqs[story].data_times)):
            if word_seqs[story].data_times[i] > fixed: break
        candidate = [[(["i"," reached"," over"," and"," secretly"," und","id"," my"," seat","belt"," and"," when"," his"], 1)]]
        data_times = np.concatenate(
            [word_seqs[story].data_times[:i][[0,1,2,3,4,5,5,6,7,7,8,9,10]], np.linspace(fixed, 7, word_rates[0]+1)[:-1]]
            )
        # candidate = [[(["i"," reached"," over"," and"," secretly"," und","id"], 1)]]
        # data_times = np.concatenate(
        #     [word_seqs[story].data_times[:i][[0,1,2,3,4,5,5]], np.linspace(fixed, 7, sum(word_rates[:unfixed_tr-1])+1)[:-1]]
        #     )
        current_sec = fixed
        reference = [(word_seqs[story].data[:i], 1)]
        gpt.model.generation_config.pad_token_ids = gpt.tokenizer.pad_token_id  

        for i in range(len(word_rates)-unfixed_tr):
            current_sec += 2
            data_times = np.concatenate([data_times, np.linspace(current_sec, current_sec+2, word_rates[i+unfixed_tr-1]+1)[:-1]])
            num_words = sum(word_rates[i:i+unfixed_tr])

            if num_words <= 0: continue
            new_words = []
            res = []

            for words_corr_list in candidate:
                words = []
                for ws, _ in words_corr_list:
                    words.extend(ws)
                tok = gpt.tokenizer(''.join(words[-1000:]), return_tensors="pt")
                outputs = gpt.model.generate(
                    input_ids = tok['input_ids'].to(config.GPT_DEVICE),
                    attention_mask = tok['attention_mask'].to(config.GPT_DEVICE),
                    max_new_tokens = num_words,
                    do_sample = True,
                    num_return_sequences = samples
                    )
                for ii, sample_output in enumerate(outputs):
                    tmp = []
                    for t in sample_output[len(gpt.tokenizer.encode(''.join(words[-1000:]))):]:
                        tmp.append(gpt.tokenizer.decode(t))
                    word_mat, wordind2tokind = features.make_stim(words+tmp, mark = '')
                    word_mean, word_std = word_mat.mean(0), word_mat.std(0)
                    ds_mat = lanczosinterp2D(word_mat, data_times[wordind2tokind], list(range(current_sec-(fixed-1), current_sec+(9-fixed)+1, 2)))
                    
                    r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
                    r_std[r_std == 0] = 1
                    ds_mat = np.nan_to_num(np.dot((ds_mat - r_mean), np.linalg.inv(np.diag(r_std))))
                    
                    del_mat = make_delayed(ds_mat, config.STIM_DELAYS)

                    pred = clf.predict(del_mat[-1].reshape(1, -1))[0]
                    res.append(words_corr_list + [(tmp[:word_rates[i]], pearsonr(pred, resp_test[i])[0])])
            candidate = sorted(res, key=lambda x: x[-1][1])[::-1][:k]
            # for _i, r in enumerate(list(map(lambda x: x[1], sorted(res, key=lambda x: x[1])[::-1][:k]))):
            #     print("r = ", r)
            #     print(''.join(candidate[_i]))

            act_pred = clf.predict(wordvec_test[i].reshape(1, -1))[0]
            logger.info(f"Likelihood with actual stim : {pearsonr(act_pred, resp_test[i])[0]}")
            reference.append((
                list(np.array(word_seqs[story].data)[(word_seqs[story].data_times < current_sec) & (word_seqs[story].data_times >= current_sec-2)]),
                pearsonr(act_pred, resp_test[i])[0]
                ))
            logger.info([can[-1] for can in candidate])
            logger.info(reference[-1])
            
        logger.info(f"candidate\n\n{candidate}")
        logger.info(f"reference\n\n{reference}")

        save_location = os.path.join(config.RESULT_DIR, args.subject, "decoding")
        os.makedirs(save_location, exist_ok = True)
        with open(os.path.join(save_location, "em%s_wr%s.pickle" % (args.em_id, args.wr_id)), mode='wb') as f:
            pickle.dump(list(map(lambda x: list(map(lambda y: (''.join(y[0]), y[1]), x)), candidate)), f)
        save_location = os.path.join(save_location, "reference", story)
        os.makedirs(save_location, exist_ok = True)
        with open(os.path.join(save_location, "em%s_wr%s.pickle" % (args.em_id, args.wr_id)), mode='wb') as f:
            pickle.dump(list(map(lambda y: (' '.join(y[0]), y[1]), reference)), f)