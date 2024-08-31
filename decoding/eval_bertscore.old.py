import os
import sys
import logging
import argparse
import numpy as np
from tqdm import tqdm
from bert_score import BERTScorer

import config

if __name__ == "__main__":

    logger = logging.getLogger("eval_bertscore")
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--id", type = str, required = True)
    parser.add_argument("--candidate", type = int, default = 0)
    args = parser.parse_args()

    test_stories = ['wheretheressmoke']
    tmp_path = os.path.join('/home', 'anna', 'semantic-decoding', 'tmp', 'tmp%d.txt')
    for story in test_stories:
        result = np.load(os.path.join(config.RESULT_DIR, args.subject, "decoding", args.id, "%s_result.npz" % story))
        bertscores = {'f1': [], 'chance': [[] for _ in range(result["chance_em"].shape[-1])]}
        chances = result['chance_stcs']

        for i in tqdm(range(len(result['ref_corr'])-1)):
            can_tmp = ''.join([stc.decode() for stc in result['can_stcs'][args.candidate][1:][max(0,i-5):i+5]]).replace('\n', '').replace('<|begin_of_text|>', '')
            ref_tmp = ' '.join([stc.decode() for stc in result['ref_stcs'][1:][max(0,i-5):i+5]]).replace('\n', '')
            with open(tmp_path%1, 'w') as f:
                f.write(can_tmp+'\n')
            with open(tmp_path%2, 'w') as f:
                f.write(ref_tmp+'\n')

            idf_sents = np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy"))
            metric = BERTScorer(lang = "en", rescale_with_baseline = False, idf = (idf_sents is not None), idf_sents = idf_sents)
            score_id = 1
            f1_tmp = metric.score(cands = [can_tmp], refs = [ref_tmp])[score_id].numpy()[0]

            bertscores['f1'].append(f1_tmp)
            logger.info(ref_tmp)
            logger.info(f"[CAND] {f1_tmp}: {can_tmp}")
                
            for j in range(len(chances)):
                can_tmp = ''.join(list(map(lambda x: x.decode(), chances[j][1:][max(0,i-5):i+5]))).replace('\n', '')
                with open(tmp_path%1, 'w') as f:
                    f.write(can_tmp+'\n')
                stdout = ''
                f1_tmp = metric.score(cands = [can_tmp], refs = [ref_tmp])[score_id].numpy()[0]
                try:
                    if stdout != '': f1_tmp = float(stdout.split('F1: ')[-1])
                    logger.info(f"{f1_tmp}, {can_tmp}")
                    bertscores['chance'][j].append(f1_tmp)
                except:
                    if stdout != '': logger.warning(f"WARNING: Could not extract the score from {stdout}, {can_tmp}")
                    bertscores['chance'][j].append(0)

        save_location = os.path.join(config.RESULT_DIR, args.subject, "decoding", args.id)
        os.makedirs(save_location, exist_ok = True)
        np.savez(os.path.join(save_location, "%s_bertscore_original" % story),
            f1 = np.array(bertscores['f1']),
            chance_f1 = np.array(bertscores['chance'])
        )