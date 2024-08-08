import os
import sys
import pickle
import logging
import subprocess
import argparse
import numpy as np
from tqdm import tqdm

import config

if __name__ == "__main__":

    logger = logging.getLogger("eval_bertscore")
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--decode_res_id", type = str, required = True)
    parser.add_argument("--candidate", type = int, default = 0)
    args = parser.parse_args()

    test_stories = ['wheretheressmoke']
    for story in test_stories:
        with open(os.path.join(config.RESULT_DIR, args.subject, 'decoding', f'{args.decode_res_id}.pickle'), 'rb') as f:
            candidate = list(map(lambda x: x[0], pickle.load(f)[args.candidate]))
        with open(os.path.join(config.RESULT_DIR, args.subject, 'decoding', 'reference', story, f'{args.decode_res_id}.pickle'), 'rb') as f:
            reference = list(map(lambda x: x[0], pickle.load(f)))

        assert len(candidate) == len(reference)

        res = []
        for i in tqdm(range(len(candidate))):
            can_tmp = ''.join(candidate[max(0,i-5):i+5]).replace('\n', '')
            ref_tmp = ' '.join(reference[max(0,i-5):i+5]).replace('\n', '')

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
            f1 = list(map(lambda x:float(x[2]), res))
            )
