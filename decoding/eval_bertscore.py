import os
import sys
import logging
import argparse
import numpy as np
from tqdm import tqdm
from bert_score import BERTScorer

import config
from utils_stim import get_story_wordseqs

if __name__ == "__main__":

    logger = logging.getLogger("eval_bertscore")
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(stdout_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--id", type = str, required = True)
    parser.add_argument("--llm", type = str, required = True)
    parser.add_argument("--large", action='store_true')
    parser.add_argument("--candidate", type = int, default = 0)
    parser.add_argument("--num_chance", type = int, default = 0)
    parser.add_argument("--only_null", action='store_true')
    args = parser.parse_args()

    test_stories = ['wheretheressmoke']
    tmp_path = os.path.join('/home', 'anna', 'semantic-decoding', 'tmp', 'tmp%d.txt')

    idf_sents = np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy"))
    kwargs = {
        "lang": "en",
        "rescale_with_baseline": False,
        "idf": (idf_sents is not None),
        "idf_sents": idf_sents
    }
    if args.large:
        kwargs["model_type"] = "microsoft/deberta-xlarge-mnli"
    metric = BERTScorer(**kwargs)
    score_id = 1
    for story in test_stories:
        save_location = os.path.join(config.RESULT_DIR, args.subject, "decoding", args.id).replace("/home", "/Storage2")
        path = os.path.join(save_location, "%s_result%s%s" % (story,("" if args.num_chance < 0 else "_" + str(args.num_chance)),("_only_null" if args.only_null else "")))
        logger.info(path)
        result = np.load(path + ".npz")
        if args.num_chance:
            bertscores = {'f1': [], 'chance': [[] for _ in range(result["chance_em"].shape[-1])]}
        else:
            bertscores = {'f1': [], 'chance': [[]]}
        chances = result['chance_stcs']

        word_seqs = get_story_wordseqs([story])
        blank = " " if args.llm == "original" else ""
        fixed = 11
        current_sec = fixed
        try:
            # for i in tqdm(range(30)):
            for i in tqdm(range(len(result['ref_corr'])-1)):
                current_sec += 2
                ref_tmp = ' '.join(list(np.array(word_seqs[story].data)[(word_seqs[story].data_times < current_sec+10) & (word_seqs[story].data_times >= max(fixed, current_sec-10))]))
                # ref_tmp = ' '.join([stc.decode() for stc in result['ref_stcs'][1:][max(0,i-5):i+5]]).replace('\n', '')
                # with open(tmp_path%2, 'w') as f:
                #     f.write(ref_tmp+'\n')

                if not args.only_null:
                    can_tmp = blank.join([stc.decode() for stc in result['can_stcs'][args.candidate][1:][max(0,i-5):i+5]]).replace('\n', '').replace('<|begin_of_text|>', '').replace("<unk> ", "")
                    # with open(tmp_path%1, 'w') as f:
                    #     f.write(can_tmp+'\n')
                    f1_tmp = metric.score(cands = [can_tmp], refs = [ref_tmp])[score_id].numpy()[0]
                    bertscores['f1'].append(f1_tmp)
                    logger.info(ref_tmp)
                    logger.info(f"[CAND] {f1_tmp}: {can_tmp}")

                for j in range(len(chances)):
                    can_tmp = blank.join(list(map(lambda x: x.decode(), chances[j][1:][max(0,i-5):i+5]))).replace('\n', '').replace("<unk> ", "")
                    # with open(tmp_path%1, 'w') as f:
                    #     f.write(can_tmp+'\n')
                    f1_tmp = metric.score(cands = [can_tmp], refs = [ref_tmp])[score_id].numpy()[0]
                    logger.info(f"{f1_tmp}, {can_tmp}")
                    bertscores['chance'][j].append(f1_tmp)

            os.makedirs(save_location, exist_ok = True)
            np.savez(path.replace("_result", "_bertscore") + ("_large" if args.large else ""),
                f1 = np.array(bertscores['f1']),
                chance_f1 = np.array(bertscores['chance']),
            )
        except Exception as e:
            logger.warning(f"Exception occurred : {e}. Saving at {i}")
            os.makedirs(save_location, exist_ok = True)
            np.savez(path.replace("_result", "_bertscore") + ("_large" if args.large else ""),
                f1 = np.array(bertscores['f1']),
                chance_f1 = np.array(bertscores['chance']),
            )