import torch
import sys
import os
import numpy as np
import logging
import copy
import json

import config
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import softmax

logger = logging.getLogger("GPT")
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)
torch.backends.cudnn.enabled = False

class GPT():    
    """wrapper for https://huggingface.co/openai-gpt
    """
    def __init__(self, path, device = 'cpu'):
        self.device = device
        self.path = path
        if path == 'eng1000':
            pass
        elif path == os.path.join(config.DATA_LM_DIR, "perceived", "model") or path == "original":
            with open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json"), "r") as f:
                vocab = json.load(f)
            self.model = AutoModelForCausalLM.from_pretrained(os.path.join(config.DATA_LM_DIR, "perceived", "model")).eval().to(self.device)
            self.vocab = vocab
            self.word2id = {w : i for i, w in enumerate(self.vocab)}
            self.UNK_ID = self.word2id['<unk>']
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, device_map="balanced")#.eval().to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            # self.UNK_ID = tokenizer.unk_token_id
            # self.UNK_ID = -1 if tokenizer.unk_token is None else tokenizer.encode(tokenizer.unk_token)[0]
            if self.tokenizer.unk_token is None:
                # _ = self.tokenizer.add_tokens('<|unk|>', special_tokens=True)
                # self.UNK_ID = len(self.tokenizer) - 1
                # self.model.resize_token_embeddings(len(self.tokenizer))
                self.UNK_ID = 0
            else:
                self.UNK_ID = self.tokenizer.encode(self.tokenizer.unk_token)[0]
            self.word2id = self.tokenizer.vocab

    def get_wordind2tokind(self, words):
        wordind2tokind = []
        ids = self.tokenizer.encode(' '.join(words), max_length=5000)
        i_i = 0
        for w_i in range(len(words)):
            c = 1
            # print(words[w_i], w_i)
            if self.path == 'meta-llama/Meta-Llama-3-8B' and ids[i_i] == 128000:
                wordind2tokind.append(w_i)
                i_i += 1
            if words[w_i] in ['']:
                if self.tokenizer.decode(ids[i_i]).replace(' ', '') == '':
                    i_i += c
                    wordind2tokind.extend([w_i for _ in range(c)])
                continue
            while self.tokenizer.decode(ids[i_i:i_i+c]) != (' ' if (self.tokenizer.decode(ids[i_i:i_i+c])[0]==' ') else '') + words[w_i]:
                c += 1
                if c > 10:
                    assert False, f'{self.tokenizer.decode(ids[i_i:i_i+c])} != {words[w_i]}\n{self.tokenizer.decode(ids[i_i])} : first token'
            logger.debug(self.tokenizer.decode(ids[i_i:i_i+c]))
            wordind2tokind.extend([w_i for _ in range(c)])
            i_i += c
        assert len(ids) == len(wordind2tokind), f"{len(ids)} != {len(wordind2tokind)}"
        return wordind2tokind
        
    def encode(self, words, mark = ' ', old_tokeni=True):
        """map from words to ids
        """
        if self.path in [os.path.join(config.DATA_LM_DIR, "perceived", "model"), "original"] :
            return [self.word2id[x] if x in self.word2id else self.UNK_ID for x in words], list(range(len(words)))

        if not old_tokeni:
            return self.tokenizer.encode(mark.join(words), max_length=5000)
        # return [self.word2id[x] if x in self.word2id else self.UNK_ID for x in words]

        i = 0
        wordind2tokind = []
        if mark == '':
            ids = []
            for i, w in enumerate(words):
                tmp = self.tokenizer.encode(w)
                if i != 0 and self.path == 'meta-llama/Meta-Llama-3-8B':
                    tmp.remove(128000)
                    if len(tmp) == 1 and tmp[0] == 128001: continue
                    ids.extend(tmp)
                    wordind2tokind.extend([i for _ in range(len(tmp))])
        else:
            ids = self.tokenizer.encode(mark.join(words), max_length=5000)
            for _i, _id in enumerate(ids):
                tok = self.tokenizer.decode(_id)
                logger.debug(f"tok :{tok}\tword[i] :{words[i]}")
                if tok in [' ']:
                    wordind2tokind.append(i-1)
                    continue
                elif tok in ['<|begin_of_text|>']:
                    wordind2tokind.append(i+1)
                    continue
                while words[i] in ['', ' ']:
                    if tok.replace(' ', '') == '': break
                    else: i += 1

                words[i] = words[i].replace(' ', '')
                if tok.replace(' ', '') == words[i] \
                or tok == words[i]:
                    wordind2tokind.append(i)
                    i += 1
                    logger.debug('1')
                elif tok[0] != ' ':# and _i != 1:
                    tmp_i = 0
                    for tmp_i in range(_i-1, -1, -1):
                        if self.tokenizer.decode(ids[tmp_i])[0] == ' ': break
                    if ((' ' if (tmp_i or (words[i-1] == '')) else '') + words[i]) == self.tokenizer.decode(ids[tmp_i:_i+1]) \
                    or (self.tokenizer.decode(ids[tmp_i]) == '<|begin_of_text|>') and (words[i] == self.tokenizer.decode(ids[tmp_i+1:_i+1])):
                        wordind2tokind.append(i)
                        i += 1
                        logger.debug(f'2 {self.tokenizer.decode(ids[tmp_i:_i+1])}')
                    elif self.tokenizer.decode(ids[tmp_i:_i+1]) == "'s" \
                    and (' ' + words[i]) == self.tokenizer.decode(ids[tmp_i])+self.tokenizer.decode(ids[_i]):
                        logger.debug('3')
                        wordind2tokind.append(i)
                        i += 1
                    elif tok.replace(' ', '') in words[i]:
                        logger.debug('4')
                        logger.debug(f' {words[i]}')
                        logger.debug(f"{(' ' if tmp_i else '') + words[i]} != {self.tokenizer.decode(ids[tmp_i:_i+1])}")
                        logger.debug(f'{self.tokenizer.decode(ids[tmp_i])+self.tokenizer.decode(ids[_i])}')
                        logger.debug(f'{tmp_i} {self.tokenizer.decode(ids[tmp_i])} {self.tokenizer.encode(self.tokenizer.decode(ids[tmp_i]))}')
                        wordind2tokind.append(i)
                    else:
                        logger.debug('5')
                        logger.debug(f'{tmp_i} {self.tokenizer.decode(ids[tmp_i:_i+1])}')
                        logger.debug(f"1 tok :{tok}\tword[i-3:i+3] :{words[max(0,i-3):i+3]}")
                        raise
                elif tok.replace(' ', '') in words[i]:
                    wordind2tokind.append(i)
                else:
                    logger.debug(f"2 tok :{tok}\tword[i-3:i+3] :{words[max(0,i-3):i+3]}")
                    raise
                    
        assert len(ids) == len(wordind2tokind), f"{len(ids)} != {len(wordind2tokind)}"
        return ids, wordind2tokind
    
    def generate(self, sentence, max_new_tokens, num_smaple, do_sample=False):
        if self.path in [os.path.join(config.DATA_LM_DIR, "perceived", "model"), "original"] :
            tok = {'input_ids': torch.tensor(self.encode(sentence.split(" "))[0]).reshape(1,-1)}
            tok['attention_mask'] = torch.ones_like(tok["input_ids"])
        else:
            tok = self.tokenizer(sentence, return_tensors="pt")
        if do_sample:
            outputs = self.model.generate(
                input_ids = tok['input_ids'].to(config.GPT_DEVICE),
                attention_mask = tok['attention_mask'].to(config.GPT_DEVICE),
                max_length = tok['input_ids'].shape[-1] + max_new_tokens,
                do_sample = do_sample,
                repetition_penalty = 2.0,
                num_return_sequences = num_smaple,
                # bad_words_ids = [[128001]], #[[self.tokenizer.eos_token_id]],
                # pad_token_id = self.tokenizer.eos_token_id
            )
        else:
            outputs = self.model.generate(
                input_ids = tok['input_ids'].to(config.GPT_DEVICE),
                attention_mask = tok['attention_mask'].to(config.GPT_DEVICE),
                max_length = tok['input_ids'].shape[-1] + max_new_tokens,
                do_sample = False,
                num_beams=100,            # ビームサーチの探索幅
                diversity_penalty=1.0,    # 生成結果の多様性を生み出すためのペナルティ
                num_beam_groups=100, 
                repetition_penalty = 2.0,
                num_return_sequences = num_smaple,
                # bad_words_ids = [[128001]], #[[self.tokenizer.eos_token_id]],
                # pad_token_id = self.tokenizer.eos_token_id
            )
        return outputs

    def get_story_array(self, words, context_words, mark=' ', old_tokeni=True):
        """get word ids for each phrase in a stimulus story
        """
        nctx = context_words + 1
        if old_tokeni:
            story_ids, wordind2tokind = self.encode(words, mark=mark, old_tokeni=old_tokeni)
        else:
            story_ids = self.encode(words, mark=mark)
        story_array = np.zeros([len(story_ids), nctx]) + self.UNK_ID
        for i in range(len(story_array)):
            segment = story_ids[i:i+nctx]
            story_array[i, :len(segment)] = segment

        if old_tokeni:
            return torch.tensor(story_array).long(), wordind2tokind
        else:
            return torch.tensor(story_array).long()

    def get_context_array(self, contexts):
        """get word ids for each context
        """
        context_array = np.array([self.encode(words) for words in contexts])
        return torch.tensor(context_array).long()

    def get_hidden(self, ids, layer):
        """get hidden layer representations
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), 
                                 attention_mask = mask.to(self.device), output_hidden_states = True)
        return outputs.hidden_states[layer].detach().cpu().numpy()

    def get_probs(self, ids):
        """get next word probability distributions
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), attention_mask = mask.to(self.device))
        probs = softmax(outputs.logits, dim = 2).detach().cpu().numpy()
        return probs

    def get_story_array_and_hidden(self, words, context_words, layer, mark=' ', old_tokeni=True):
        wordind2tokind = None
        nctx = context_words + 1
        if old_tokeni:
            story_ids, wordind2tokind = self.encode(words, mark=mark, old_tokeni=old_tokeni)
        else:
            story_ids = self.encode(words, mark=mark)
        story_array = np.zeros([len(story_ids), nctx]) + self.UNK_ID
        for i in range(len(story_array)):
            segment = story_ids[i:i+nctx]
            story_array[i, :len(segment)] = segment

        if context_words == -1:
            return None, wordind2tokind, None
        else:
            ids = torch.tensor(story_array).long()
            mask = torch.ones(ids.shape).int()
            outputs = []
            with torch.no_grad():
                for i in range(len(ids)):
                    output = self.model(input_ids = torch.reshape(ids[i], (1, -1)).to(self.device), 
                                        attention_mask = torch.reshape(mask[i], (1,-1)).to(self.device), output_hidden_states = True)
                    outputs.append(output.hidden_states[layer][0].detach().cpu().numpy())
            return ids, wordind2tokind, np.array(outputs)