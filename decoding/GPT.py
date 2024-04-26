import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import softmax

class GPT():    
    """wrapper for https://huggingface.co/openai-gpt
    """
    def __init__(self, path, device = 'cpu'):
        self.device = device
        self.path = path
        if path != 'eng1000':
            self.model = AutoModel.from_pretrained(path, device_map="balanced")#.eval().to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(path)
            self.word2id = tokenizer.vocab
            if tokenizer.unk_token == '':
                self.UNK_ID = -1
            else:
                self.UNK_ID = tokenizer.encode(tokenizer.unk_token)[0]

    def encode(self, words):
        """map from words to ids
        """
        return [self.word2id[x] if x in self.word2id else self.UNK_ID for x in words]
        
    def get_story_array(self, words, context_words):
        """get word ids for each phrase in a stimulus story
        """
        nctx = context_words + 1
        story_ids = self.encode(words)
        story_array = np.zeros([len(story_ids), nctx]) + self.UNK_ID
        for i in range(len(story_array)):
            segment = story_ids[i:i+nctx]
            story_array[i, :len(segment)] = segment
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