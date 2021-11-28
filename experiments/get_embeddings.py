from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
from tqdm import tqdm
import pickle
import numpy as np

device = 'cuda:0'


class Embeddings(object):
    def __init__(self, path, sentences, checkpoints, size):
        self.path = path
        self.sentences = sentences
        self.checkpoints = checkpoints
        self.size = size

    def load_model(self, checkpoint_path):
        """
        Loads a transformer model
        :return: a model and a tokenizer
        """
        m = torch.load(checkpoint_path)
        model = T5ForConditionalGeneration(T5Config(output_hidden_states=True))
        model.load_state_dict(m['model_state_dict'])
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model.to(device)
        return model, tokenizer


    def get_emb(self, sent, model, tokenizer):
        """
        Encodes a sentence and returns an embedding
        :param sent: a sentence, str
        :param model: a transformer model
        :param tokenizer:
        :return:
        """
        with torch.no_grad():
            enc = tokenizer(sent, padding=True, truncation=True, return_tensors='pt')
            enc.to(device)
            output = model.encoder(
              input_ids=enc['input_ids'],
              attention_mask=enc['attention_mask'],
              return_dict=True
            )
        emb = output.hidden_states
        return [torch.mean(e, 1).squeeze(0).cpu().numpy() for e in emb]

    def calculate_embeddings(self, path):
        """
        Calculates embeddings for all sentences
        :param path: a path to a checkpoint
        :param sentences: a corpus of texts
        :return: a matrix of embeddings
        """
        model, tokenizer = self.load_model(path)
        print('Model is loaded')
        embeddings = np.zeros((len(self.sentences), self.size))
        for i, sentence in enumerate(self.sentences):
            embeddings[i] = self.get_emb(sentence, model, tokenizer)
        print('Embeddings are calculated')
        return embeddings

    def save_embeddings(self, embeddings, checkpoint):
        """
        Saves embeddings to a pickle file
        :param embeddings: a matrix of embeddings
        :param checkpoint: a checkpoint
        :return: a pickle file
        """
        with open(f'T5_checkpoints_{checkpoint}.pickle', 'wb') as f:
            pickle.dump(embeddings, f)

    def calculate(self):
        for checkpoint in tqdm(range(self.checkpoints)):
            path = self.path[checkpoint]
            embs = self.calculate_embeddings(path)
            self.save_embeddings(embs, checkpoint)
