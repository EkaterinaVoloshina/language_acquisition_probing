from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
from tqdm import tqdm
import pickle
import numpy as np

device = 'cuda:0'


class Embeddings(object):
    def __init__(self, path, sentences, labels, checkpoints, size, batch_size, emb_name, delay=0):
        self.path = path
        self.sentences = sentences
        self.labels = labels
        self.checkpoints = checkpoints
        self.size = size
        self.batch_size = batch_size
        self.emb_name = emb_name
        self.delay = delay

    def load_model(self, checkpoint_path):
        """
        Loads a transformer model
        :return: a model and a tokenizer
        """
        m = torch.load(checkpoint_path)
        model = T5ForConditionalGeneration(T5Config(output_hidden_states=True))
        model.load_state_dict(m['model_state_dict'])
        model.to(device)
        return model

    def encode(self, tokenizer):
        input_ids = []
        attention_mask = []
        for text in self.sentences:
              tokenized_text = tokenizer.batch_encode_plus(text,
                                                  #max_length=512,
                                                  add_special_tokens=True,
                                                  padding='max_length',
                                                  truncation=True,
                                                  return_attention_mask=True)
              input_ids.append(tokenized_text['input_ids'])
              attention_mask.append(tokenized_text['attention_mask'])
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)

    def mean_pooling(self, model_output, attention_mask):
        tokens = np.zeros((self.model_output[0].shape[0], self.size, 7))
        for num, i in enumerate(model_output):
            tokens[:,:, num] = i[:,0,:].cpu().detach().numpy()
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(tokens.shape).float().cpu().detach().numpy()
        normalized = np.clip(input_mask_expanded, a_min=1e-9, a_max=None)
        summa = np.sum(tokens * input_mask_expanded)/normalized
        return tokens

    def get_emb(self, batch, model):
        """
        Encodes a sentence and returns an embedding
        :param batch: a batch
        :param model: a transformer model
        :return:
        """
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            output = model(
              input_ids=input_ids,
              decoder_input_ids=input_ids,
              attention_mask=attention_mask,
              return_dict=True
            )
        emb = output.decoder_hidden_states
        mean_pool = self.mean_pooling(emb, attention_mask)
        embedding = torch.cat(mean_pool, dim=1)
        return embedding

    def calculate_embeddings(self, path, tokenizer):
        """
        Calculates embeddings for all sentences
        :param path: a path to a checkpoint
        :param sentences: a corpus of texts
        :return: a matrix of embeddings
        """
        model = self.load_model(path)
        print('Model is loaded')
        batches = self.encode(tokenizer)
        emb_number = len(batches[0][0])
        embeddings = np.zeros((len(self.sentences), self.size * emb_number, 7))
        for i, batch in enumerate(batches):
            embeddings[i] = self.get_emb(batch, model)
        print('Embeddings are calculated')
        return embeddings

    def save_embeddings(self, embeddings, checkpoint):
        """
        Saves embeddings to a pickle file
        :param embeddings: a matrix of embeddings
        :param checkpoint: a checkpoint
        :return: a pickle file
        """
        with open(f'embeddings/T5_checkpoints_{self.emb_name}_{checkpoint+self.delay}.pickle', 'wb') as f:
            pickle.dump(embeddings, f)

    def calculate(self):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        for checkpoint in tqdm(range(self.checkpoints)):
            path = self.path[checkpoint]
            embs, labels = self.calculate_embeddings(path, tokenizer)
            self.save_embeddings(embs, checkpoint)
