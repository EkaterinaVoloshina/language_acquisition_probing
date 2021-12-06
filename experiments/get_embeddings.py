from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
from tqdm import tqdm
import pickle
import numpy as np

device = 'cuda:0'


class Embeddings(object):
    def __init__(self, path, sentences, labels, checkpoints, size, batch_size, emb_name):
        self.path = path
        self.sentences = sentences
        self.labels = labels
        self.checkpoints = checkpoints
        self.size = size
        self.batch_size = batch_size
        self.emb_name = emb_name

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

    def encode(self, tokenizer):
        input_ids = []
        decoder_input_ids = []
        attention_mask = []
        for text in self.sentences:
              tokenized_text = tokenizer.encode_plus(text,
                                                  #max_length=512,
                                                  add_special_tokens=True,
                                                  padding='max_length',
                                                  truncation=True,
                                                  return_attention_mask=True)
              input_ids.append(tokenized_text['input_ids'])
              attention_mask.append(tokenized_text['attention_mask'])
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)

    def get_batches(self, tokenizer):
        y = torch.tensor(self.labels, dtype=torch.long)
        input_ids, attention_mask = self.encode(tokenizer)
        tensor_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, y)
        tensor_dataloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=self.batch_size)
        return tensor_dataloader

    def mean_pooling(self, model_output, attention_mask):
        tokens = np.zeros((32, 512, 7))
        for num, i in enumerate(model_output):
            tokens[:,:, num] = i[:,0,:].cpu().detach().numpy()
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(tokens.shape).float().cpu().detach().numpy()
        normalized = np.clip(input_mask_expanded, a_min=1e-9, a_max=None)
        summa = np.sum(tokens * input_mask_expanded)/normalized
        return tokens

    def get_emb(self, batch, model, tokenizer):
        """
        Encodes a sentence and returns an embedding
        :param batch: a batch
        :param model: a transformer model
        :param tokenizer:
        :return:
        """
        input_ids, attention_mask, labels = batch
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
        return mean_pool, labels


    def calculate_embeddings(self, path):
        """
        Calculates embeddings for all sentences
        :param path: a path to a checkpoint
        :param sentences: a corpus of texts
        :return: a matrix of embeddings
        """
        model, tokenizer = self.load_model(path)
        print('Model is loaded')
        batchs = self.get_batches(tokenizer)
        labels = []
        embeddings = np.zeros((len(self.sentences), self.size, 7))
        for i, batch in enumerate(batchs):
            embeddings[i:(i+self.batch_size)], label = self.get_emb(batch, model, tokenizer)
            labels.extend(label)
        print('Embeddings are calculated')
        return embeddings

    def save_embeddings(self, embeddings, checkpoint):
        """
        Saves embeddings to a pickle file
        :param embeddings: a matrix of embeddings
        :param checkpoint: a checkpoint
        :return: a pickle file
        """
        with open(f'/content/gdrive/MyDrive/T5_checkpoints_{self.emb_name}_{checkpoint}.pickle', 'wb') as f:
            pickle.dump(embeddings, f)

    def calculate(self):
        for checkpoint in tqdm(range(self.checkpoints)):
            path = self.path[checkpoint]
            embs = self.calculate_embeddings(path)
            self.save_embeddings(embs, checkpoint)
