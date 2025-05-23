import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
# nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose (e.g., "<extra_id_0>").
            * Class behavior should be different on the test set.
        '''
        
        # TODO
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.split = split
      
        self.bos_token = '<extra_id_0>'

        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(self.bos_token)
        self.data = []
        self.process_data(data_folder, split)

       
        

    def process_data(self, data_folder, split):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        sql_path = os.path.join(data_folder, f"{split}.sql") 

        nl_lines = load_lines(nl_path)
        if split in ['train', 'dev']:

            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql_lines = load_lines(sql_path)
            assert len(nl_lines) == len(sql_lines)
            for nl, sql in zip(nl_lines, sql_lines):
                prompted_nl = "translate English to SQL: " + nl
                enc = self.tokenizer(prompted_nl, add_special_tokens=True, return_attention_mask=True)
                dec = self.tokenizer(f"{self.bos_token} {sql}", add_special_tokens=True)
                self.data.append({
                        "encoder_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                        "encoder_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                        "decoder_ids": torch.tensor(dec["input_ids"], dtype=torch.long),
                        "sql_line": sql
                    })
        elif split == 'test':
            for nl in nl_lines:
                prompted_nl = "translate English to SQL: " + nl
                enc = self.tokenizer(prompted_nl, add_special_tokens=True, return_attention_mask=True)
                dec_ids = [self.bos_token_id]
                self.data.append({
                    "encoder_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                    "encoder_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                    "decoder_ids": torch.tensor(dec_ids, dtype=torch.long),
                    "sql_line": ''
                })
        else:
            raise ValueError(f"Unknown split: {split}")
 

    
    def __len__(self):

        # TODO
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data[idx]

        #sql_line is just a string of the sql command with the special bos token added in the beginning. 
        return data_dict["encoder_ids"], data_dict["encoder_mask"], data_dict["decoder_ids"], \
            data_dict["sql_line"]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to the decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)

    decoder_ids = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=PAD_IDX)
    
    decoder_inputs = decoder_ids[:, :-1].contiguous()
    decoder_targets = decoder_ids[:, 1:].contiguous()
    initial_decoder_inputs = decoder_ids[:, 0].unsqueeze(1)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to the decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_ids = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = decoder_ids[:, 0].unsqueeze(1)  # SQL string

    return encoder_ids, encoder_mask, initial_decoder_inputs
    

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    
    return train_x, train_y, dev_x, dev_y, test_x