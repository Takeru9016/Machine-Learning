import torch
from torch import nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor(
            [tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64)
        self.eos_token = torch.Tensor(
            [tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_token = torch.Tensor(
            [tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __pattern__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src(src_text).ids
        dec_input_tokens = self.tokenizer_tgt(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long!')

        # Add SOS & EOS to the source text
        encoder_input = torch.cat(
            [self.sos_token, torch.tensor(
                enc_input_tokens, dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)]
        )

        # Add SOS to the decoder text
        decoder_input = torch.cat(
            [self.sos_token, torch.tensor(
                dec_input_tokens, dtype=torch.int64), torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)]
        )

        # Add EOS to the label (what we expect  as output from decoder)
        label = torch.cat(
            [torch.tensor(dec_input_tokens, dtype=torch.int64), self.sos_token, torch.tensor(
                [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # Seq_len
            "decoder_input": decoder_input,  # Seq_len
            "encoder_mask" : (encoder_input !== self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1, Seq_len)
            "decoder_mask" : (decoder_input !== self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # (1, Seq_len) & (1, Seq_len, Seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
        
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
