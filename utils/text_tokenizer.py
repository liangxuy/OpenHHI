import codecs as cs
from os.path import join as pjoin


class CaptionTokenizer:
    PAD_TOKEN = '<pad>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'

    def __init__(self, vocab_tokens):
        base_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        deduped_tokens = []
        seen = set(base_tokens)
        for token in vocab_tokens:
            if token not in seen:
                deduped_tokens.append(token)
                seen.add(token)

        self.idx_to_token = base_tokens + deduped_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        self.pad_id = self.token_to_idx[self.PAD_TOKEN]
        self.bos_id = self.token_to_idx[self.BOS_TOKEN]
        self.eos_id = self.token_to_idx[self.EOS_TOKEN]
        self.unk_id = self.token_to_idx[self.UNK_TOKEN]

    @classmethod
    def from_split_files(cls, text_dir, split_files):
        vocab_tokens = []
        seen = set()
        for split_file in split_files:
            with cs.open(split_file, 'r') as f:
                sample_ids = [line.strip() for line in f.readlines()]
            for sample_id in sample_ids:
                text_path = pjoin(text_dir, sample_id + '.txt')
                try:
                    with cs.open(text_path, 'r') as f:
                        for line in f.readlines():
                            line_split = line.strip().split('#')
                            if len(line_split) < 2:
                                continue
                            for token in line_split[1].split(' '):
                                if token not in seen:
                                    vocab_tokens.append(token)
                                    seen.add(token)
                except FileNotFoundError:
                    continue
        return cls(vocab_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def encode_tokens(self, tokens, max_text_len):
        tokens = list(tokens[:max_text_len])
        tokens = [self.BOS_TOKEN] + tokens + [self.EOS_TOKEN]
        sent_len = len(tokens)
        ids = [self.token_to_idx.get(token, self.unk_id) for token in tokens]
        max_seq_len = max_text_len + 2
        if len(ids) < max_seq_len:
            ids = ids + [self.pad_id] * (max_seq_len - len(ids))
        return ids, sent_len

    def decode_ids(self, ids, skip_special_tokens=True, strip_pos=False):
        tokens = []
        for idx in ids:
            token = self.idx_to_token[int(idx)]
            if skip_special_tokens and token in {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN}:
                if token == self.EOS_TOKEN:
                    break
                continue
            if strip_pos and '/' in token:
                token = token.rsplit('/', 1)[0]
            tokens.append(token)
        return tokens
