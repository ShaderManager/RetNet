
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding

from datasets import load_dataset

from typing import Any, Tuple, Optional

from itertools import chain

import argparse

from tokenizers import Tokenizer, models, decoders, pre_tokenizers, trainers

class MSR(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 head_dim: int,
                 ):        
        super().__init__()

        self.n_heads = n_heads = hidden_dim // head_dim
        self.head_dim = head_dim

        self.gn = nn.GroupNorm(n_heads, n_heads)
        self.act = nn.Mish()

        self.register_buffer('gamma', 1 - torch.pow(2.0, -5.0 - torch.arange(0, n_heads)))

        self.wg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.wqkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.pos_emb = RotaryEmbedding(head_dim // 2, use_xpos=True)

    def forward(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
            x = input vector of shape [bs, seqlen, hidden dim]
            hidden_state = recurrent state from previous step. Initialized by layer itself is null and module is in eval state
        """
        bs, seqlen, _ = x.shape

        qkv: torch.Tensor = self.wqkv(x) # [bs, seqlen, 3xhidden dim]
        qkv = rearrange(qkv, 'B S (H D) -> B H S D', H=self.n_heads)
        q, k, v = qkv.chunk(3, dim=-1) # 3x[bs, heads, seqlen, head dim]

        # Eq. (5)
        # Apply xPos embedding on Q/K
        q, k = self.pos_emb.rotate_queries_and_keys(q, k)

        if self.training:
            # Eq. (5) 
            # Dnm = pow(gamma, n-m) if n>=m else 0
            # TODO: implement Retention Score normalization
            nm_index = torch.arange(1, seqlen+1, device=x.device)
            nm = repeat(nm_index, 'W -> W H', H=seqlen) - repeat(nm_index, 'W -> H W', H=seqlen)
            decay_mask = torch.pow(self.gamma.view(-1, 1, 1), nm) * (nm >= 0).int()

            ret: torch.Tensor = q @ k.transpose(-1, -2)
            ret = ret * decay_mask
            ret = ret @ v            
            ret = self.gn(ret)
        else:
            # Eq. (6)
            if hidden_state is None:
                hidden_state = torch.zeros(bs, self.n_heads, q.shape[-1], v.shape[-1])                

            hidden_state = self.gamma.view(1, -1, 1, 1) * hidden_state + k.transpose(-1, -2) @ v
            ret = q @ hidden_state

        # Eq. (8)
        y = rearrange(ret, 'B H S D -> B S (H D)')
        y = self.act(self.wg(x)) * y
        y = self.wo(y)
        
        return y, hidden_state

class FFN(nn.Module):
    def __init__(self, hidden_dim: int,):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))

class RetNetBlock(nn.Module):
    def __init__(self, hidden_dim: int, 
                 head_dim: int,):
        super().__init__()

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.msr = MSR(hidden_dim=hidden_dim, head_dim=head_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = FFN(hidden_dim)

    def forward(self, x: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Eq. (9)        
        y, hidden_state = self.msr(self.ln1(x), hidden_state)
        y = y + x
        return self.ffn(self.ln2(y)) + y, hidden_state

class RetentionNetwork(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 hidden_dim: int,
                 n_layers: int,
                 head_dim: int = 256,
                 ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([RetNetBlock(hidden_dim=hidden_dim, head_dim=head_dim) for i in range(n_layers)])

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        x = self.emb(x)
        hidden_state: torch.Tensor = None

        for layer in self.layers:
            x, hidden_state = layer(x, hidden_state)

        return x

class RetNetClassification(pl.LightningModule):
    def __init__(self,
                 vocab_size: int,
                 hidden_dim: int,
                 n_layers: int,
                 head_dim: int = 256,
                 max_seqlen: int = 512,
                 num_classes: int = 2,
                 *,
                 lr = 0.001,
                 betas = (0.9, 0.98),
                 weight_decay = 0.05
                 ):
        super().__init__()     

        self.hidden_dim = hidden_dim
        self.max_seqlen = max_seqlen

        self.retnet = RetentionNetwork(vocab_size=vocab_size, hidden_dim=hidden_dim, n_layers=n_layers, head_dim=head_dim)   
        self.head = nn.Linear(max_seqlen * hidden_dim, num_classes)

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        y = self.retnet(x)
        y = self.head(y.view(-1, self.max_seqlen * self.hidden_dim))
        y = F.softmax(y, dim=1)
        return y
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(chain(self.retnet.parameters(), self.head.parameters()), 
                                  lr=self.lr, 
                                  betas=self.betas, 
                                  weight_decay=self.weight_decay
                                  )
        return optim
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        y = self.forward(x)
        
        loss = self.loss(y, labels)
        self.log_dict({'train_loss' : loss})
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, labels = batch
        y = self.forward(x)
        
        loss = self.loss(y, labels)

        self.log_dict({'test_loss' : loss})

class IMDBDataModule(pl.LightningDataModule):
    def __init__(self, 
                 tokenizer: Tokenizer,
                 train_batch_size: int = 32,
                 test_batch_size: int = 4,
                 predict_batch_size: int = 4,    
                 max_len: int = 512,             
                 ):
        super().__init__()
        self.name = 'imdb'
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.predict_batch_size = predict_batch_size     
        self.max_len = 512   

        self.tokenizer = tokenizer

    def prepare_data(self):
        load_dataset(self.name)        

    def train_tokenizer(self, trainer: trainers.Trainer):
        for split, dataset in load_dataset(self.name).items():
            def batch_iterator(batch_size=1000):
                for i in range(0, len(dataset), batch_size):
                    yield dataset[i:i+batch_size]["text"]

            print(f'Train tokenizer for split {split}')            

            self.tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(dataset))

    def tokenize(self, item):
        return {
                'tokens' : [o.ids for o in self.tokenizer.encode_batch(item['text'])]
                }

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = load_dataset(self.name, split='train')
            self.train_dataset.set_format(type='torch')
            self.train_dataset = self.train_dataset.map(lambda e: self.tokenize(e), batched=True)

        elif stage == 'test':
            self.test_dataset = load_dataset(self.name, split='test')
            self.test_dataset.set_format(type='torch')
            self.test_dataset = self.test_dataset.map(lambda e: self.tokenize(e), batched=True)

        elif stage == 'predict':
            self.predict_dataset = load_dataset(self.name, split='unsupervised')
            self.predict_dataset.set_format(type='torch')
            self.predict_dataset = self.predict_dataset.map(lambda e: self.tokenize(e), batched=True)

    def collate(self, batch):
        bs = len(batch)
        batched_ids = torch.zeros(bs, self.max_len, dtype=torch.long)
        batched_labels = torch.zeros(bs, dtype=torch.long)

        for id, item in enumerate(batch):
            tokens, labels = item['tokens'], item['label']
            l = min(tokens.shape[0], self.max_len)
            batched_ids[id, :l] = tokens[:l]
            batched_labels[id] = labels
        return batched_ids, batched_labels

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, collate_fn=lambda e: self.collate(e))
    
    def test_dataloader(self):
        return DataLoader(self.test_dataloader, batch_size=self.test_batch_size, collate_fn=lambda e: self.collate(e))
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_sampler=self.predict_batch_size, collate_fn=lambda e: self.collate(e))

if __name__=='__main__':

    torch.set_float32_matmul_precision('high')

    try:
        tokenizer = Tokenizer.from_file('imdb.json')
        tokenizer_trainer = None
    except:
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

        tokenizer.decoder = decoders.WordPiece()
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer_trainer = trainers.WordPieceTrainer(vocab_size=3000, 
                                                      special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
                                                      )    

    datamodule = IMDBDataModule(tokenizer=tokenizer)
    datamodule.prepare_data()

    if tokenizer_trainer is not None:
        datamodule.train_tokenizer(tokenizer_trainer)
        tokenizer.save('imdb.json')

    model = RetNetClassification(vocab_size=tokenizer.get_vocab_size(), hidden_dim=256, n_layers=3, head_dim=32, max_seqlen=512)    
    # model = MSR(256, 32)
    # model = model.eval()
    # x = torch.zeros(1, 512, dtype=torch.long)
    # y = model(x)
    # print(x.shape, y.shape)

    trainer = pl.Trainer(max_epochs=10)

    trainer.fit(model, datamodule)
