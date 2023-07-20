
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding

from typing import Any, Tuple, Optional

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
            nm_index = torch.arange(1, seqlen+1)
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
                 lr = 0.0001,
                 betas = (0.9, 0.98),
                 weight_decay = 0.05
                 ):
        super().__init__()     

        self.retnet = RetentionNetwork(vocab_size=vocab_size, hidden_dim=hidden_dim, n_layers=n_layers, head_dim=head_dim)   
        self.head = nn.Linear(max_seqlen * hidden_dim, num_classes)

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay        

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.retnet(x)
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW([self.retnet.parameters(), self.head.parameters()], 
                                  lr=self.lr, 
                                  betas=self.betas, 
                                  weight_decay=self.weight_decay
                                  )
        return optim
    
    def training_step(self, batch, batch_idx):
        pass

if __name__=='__main__':
    model = RetNetClassification(vocab_size=10, hidden_dim=256, n_layers=3, head_dim=32)    
    # model = MSR(256, 32)
    # model = model.eval()
    x = torch.zeros(1, 512, dtype=torch.long)
    y = model(x)
    print(x.shape, y.shape)

    # trainer = pl.Trainer()

    # trainer.fit(model)
