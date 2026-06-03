import torch
import torch.nn as nn
from transformers import AutoTokenizer

class PromptLearner(nn.Module):
    def __init__(self, classnames, backbone, model_name="google/siglip2-base-patch32-256", n_ctx=4, ctx_init="a photo of a"):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        ctx_dim = backbone.hidden_size
        
        # Initialize Context Tokens (CoOp)
        if ctx_init:
            ctx_init_text = ctx_init.replace("_", " ")
            # Tokenize the initialization text without adding special tokens
            prompt_ids = self.tokenizer(ctx_init_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
            
            if len(prompt_ids) == n_ctx:
                with torch.no_grad():
                    ctx_vectors = backbone.token_embeddings(prompt_ids).detach()
                print(f"Initialized CoOp with context: '{ctx_init_text}'")
            else:
                print(f"Warning: n_ctx={n_ctx} but '{ctx_init_text}' produces {len(prompt_ids)} tokens. Falling back to Random Initialization.")
                ctx_vectors = torch.empty(n_ctx, ctx_dim)
                nn.init.normal_(ctx_vectors, std=0.02)
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            
        self.ctx = nn.Parameter(ctx_vectors) 
        
        # Tokenize classnames 
        max_len = 64
        class_prompts = [name for name in classnames]
        tokenized = self.tokenizer(
            class_prompts, 
            padding="max_length", 
            truncation=True, 
            max_length=max_len - n_ctx, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            class_embeddings = backbone.token_embeddings(tokenized.input_ids)
            
        # === FIX LỖI DEVICE Ở ĐÂY ===
        # Đăng ký class_embeddings như một buffer để PyTorch tự động mang nó lên GPU cùng model
        self.register_buffer("class_embeddings", class_embeddings)
        
        # Safely extract or construct the attention mask
        if "attention_mask" in tokenized:
            base_mask = tokenized.attention_mask
        else:
            # Fallback: Create mask manually (1 for real tokens, 0 for padding tokens)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            base_mask = (tokenized.input_ids != pad_id).long()
            
        self.register_buffer("base_attention_mask", base_mask)
        
    def forward(self):
        # Expand Context Vectors for all classes: [n_cls, n_ctx, dim]
        ctx_expanded = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        # Concatenate Context Tokens ahead of Class Embeddings
        prompts = torch.cat([ctx_expanded, self.class_embeddings], dim=1)
        
        # Generate a mask of 1s for Context Tokens and concatenate with the base class mask
        ctx_mask = torch.ones((self.n_cls, self.n_ctx), dtype=self.base_attention_mask.dtype, device=self.base_attention_mask.device)
        attention_mask = torch.cat([ctx_mask, self.base_attention_mask], dim=1)
        
        return prompts, attention_mask