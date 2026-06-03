import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from src.models.siglip_model import SigLIPBackbone
from src.models.prompt_learner import PromptLearner

class CustomCLIP(nn.Module):
    def __init__(self, classnames, model_name="google/siglip2-base-patch32-256", n_ctx=4, ctx_init="a photo of a", apply_lora=True):
        super().__init__()
        self.backbone = SigLIPBackbone(model_name)
        
        # === APPLY LORA TO VISION ENCODER ===
        if apply_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"], # Freeze structural weights, optimize attention layers only
                lora_dropout=0.1,
                bias="none",
            )
            self.backbone.image_encoder = get_peft_model(self.backbone.image_encoder, lora_config)
            print("Successfully integrated LoRA into Vision Encoder:")
            self.backbone.image_encoder.print_trainable_parameters()
            
        # === INITIALIZE PROMPT LEARNER (CoOp) ===
        self.prompt_learner = PromptLearner(classnames, self.backbone, model_name, n_ctx, ctx_init)
        
        # Learnable logit scale and bias specific to SigLIP framework
        self.logit_scale = nn.Parameter(torch.ones([]) * 10.0)
        self.logit_bias = nn.Parameter(torch.zeros([]))
        
    def forward(self, image):
        # 1. Extract Image Features via LoRA-enhanced Vision Encoder
        image_features = self.backbone.encode_image(image)
        
        # 2. Extract Text Features via CoOp Prompt Learner and Frozen Text Encoder
        prompts, attention_mask = self.prompt_learner()
        text_features = self.backbone.encode_text(prompts, attention_mask)
        
        # 3. Compute Logits via Sigmoid scaled dot-product
        logits = (image_features @ text_features.t()) * self.logit_scale.exp() + self.logit_bias
        return logits