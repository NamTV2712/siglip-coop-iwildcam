import torch
import torch.nn as nn
from transformers import AutoModel

class SigLIPBackbone(nn.Module):
    def __init__(self, model_name="google/siglip2-base-patch32-256"):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(model_name)
        
        self.image_encoder = self.model.vision_model
        self.text_encoder = self.model.text_model
        
        self.token_embeddings = self.model.text_model.embeddings.token_embedding
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.model.config.text_config.hidden_size 
        
    def encode_image(self, images):
        """Input: Image Tensor [Batch, 3, 256, 256] -> Output: Vector [Batch, 768]"""
        vision_outputs = self.image_encoder(pixel_values=images)
        image_features = vision_outputs.pooler_output
        
        # L2 normalization
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, text_embeddings, attention_mask=None):
        """Input: String Tensor [Num_Classes, Seq_Len, 768] -> Output: Vector [Num_Classes, 768]"""
        position_ids = self.text_encoder.embeddings.position_ids[:, :text_embeddings.size(1)]
        position_embeddings = self.text_encoder.embeddings.position_embedding(position_ids)
        
        input_embeddings = text_embeddings + position_embeddings
        
        # === BẢN VÁ LỖI SDPA ATTENTION MASK ===
        if attention_mask is not None:
            # Lấy shape đầu vào [Num_Classes, Seq_Len]
            input_shape = text_embeddings.size()[:-1]
            
            # Sử dụng hàm nội bộ của Hugging Face để mở rộng mask từ 2D int (0, 1) 
            # thành 4D float (0.0, -10000.0) khớp hoàn toàn với định dạng PyTorch yêu cầu
            attention_mask = self.text_encoder.get_extended_attention_mask(attention_mask, input_shape)
        
        encoder_outputs = self.text_encoder.encoder(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask
        )
        
        last_hidden_state = encoder_outputs.last_hidden_state
        text_features = last_hidden_state[:, -1, :] 
        
        # L2 normalization
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features