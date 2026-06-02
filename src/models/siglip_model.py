import torch
import torch.nn as nn
from transformers import AutoModel

class SigLIPBackbone(nn.Module):
    def __init__(self, model_name="google/siglip2-base-patch32-256"):
        super().__init__()
        print(f"Loading pre-trained SigLIP 2 (ViT-B/32) from Hugging Face: {model_name}...")
        
        # Load the original pre-trained SigLIP 2 model
        self.model = AutoModel.from_pretrained(model_name)
        
        # Separate Image Encoder and Text Encoder to coordinate with the CoOp algorithm
        self.image_encoder = self.model.vision_model
        self.text_encoder = self.model.text_model
        
        # Get the original token embedding layer of the model
        self.token_embeddings = self.model.text_model.embeddings.token_embedding
        
        # Freeze all backbone weights of SigLIP to act as a fixed base for CoOp learning
        for param in self.parameters():
            param.requires_grad = False
            
        # Feature vector size (Embedding Dimension) for the Base model remains 768
        self.hidden_size = self.model.config.text_config.hidden_size 
        
    def encode_image(self, images):
        """Extract feature vectors from the input images"""
        vision_outputs = self.image_encoder(pixel_values=images)
        image_features = vision_outputs.pooler_output
        
        # L2 normalize the feature vectors for accurate cosine similarity calculation
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, text_embeddings, attention_mask=None):
        """Encode text sequences by receiving custom embedding vectors directly from CoOp"""
        # Extract the position embeddings layer of the model
        position_ids = self.text_encoder.embeddings.position_ids[:, :text_embeddings.size(1)]
        position_embeddings = self.text_encoder.embeddings.position_embedding(position_ids)
        
        # Merge CoOp's custom text embeddings with SigLIP's position embeddings
        input_embeddings = text_embeddings + position_embeddings
        
        encoder_outputs = self.text_encoder.encoder(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask
        )
        
        last_hidden_state = encoder_outputs.last_hidden_state
        text_features = last_hidden_state[:, -1, :] 
        
        # L2 normalize the text features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
