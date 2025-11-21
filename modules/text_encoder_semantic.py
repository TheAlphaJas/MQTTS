import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class SemanticEncoder(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L12-v2', freeze=True, device='cpu'):
        super().__init__()
        print(f"Loading Semantic Encoder: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        # Standard SBERT Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, text_list):
        """
        Args:
            text_list: List[str] of raw text
        Returns:
            sentence_embeddings: (B, 384) - SBERT embedding
        """
        # Tokenize
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_input = {k: v.to(self.model.device) for k, v in encoded_input.items()}

        with torch.set_grad_enabled(not self.model.training): # Respect freeze status
            model_output = self.model(**encoded_input)
        
        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings
