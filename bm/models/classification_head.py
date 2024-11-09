import torch
from torch.nn.modules import Module
import torch.nn.functional as F

class EEG_Encoder_Classification_Head(Module):
    def __init__(
        self,
        eeg_encoder,
        num_classes,
        eeg_projection
    ):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        # freeze eeg encoder
        for param in eeg_encoder.parameters():
            param.requires_grad = False

        self.num_classes = num_classes
        self.fc = torch.nn.Linear(eeg_encoder.embed_dim, num_classes)
        self.eeg_projection = eeg_projection
        self.maxpool = torch.nn.AdaptiveMaxPool1d(1)
        self.ce_loss = torch.nn.CrossEntropyLoss()
    
    def freeze(self):
        for param in self.eeg_encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.eeg_encoder.parameters():
            param.requires_grad = True

    def forward(self, batch):
        eeg_input = batch["eeg"].float()
        # use rand_like to replace eeg_input
        mask_ratio = 0
        # --------------------Encode---------------------
        mae_loss, eeg_pred, eeg_mask, eeg_latent = self.eeg_encoder(eeg_input,mask_ratio)
        # eeg_latent, eeg_mask, eeg_ids_restore= self.eeg_encoder.mask_encode(eeg_input,mask_ratio)
        eeg_mae_encode = eeg_latent[:, 1:, :]
        # apply max pooling to get eeg token
        eeg_mae_encode = self.maxpool(eeg_mae_encode.transpose(1,2)).squeeze(-1)
        
        eeg_embeddings = self.eeg_projection(eeg_mae_encode)

        batch_preds = self.fc(eeg_embeddings)

        return {
            'clf_logits': batch_preds
        }

    def generate(self, batch):
        x = self.forward(batch)
        return {
            'clf_logits': x['clf_logits'],
            'ce_loss': self.ce_loss(x['clf_logits'], batch['w_lbs'])
        }

    def predict(self, batch):
        x = self.forward(batch)
        x = F.softmax(x, dim=-1)
        x = x.argmax(dim=-1).squeeze(-1)

        return x
