from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)




@dataclass
class BeltSimilarity(ModelOutput):
    sim_i2t: torch.FloatTensor = None
    sim_t2i: torch.FloatTensor = None

    sim_i2t_m: Optional[torch.FloatTensor] = None
    sim_t2i_m: Optional[torch.FloatTensor] = None

    sim_i2t_targets: Optional[torch.FloatTensor] = None
    sim_t2i_targets: Optional[torch.FloatTensor] = None

@dataclass
class BeltIntermediateOutput(ModelOutput):
    """
    Data class for intermediate outputs of Belt models.

    image_embeds (torch.FloatTensor): Image embeddings, shape (batch_size, num_patches, embed_dim).
    text_embeds (torch.FloatTensor): Text embeddings, shape (batch_size, seq_len, embed_dim).

    image_embeds_m (torch.FloatTensor): Image embeddings from momentum visual encoder, shape (batch_size, num_patches, embed_dim).
    text_embeds_m (torch.FloatTensor): Text embeddings from momentum text encoder, shape (batch_size, seq_len, embed_dim).

    encoder_output (BaseModelOutputWithPoolingAndCrossAttentions): output from the image-grounded text encoder.
    encoder_output_neg (BaseModelOutputWithPoolingAndCrossAttentions): output from the image-grounded text encoder for negative pairs.

    decoder_output (CausalLMOutputWithCrossAttentions): output from the image-grounded text decoder.
    decoder_labels (torch.LongTensor): labels for the captioning loss.

    itm_logits (torch.FloatTensor): logits for the image-text matching loss, shape (batch_size * 3, 2).
    itm_labels (torch.LongTensor): labels for the image-text matching loss, shape (batch_size * 3,)

    """
    # uni-modal features
    image_embeds: torch.FloatTensor = None
    text_embeds: Optional[torch.FloatTensor] = None

    image_embeds_m: Optional[torch.FloatTensor] = None
    text_embeds_m: Optional[torch.FloatTensor] = None

    # intermediate outputs of multimodal encoder
    encoder_output: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    encoder_output_neg: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None

    itm_logits: Optional[torch.FloatTensor] = None
    itm_labels: Optional[torch.LongTensor] = None

    # intermediate outputs of multimodal decoder
    decoder_output: Optional[CausalLMOutputWithCrossAttentions] = None
    decoder_labels: Optional[torch.LongTensor] = None

@dataclass
class PureLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None

@dataclass
class BeltOutput(ModelOutput):
    # some finetuned models (e.g. BeltVQA) do not compute similarity, thus optional.
    sims: Optional[BeltSimilarity] = None

    intermediate_output: BeltIntermediateOutput = None

    loss: Optional[torch.FloatTensor] = None

    loss_itc: Optional[torch.FloatTensor] = None

    loss_itm: Optional[torch.FloatTensor] = None

    loss_de: Optional[torch.FloatTensor] = None

    loss_lm: Optional[torch.FloatTensor] = None

    acc_eeg2t: Optional[torch.FloatTensor] = None

    acc_t2eeg: Optional[torch.FloatTensor] = None

    acc_match: Optional[torch.FloatTensor] = None


@dataclass
class BeltVQOutput(ModelOutput):    
    sims: Optional[BeltSimilarity] = None
    intermediate_output: BeltIntermediateOutput = None
    loss: Optional[torch.FloatTensor] = None
    loss_vq_div: Optional[torch.FloatTensor] = None
    loss_vq_embed: Optional[torch.FloatTensor] = None
    loss_neg_ct: Optional[torch.FloatTensor] = None
    loss_recon: Optional[torch.FloatTensor] = None
    clip_loss: Optional[torch.FloatTensor] = None
    loss_lm: Optional[torch.FloatTensor] = None
    acc_sth: Optional[torch.FloatTensor] = None

@dataclass
class Belt1Output(ModelOutput):
    sims: Optional[BeltSimilarity] = None

    intermediate_output: BeltIntermediateOutput = None

    loss: Optional[torch.FloatTensor] = None
    loss_vq : Optional[torch.FloatTensor] = None
    loss_lm: Optional[torch.FloatTensor] = None
    loss_clip: Optional[torch.FloatTensor] = None


@dataclass
class BeltVQwlOutput(ModelOutput):
    # some finetuned models (e.g. BeltVQA) do not compute similarity, thus optional.
    sims: Optional[BeltSimilarity] = None

    intermediate_output: BeltIntermediateOutput = None

    loss: Optional[torch.FloatTensor] = None

    loss_vq_div: Optional[torch.FloatTensor] = None

    loss_vq_embed: Optional[torch.FloatTensor] = None

    loss_neg_ct: Optional[torch.FloatTensor] = None

    loss_wl_ct: Optional[torch.FloatTensor] = None

    loss_recon: Optional[torch.FloatTensor] = None

    clip_loss: Optional[torch.FloatTensor] = None

    loss_lm: Optional[torch.FloatTensor] = None

    acc_sth: Optional[torch.FloatTensor] = None

@dataclass
class BeltMtOutput(ModelOutput):
    # some finetuned models (e.g. BeltVQA) do not compute similarity, thus optional.
    loss: Optional[torch.FloatTensor] = None
    translation_loss: Optional[torch.FloatTensor] = None
    summary_loss: Optional[torch.FloatTensor] = None
    sst_loss: Optional[torch.FloatTensor] = None
    acc_sth: Optional[torch.FloatTensor] = None

@dataclass
class BeltOutputWithLogits(BeltOutput):
    logits: torch.FloatTensor = None
    logits_m: torch.FloatTensor = None


@dataclass
class Belt3Output(BeltOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_lm: Optional[torch.FloatTensor] = None
    loss_mae: Optional[torch.FloatTensor] = None



@dataclass
class BeltOutputFeatures(ModelOutput):
    """
    Data class of features from BeltFeatureExtractor.

    Args:
        image_embeds: (torch.FloatTensor) of shape (batch_size, num_patches+1, embed_dim), optional
        image_features: (torch.FloatTensor) of shape (batch_size, num_patches+1, feature_dim), optional
        text_embeds: (torch.FloatTensor) of shape (batch_size, sequence_length+1, embed_dim), optional
        text_features: (torch.FloatTensor) of shape (batch_size, sequence_length+1, feature_dim), optional

        The first embedding or feature is for the [CLS] token.

        Features are obtained by projecting the corresponding embedding into a normalized low-dimensional space.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    image_embeds_proj: Optional[torch.FloatTensor] = None

    text_embeds: Optional[torch.FloatTensor] = None
    text_embeds_proj: Optional[torch.FloatTensor] = None

    multimodal_embeds: Optional[torch.FloatTensor] = None


@dataclass
class Belt3MaskPretrainOutput(ModelOutput):
    # some finetuned models (e.g. BeltVQA) do not compute similarity, thus optional.
    loss: Optional[torch.FloatTensor] = None

@dataclass
class Belt3MaeClfOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_mae: Optional[torch.FloatTensor] = None
    loss_clf: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    acc: Optional[torch.FloatTensor] = None

@dataclass
class Belt3MaeRegOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_mae: Optional[torch.FloatTensor] = None
    loss_reg: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

@dataclass
class BeltCaptionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    acc1: Optional[torch.FloatTensor] = None
    acc5: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None
    targets: Optional[torch.LongTensor] = None
    sort_ind: Optional[torch.LongTensor] = None
    decode_lengths: Optional[torch.LongTensor] = None

@dataclass
class BeltWorCLIPOuput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    clf_loss: Optional[torch.FloatTensor] = None
    clip_loss: Optional[torch.FloatTensor] = None
    mae_loss: Optional[torch.FloatTensor] = None
    clf_logits: Optional[torch.FloatTensor] = None
    focal_loss: Optional[torch.FloatTensor] = None
    ldam_loss: Optional[torch.FloatTensor] = None
    eeg_embeds: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
