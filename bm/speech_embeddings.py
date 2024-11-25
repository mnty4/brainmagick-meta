import torch.nn.functional as F
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import librosa
import os
import typing as tp

# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
 # model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", torch_dtype=torch.float16, attn_implementation="flash_attention_2", output_hidden_states=True).to(device)

class SpeechEmbeddings:
    def __init__(self, data_dir = 'data/gwilliams2022/download', model="facebook/wav2vec2-large-xlsr-53", device=None, layers: tp.Tuple[int, ...] = (14, 15, 16, 17, 18), **kwargs):
        self.data_dir = data_dir
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.processor = Wav2Vec2Processor.from_pretrained(model)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model)
        self.model = Wav2Vec2Model.from_pretrained(model).to(self.device)
        self.layers = layers

    def get_audio_embeddings(self, wav_path, start, duration, average_last_4_hidden_states=True, audio_embedding_length=30):
        
        audio, rate = librosa.load(os.path.join(self.data_dir, wav_path), sr = 16000, offset=start, duration=duration)
        
        # Process the audio to extract input features
        # input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        input_features = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_features, output_hidden_states=True)

        audio_embeddings = outputs.last_hidden_state
        
        if average_last_4_hidden_states:
            out = outputs.hidden_states
            if isinstance(out, tuple):
                out = torch.stack(out)
            if self.layers is not None:
                out = out[list(self.layers)].mean(0)
            audio_embeddings = out.squeeze(0).cpu()
            # hidden_states = outputs.hidden_states[-4:]
            # audio_embeddings = torch.cat(hidden_states).mean(dim=0)

        # pad
        if audio_embeddings.shape[0] < audio_embedding_length:
            to_pad = audio_embedding_length - audio_embeddings.shape[0]
            audio_embeddings = F.pad(audio_embeddings, (0, 0, 0, to_pad), "constant", 0)
        # trim
        if audio_embeddings.shape[0] > audio_embedding_length:
            audio_embeddings = audio_embeddings[:audio_embedding_length]

        return audio_embeddings