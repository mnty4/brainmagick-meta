import torch.nn.functional as F
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import librosa
import os

# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
 # model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", torch_dtype=torch.float16, attn_implementation="flash_attention_2", output_hidden_states=True).to(device)

class SpeechEmbeddings:
    def __init__(self, data_dir = 'data/gwilliams2022/download', **kwargs):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)

    def get_audio_embeddings(self, wav_path, start, duration, average_last_4_hidden_states=True, audio_embedding_length=30):
        
        audio, rate = librosa.load(os.path.join(self.data_dir, wav_path), sr = 16000, offset=start, duration=duration)
        
        # Process the audio to extract input features
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_features, output_hidden_states=True)

        audio_embeddings = outputs.last_hidden_state

        if average_last_4_hidden_states:
            hidden_states = outputs.hidden_states[-4:]
            audio_embeddings = torch.cat(hidden_states).mean(dim=0)

        # pad
        if audio_embeddings.shape[0] < audio_embedding_length:
            to_pad = audio_embedding_length - audio_embeddings.shape[0]
            audio_embeddings = F.pad(audio_embeddings, (0, 0, 0, to_pad), "constant", 0)
        # trim
        if audio_embeddings.shape[0] > audio_embedding_length:
            audio_embeddings = audio_embeddings[:audio_embedding_length]

        return audio_embeddings