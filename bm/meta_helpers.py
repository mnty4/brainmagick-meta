import torch
import typing as tp

class Batch:
    def __init__(self, subject_index, recording_index, study_name, features):
        self._subject_index = subject_index
        self._recording_index = recording_index
        self._study_name = study_name
        self._features = features
    @property
    def features(self) -> int:
        if self._features is None:
            raise RuntimeError("Recording.features has not been initialized")
        return self._features
    @property
    def subject_index(self) -> int:
        if self._subject_index is None:
            raise RuntimeError("Recording.subject_index has not been initialized")
        return self._subject_index

    @property
    def recording_index(self) -> int:
        if self._recording_index is None:
            raise RuntimeError("Recording.recording_index has not been initialized")
        return self._recording_index
    
    @classmethod
    def study_name(self) -> str:
        return self._study_name
    
def parse_to_input_batch_format(input_batch: dict, DEVICE) -> tp.Tuple[dict, dict]:
    batch = Batch(
                subject_index=torch.tensor([int(input_batch['sub_id'].split('-')[1])] * input_batch['eeg'].shape[0]).to(DEVICE), 
                recording_index=[int(input_batch['story_uid'])] * input_batch['eeg'].shape[0],
                study_name='gwilliams2022',
                features=input_batch['audio'])
    
    input = dict(meg=input_batch['eeg'])

    mask = torch.ones((input_batch['eeg'].shape[0], 1, input_batch['eeg'].shape[-1])).to(torch.bool)
    
    return input, batch, mask