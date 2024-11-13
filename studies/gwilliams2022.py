# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example dataset whereby 21 subjects listening to two sessions of
~1 hour while they listened to audio stories ('tasks').

There are four stories; the subjects are presented to the same
stories in the first and second sessions.

Each story is composed of multiple wav files.
Each meg recording consists of a single story (and thus of different wav files)
"""
import typing as tp
from itertools import product
from pathlib import Path

import mne
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from . import api
from . import utils
from .download import download_osf
from ..events import extract_sequence_info


class StudyPaths(utils.StudyPaths):
    def __init__(self) -> None:
        super().__init__(Gwilliams2022Recording.study_name())
        self.megs = self.download / "all_data" / "MEG"
        self.events = self.download / "stimuli" / "events"


STORIES = ("lw1", "cable_spool_fort", "easy_money", "The_Black_Willow")


class Gwilliams2022Recording(api.Recording):
    data_url = "https://drive.google.com/drive/u/0/folders/"
    data_url += "1u1l4oX_OfammKPT49OlgbAmjGGuaA4qE"
    paper_url = "https://www.biorxiv.org/content/10.1101/2020.04.04.025684v2"
    doi = "https://doi.org/10.1101/2020.04.04.025684"
    licence = ''
    modality = "audio"
    language = "en"
    device = "meg"
    description = "21 subjects listened to 4 stories, in 2 x 1h identical sessions."
    def __init__(self, subject_uid: str, session: str, story: str) -> None:
        recording_uid = f'{subject_uid}_session{session}_story{story}'
        super().__init__(subject_uid=subject_uid, recording_uid=recording_uid)
        # FIXME in this dataset, the "task" is the story. The task is alway audio.
        self.story = story
        self.session = session
    
    @classmethod
    def download(cls) -> None:
        download_osf('ag3kj', StudyPaths().download.parent, 'ag3kj')
        download_osf('h2tzn', StudyPaths().download.parent, 'h2tzn')
        download_osf('u5327', StudyPaths().download.parent, 'u5327')
    # pytest: disable=arguments-differ
    @classmethod
    def iter(cls) -> tp.Iterator["Gwilliams2022Recording"]:  # type: ignore
        """Returns a generator of all recordings"""
        # download, extract, organize
        print('!!!! [Gwilliams2022Recording], skipping download in line 63')
        # cls.download()
        # List all recordings: depends on study structure
        paths = StudyPaths()
        subject_file = paths.download / "participants.tsv"
        print(' [Gwilliams2022Recording] subject_file,',subject_file,' file exists:', subject_file.exists())
        subjects = pd.read_csv(subject_file, sep="\t")

        def get_subject_id(x):
            return x.split("-")[1]  # noqa

        subjects = subjects.participant_id.apply(get_subject_id).values
        stories = [str(x) for x in range(4)]   # 4stories
        sessions = [str(x) for x in range(2)]  # 2 recording sessions
        for subject, session, story in product(subjects, sessions, stories):
            bids_path = BIDSPath(
                subject=subject,
                session=session,
                task=story,
                root=paths.download,
                datatype="meg",
            )
            if not Path(str(bids_path)).exists():
                continue

            recording = cls(subject_uid=subject, session=session, story=story)
            yield recording

    def _load_raw(self) -> mne.io.RawArray:
        print('[Gwilliams2022Recording] loading raw data')
        paths = StudyPaths()
        bids_path = BIDSPath(
            subject=self.subject_uid,
            session=self.session,
            task=self.story,
            root=paths.download,
            datatype="meg",
        )
        print(bids_path)
        assert Path(str(bids_path)).exists() 
        # /projects/SilSpeech/Dev/SilentSpeech_Se2/listen_meg_eeg_preprocess/brainmagick/data/gwilliams2022_newdl/download/sub-15/ses-1/meg/sub-15_ses-1_task-0_meg.con
        # check if this path exists
        # print('path exists:', Path(str(bids_path)).exists())

        raw = read_raw_bids(bids_path,verbose=False)  # FIXME this is NOT a lazy read
        print('raw:',type(raw))
        self.raw_sample_rate = raw.info["sfreq"]
        picks = dict(meg=True, eeg=False, stim=False, eog=False, ecg=False, misc=False)
        raw = raw.pick_types(**picks,verbose=False)
        print('raw:',type(raw))
        return raw

    def _load_events(self) -> pd.DataFrame:
        print('[Gwilliams2022Recording] loading events')
        """
        in this particular data, I'm transforming our original rich dataframe
        into mne use a Annotation class in order to save the whole thing into
        a *.fif file, At reading time, I'm converting it back to a DataFrame
        """
        raw = self.raw()
        paths = StudyPaths()

        # extract annotations
        events = list()
        for annot in raw.annotations:
            event = eval(annot.pop("description"))
            event['audio_start'] = event['start']
            event['start'] = annot['onset']
            event['duration'] = annot['duration']
            if event["kind"] == "sound":
                stem, _, ext = event["sound"].lower().rsplit(".", 2)
                event["filepath"] = paths.download / (stem + "." + ext)
            events.append(event)


        events_df = pd.DataFrame(events)
        events_df[['language', 'modality']] = 'english', 'audio'
        events_df = extract_sequence_info(events_df)
        events_df = events_df.event.create_blocks(groupby='sentence')

        return events_df
