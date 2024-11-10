# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path
from osfclient import OSF
from tqdm import tqdm
import subprocess
import shutil
from dora import to_absolute_path


def download_osf(
    study: str, dset_dir: tp.Union[str, Path], success="osf_download.txt"
):

    print('dset_dir',dset_dir)
    dset_dir = to_absolute_path(Path(dset_dir))
    print("+++++++++++++++++++++++++++")
    print('dset_dir',dset_dir)
    
    assert dset_dir.parent.exists()
    dl_dir = dset_dir / "download"

    success_file = dl_dir / success
    if success_file.exists():
        return
    print(f"Downloading {study} to {dl_dir}...")
    project = OSF().project(study)

    store = list(project.storages)
    assert len(store) == 1
    assert store[0].name == "osfstorage"
    
    pbar = tqdm()
    for source in store[0].files:
        path = source.path
        if path.startswith("/"):
            path = path[1:]

        file_ = dl_dir / path

        if file_.exists():
            continue

        pbar.set_description(file_.name)
        file_.parent.mkdir(parents=True, exist_ok=True)
        with file_.open("wb") as fb:
            source.write_to(fb)

    with success_file.open("w") as f:
        f.write("success")
    print("Done!")



# To donwload manually, we can https://webdav.data.ru.nl/dccn/DSC_3011020.09_236_v1/ 
# and login with the following credentials:
# user
# password
def download_donders(study, dset_dir, parent="dccn", overwrite=False):
    dset_dir.mkdir(exist_ok=True, parents=True)
    success = dset_dir / "download" / "success.txt"
    if not success.exists() or overwrite:
        print(f"Downloading {study} to {dset_dir}...")
        user = input("user:").strip()
        password = input("password:").strip()
        command = "wget -r -nH -np --cut-dirs=1"
        command += " -U Mozilla"
        command += f" --user=0000-0002-6620-604x@orcid.org --password=SXKCBX5XGELJXE7YPTTJCXDYTI"
        command += f" https://webdav.data.ru.nl/dccn/DSC_3011020.09_236_v1"



        #user 0000-0002-6620-604x@orcid.org
        #password SXKCBX5XGELJXE7YPTTJCXDYTI
        # command = "wget -r -nH -np --cut-dirs=1"
        # command += " --no-check-certificate -U Mozilla"
        # command += f" --user={user} --password={password}"
        # command += f" https://webdav.data.donders.ru.nl/{parent}/{study}/"
        # print('?????????????????') #https://webdav.data.ru.nl/dccn/DSC_3011020.09_236_v1
        # print(f">>>>> https://webdav.data.donders.ru.nl/{parent}/{study}/")
        # print('?????????????????')

        command += f" -P {dset_dir}"
        command += ' -R "index.html*" -e robots=off'
        command += ' --show-progress'
        subprocess.run(command.split(), text=True, check=True)
        shutil.move(dset_dir / study, dset_dir / "download")
        print("Done!")
        with open(success, "w") as f:
            f.write("download success")
