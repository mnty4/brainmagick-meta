import torch
from .models.EEGNet import EEGNet

def test_spoken_eegNet(**params):

    ######################################
    # Wave based classifier
    ######################################
    batch_size = 2
    input_sample = {
        'eeg': torch.rand(batch_size, 122,1, 250),
        'label': torch.randint(0, 2, (batch_size,)),
        "epoch": 0,
        "iters": 0,
        "mask_ratio": 0.50,
    }
   
    # 'pretrain_time_mask', 'pretrain_channel_mask','freq_band_mask_welch_0.5'
    for mae_type in ['spoken_word']:
        model = EEGNet(**params)
        print('>>>>>>>>>>>>>>> clf model using {}'.format(mae_type))
        print('build success')
        ########################
        # forwad
        print('>>>>>>>>>>>>>>>forward')
        output = model(input_sample)
        for k in output:
            try:
                print(k, output[k].size())
            except:
                print(k, type(output[k]))
        ########################
        # generate
        print('>>>>>>>>>>>>>>>generate')
        output = model.generate(input_sample)
        for k in output:
            try:
                print(k, output[k].size())
            except:
                print(k, type(output[k]))

if __name__ == '__main__':
    test_spoken_eegNet(n_classes=2,
                 channels=122,
                 time_window=250,
                 )