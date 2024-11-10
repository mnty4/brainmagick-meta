from torch import optim
from copy import deepcopy
from torch.nn.modules import Module
import numpy as np
from lavis import registry
# import lavis.models.belt3_models.belt_clip_mae import Clip_TemporalConformer2D
import lavis.models.belt3_models.belt_clip_mae
import torch
from bm.models.classification_head import EEG_Encoder_Classification_Head
import os
import logging
from bm.meta_dataset3 import get_datasets
from torch.utils.data import DataLoader
from bm.setup_logging import configure_logging
base = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)

seed = 0

class RandomModule(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        

    def forward(self, x):
        return {
            'clf_logits': torch.rand((x['eeg'].shape[0], self.num_classes))
        }
    
    def generate(self, x):
        return self.forward(x)

# rng = np.random.RandomState(seed)
torch.manual_seed(seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test(model_dir, **kwargs):
    test_dset, word_index = get_datasets(is_train=False, **kwargs)
    model, inner_optim = load_model(os.path.join(base, model_dir), word_index, **kwargs)
    random_model = RandomModule(num_classes=len(word_index) // 2)
    
    test_loader = DataLoader(test_dset, batch_size=2, shuffle=True, collate_fn=lambda x: x)

    top_k_accs = top_k(model, test_loader, inner_optim=inner_optim, **kwargs)
    top_k_accs_random = top_k(random_model, test_loader, model_type='random', **kwargs)

    # write to stdout and file
    log_results(top_k_accs=top_k_accs, top_k_accs_random=top_k_accs_random, **kwargs)
    

def log_results(top_k_accs, top_k_accs_random, save_dir='evals', ks: list = [1, 5, 15], model_name='', **kwargs):
    v = 1
    dir_path = os.path.join(base, save_dir)
    os.makedirs(dir_path, exist_ok=True)
    save_path = os.path.join(dir_path, f'{model_name}_v{v}.txt')
    while os.path.exists(save_path):
        v += 1
        save_path = os.path.join(dir_path, f'{model_name}_v{v}.txt')
    with open(save_path, 'w') as f:
        msgs = []
        for k, acc, rand_acc in zip(ks, top_k_accs, top_k_accs_random):
            msg = f'Top {k} accuracy = {100 * acc:.2f}%, random = {100 * rand_acc:.2f}%.\n'
            logger.info(msg)
            msgs.append(msg)
            
        f.writelines(msgs)


def top_k(model: Module, test_loader, ks: list = [1, 5, 15], n_shot=0, unfreeze_encoder_on_support=False, inner_optim=None, loss_type='ce_loss', model_type="normal", **kwargs):
    correct_ks = np.zeros(len(ks))
    total = 0
    if unfreeze_encoder_on_support:
        model.unfreeze()

    for meta_batch in test_loader:
        for batches in meta_batch:
            
            old = deepcopy(model.state_dict())

            if model_type != 'random':
                model.train()
                
                for i in range(n_shot):
                    batch = batches[i]
                    batch['eeg'] = batch['eeg'].to(DEVICE)
                    batch['audio'] = batch['audio'].to(DEVICE)
                    batch['w_lbs'] = batch['w_lbs'].to(DEVICE)
                    output = model.generate(batch)
                    loss = output[loss_type].to(DEVICE)
                    inner_optim.zero_grad()
                    loss.backward()
                    inner_optim.step()

            model.eval()
            for i in range(n_shot, len(batches)):
                batch = batches[i]
                with torch.no_grad():
                    batch['eeg'] = batch['eeg'].to(DEVICE)
                    batch['audio'] = batch['audio'].to(DEVICE)
                    batch['w_lbs'] = batch['w_lbs'].to(DEVICE)
                    output = model.generate(batch)
                    logits = output['clf_logits'].to(DEVICE)
                print('len(ks)',len(ks))
                print('ks',ks, 'logits', logits.shape)
                # ks [1, 5, 15, 50, 500, 1500] logits torch.Size([8, 1391])

                for j in range(len(ks)):
                    print('j',j)
                    _, top_k_indices = torch.topk(logits, k=ks[j], dim=-1)
                    correct = top_k_indices.eq(batch['w_lbs'].view(-1, 1).expand_as(top_k_indices))
                    correct_ks[j] += sum(correct.sum(dim=-1)).item()
                total += logits.shape[0]
                
            
            model.load_state_dict(old)

    return correct_ks / total


def load_model(model_dir, word_index, type='classifier_head', inner_lr=0.001, **kwargs):
    """
    'meta_epoch': meta_epoch,
            'batch': i,
            'is_meta_train': True if meta_optim else False,
            'model_state_dict': model.state_dict(),
            'meta_optim_state_dict': meta_optim.state_dict() if meta_optim else None,
            'inner_optim_state_dict': inner_optim.state_dict(),
            'loss': loss,
"""
    loaded = torch.load(model_dir)
    model_state_dict = loaded['model_state_dict']
    model_cls = registry.get_model_class("Clip-Audio-Emformer-Lukas")

    if type == 'classifier_head':
        clip_model = model_cls.from_config(type='base', use_classifier=False)
        model = EEG_Encoder_Classification_Head(eeg_encoder=clip_model.eeg_encoder, num_classes=len(word_index) // 2, eeg_projection=clip_model.eeg_projection)
    elif type == 'classifier_combined':
        model = model_cls.from_config(type='base', num_classes=len(word_index) // 2, use_classifier=True)

    if loaded['is_meta_train']:
        inner_optim = optim.SGD(model.parameters(), inner_lr)
    else:
        inner_optim = optim.Adam(model.parameters(), inner_lr)

    model.load_state_dict(model_state_dict)
    model.to(DEVICE)
    return model, inner_optim

def test_model():
    pass



# top-1, top-5 word score

# eval_loss = val_result['eval_loss']
#         results = val_result['results']
#         # print('results',len(results))
#         # need to save 3 files, one is the mectrics result, one is the prediction for each word location, and also the confusion matrix for classifcation, currently dont save the reconstruction. 
#         # --------------------- classification ---------------------
#         # print('>>>>>>>>>>>>>>>>>>>>>>>>>> Evaluation Debug <<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#         gt_label_list = np.array([item['label'].numpy() for item in results])   # [n]
#         pred_score_list = np.array([item['logits'].numpy() for item in results])# [n, num_classes]
#         prediction_correct_clf = [False for _ in results]
#         predictions = np.argsort(pred_score_list, axis=1) # ascending order  the last one is the largest probability
        
#         start_time = time.time()
        
#         top1_predicted_label = predictions[:,-1] # [n]
#         top1_scores = [pred_score_list[i,top1_predicted_label[i]] for i in range(len(top1_predicted_label))]
#         top1_acc = np.mean(top1_predicted_label == gt_label_list)
#         # print('top1_acc',top1_acc, 'time',time.time()-start_time)
#         top1_word = [self.id2word[item] for item in top1_predicted_label]
#         # continue to calculate top 5
#         top5_predicted_label = predictions[:,-5:]
#         top5_scores = [pred_score_list[i,top5_predicted_label[i]] for i in range(len(top5_predicted_label))]
#         top5_acc = np.mean([1 if gt_label_list[i] in top5_predicted_label[i] else 0 for i in range(len(gt_label_list))])
#         # print('top5_acc',top5_acc, 'time',time.time()-start_time)
#         top5_words = [[self.id2word[item] for item in top5_predicted_label[i]] for i in range(len(top5_predicted_label))]      
#         # continue to calculate top 10
#         top10_predicted_label = predictions[:,-10:]
#         top10_scores = [pred_score_list[i,top10_predicted_label[i]] for i in range(len(top10_predicted_label))]
#         top10_acc = np.mean([1 if gt_label_list[i] in top10_predicted_label[i] else 0 for i in range(len(gt_label_list))])
#         # print('top10_acc',top10_acc, 'time',time.time()-start_time)
#         top10_words = [[self.id2word[item] for item in top10_predicted_label[i]] for i in range(len(top10_predicted_label))]     
#         # continue to calculate top 15
#         top15_predicted_label = predictions[:,-15:]
#         top15_scores = [pred_score_list[i,top15_predicted_label[i]] for i in range(len(top15_predicted_label))]
#         top15_acc = np.mean([1 if gt_label_list[i] in top15_predicted_label[i] else 0 for i in range(len(gt_label_list))])
#         # print('top15_acc',top15_acc, 'time',time.time()-start_time)
#         top15_words = [[self.id2word[item] for item in top15_predicted_label[i]] for i in range(len(top15_predicted_label))]

#         # continue to calculate top 20
#         top20_predicted_label = predictions[:,-20:]
#         top20_scores = [pred_score_list[i,top20_predicted_label[i]] for i in range(len(top20_predicted_label))]
#         top20_acc = np.mean([1 if gt_label_list[i] in top20_predicted_label[i] else 0 for i in range(len(gt_label_list))])
#         top20_words = [[self.id2word[item] for item in top20_predicted_label[i]] for i in range(len(top20_predicted_label))] 
#         # print('top20_acc',top20_acc)
#         # assign prediction_correct = 1 if gt word in top 20 words
#         for i in range(len(gt_label_list)):
#             if gt_label_list[i] in top20_predicted_label[i]:
#                 prediction_correct_clf[i] = True
#         # Now calculate some balanced accuracy. 
#         # First get a list of all prediction labels. 
#         # Then get a list of all GT labels
#         # Then we use the sklearn method. 
#         # print('gt_label_list>>>\n',gt_label_list)
#         # print('predictions_list>>>\n',predictions.shape)
#         # argmax
#         predicted_label_top1 =   []
#         predicted_label_top5 =   []
#         predicted_label_top10 =  []
#         predicted_label_top15 =  []
#         predicted_label_top20 =  []
#         for i in range(len(gt_label_list)):
#             if predictions[i,-1] == gt_label_list[i]:
#                 predicted_label_top1.append(gt_label_list[i])
#             else:
#                 predicted_label_top1.append(predictions[i,-1])
            
#             if gt_label_list[i] in predictions[i,-5:]:
#                 predicted_label_top5.append(gt_label_list[i])
#             else:
#                 predicted_label_top5.append(predictions[i,-1])

#             if gt_label_list[i] in predictions[i,-10:]:
#                 predicted_label_top10.append(gt_label_list[i])
#             else:
#                 predicted_label_top10.append(predictions[i,-1])
#             if gt_label_list[i] in predictions[i,-15:]:
#                 predicted_label_top15.append(gt_label_list[i])
#             else:
#                 predicted_label_top15.append(predictions[i,-1])
#             if gt_label_list[i] in predictions[i,-20:]:
#                 predicted_label_top20.append(gt_label_list[i])
#             else:
#                 predicted_label_top20.append(predictions[i,-1])
#         # do the calculation
#         clf_top1_acc_balanced = balanced_accuracy_score(gt_label_list, predicted_label_top1, adjusted=False)
#         clf_top5_acc_balanced = balanced_accuracy_score(gt_label_list, predicted_label_top5, adjusted=False)
#         clf_top10_acc_balanced = balanced_accuracy_score(gt_label_list, predicted_label_top10, adjusted=False)
#         clf_top15_acc_balanced = balanced_accuracy_score(gt_label_list, predicted_label_top15, adjusted=False)
#         clf_top20_acc_balanced = balanced_accuracy_score(gt_label_list, predicted_label_top20, adjusted=False)    

#         precision_scores_top1= precision_score(gt_label_list, predicted_label_top1, average='weighted')
#         precision_scores_top5= precision_score(gt_label_list, predicted_label_top5, average='weighted')
#         precision_scores_top10= precision_score(gt_label_list, predicted_label_top10, average='weighted')
#         precision_scores_top15= precision_score(gt_label_list, predicted_label_top15, average='weighted')
#         precision_scores_top20= precision_score(gt_label_list, predicted_label_top20, average='weighted')
        




















 
#         # --------------------- CLIP Similarity ---------------------
#         eeg_fea = torch.stack([item['eeg_embed'] for item in results])
#         word_fea = torch.stack([item['word_embed'] for item in results])
#         gt_words = [item['word'] for item in results]
#         prediction_correct_clip = [False for _ in results]
#         # only use unique words from gt_words, get the index 
#         unique_word_index = []
#         unique_words = []
#         for i in range(len(gt_words)):
#             if gt_words[i] not in unique_words:
#                 unique_words.append(gt_words[i])
#                 unique_word_index.append(i)
#         # select the unique words from eeg_fea and word_fea
#         word_fea_unique = word_fea[unique_word_index] 
#         word_unique = [gt_words[i] for i in unique_word_index]
#         acc_count_1= 0
#         acc_count_5 = 0
#         acc_count_10 = 0
#         acc_count_15 = 0
#         acc_count_20 = 0
#         top1_clip_predicated_words =[]
#         top5_clip_predicated_words =[]
#         top10_clip_predicated_words =[]
#         top15_clip_predicated_words =[]
#         top20_clip_predicated_words =[]
#         # start to find the top k for each eeg
#         for i in tqdm(range(len(eeg_fea))):
#             # print('eeg_fea',eeg_fea.shape,'word_fea',word_fea_unique.shape)
#             topk = self.find_matches(eeg_fea[i],word_fea_unique, k=20)
#             # print('topk',topk.values.device,topk.indices.device)
#             topk_value = topk.values.cpu().numpy()
#             top_index = topk.indices.cpu().numpy()
#             # get the index of the unique words
#             pred_words_top_20 = [word_unique[i] for i in top_index] 
#             if gt_words[i] in pred_words_top_20:
#                 acc_count_20 +=1
#             if gt_words[i] in pred_words_top_20[:15]:
#                 acc_count_15 +=1
#             if gt_words[i] in pred_words_top_20[:10]:
#                 acc_count_10 +=1
#             if gt_words[i] in pred_words_top_20[:5]:
#                 acc_count_5 +=1
#             if gt_words[i] in pred_words_top_20[:1]:
#                 acc_count_1 +=1
#             top1_clip_predicated_words.append(pred_words_top_20[:1])
#             top5_clip_predicated_words.append(pred_words_top_20[:5])
#             top10_clip_predicated_words.append(pred_words_top_20[:10])
#             top15_clip_predicated_words.append(pred_words_top_20[:15])
#             top20_clip_predicated_words.append(pred_words_top_20[:20])
#             prediction_correct_clip[i] = gt_words[i] in pred_words_top_20
#             del topk
#             del topk_value
#             del top_index
#             # if i < 5:
#             #     print('top1 index',top_index[:1],'top1 value',topk_value[:1],'top1 word',pred_words_top_20[:1])
#             #     print('top5 index',top_index[:5],'top5 value',topk_value[:5],'top5 word',pred_words_top_20[:5])
#             #     print('top10 index',top_index[:10],'top10 value',topk_value[:10],'top10 word',pred_words_top_20[:10])
#             #     print('top15 index',top_index[:15],'top15 value',topk_value[:15],'top15 word',pred_words_top_20[:15])
#             #     print('top20 index',top_index[:20],'top20 value',topk_value[:20],'top20 word',pred_words_top_20[:20])
#         # calculate the accuracy
#         acc_1 = acc_count_1/len(eeg_fea)
#         acc_5 = acc_count_5/len(eeg_fea)
#         acc_10 = acc_count_10/len(eeg_fea)
#         acc_15 = acc_count_15/len(eeg_fea)
#         acc_20 = acc_count_20/len(eeg_fea)
#         # print('clip acc_1',acc_1,'acc_5',acc_5,'acc_10',acc_10,'acc_20',acc_20)
#         # print('done eval clip output')
#         # --------------------- Confusion matrix ---------------------
#         predicted_label = np.argmax(pred_score_list, axis=1) # [n]
#         cm_clf = confusion_matrix(gt_label_list, predicted_label)
#         if cm_clf.shape[0] > 100:
#             cm_clf = cm_clf[:100,:100]
#         fig, axs = plt.subplots(1, 2, figsize=(10,8))
#         axs[0].set_title('Accuracy {} Classifier'.format(top1_acc))
#         axs[0].imshow(cm_clf, cmap='binary')
#         if cm_clf.shape[0] < 10:
#             for i in range(cm_clf.shape[0]):
#                 for j in range(cm_clf.shape[1]):
#                     axs[0].text(j, i, cm_clf[i, j], ha='center', va='center', color='r')        
#         predicted_label_clip_top_20 = []
#         for i in range(len(gt_label_list)):
#             if gt_label_list[i] in top20_predicted_label[i]:
#                 predicted_label_clip_top_20.append( gt_label_list[i])
#             else:
#                 predicted_label_clip_top_20.append(self.word2id[top20_clip_predicated_words[i][0]])
#         predicted_label_clip_top_20 = np.array(predicted_label_clip_top_20)
#         cm_clip = confusion_matrix(gt_label_list, predicted_label_clip_top_20)
#         if cm_clip.shape[0] > 100:
#             cm_clip = cm_clip[:100,:100]
#         axs[1].set_title('Accuracy {} CLIP'.format(acc_20))
#         axs[1].imshow(cm_clip, cmap='binary')
#         # plot classifier confusion matrix
#         fig_name = 'prediction_result-{}-{}'.format(split_name, epoch)
#         img_output_path = os.path.join(registry.get_path("result_dir"), 'prediction')
#         if not os.path.exists(img_output_path):
#             os.makedirs(img_output_path)        
#         fig.savefig(os.path.join(img_output_path, f'{fig_name}.png'))
#         plt.close(fig)
#         plt.cla()
#         plt.clf()        


if __name__ == '__main__':
    configure_logging()
    model_dir = '/home/lukas/projects/brainmagick/bm/runs/non_meta_v26/best_checkpoint_0_142.pth'
    # model_dir = '/home/lukas/projects/brainmagick/bm/runs/non_meta_v1/best_checkpoint_3_142.pth'

    ks = [1, 5, 15, 50, 500, 1500]
    # test(model_dir, seed=42, ks=ks)
    test_kwargs = {
        'mini_batches_per_trial': 8, 
        'samples_per_mini_batch': 8, 
        'batch_size': 1,
        'unfreeze_encoder_on_support': False,
        'n_shot': 4,
    }
    test(model_dir, seed=42, ks=ks, type='classifier_combined', loss_type='combined_loss', **test_kwargs)