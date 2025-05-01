import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.fusion import Fusion
from models.ehr_models import LSTM
from models.cxr_models import CXRModels
from .trainer import Trainer
import pandas as pd


import numpy as np
from sklearn import metrics

# class FusionTrainer(Trainer):
#     def __init__(self, 
#         train_dl, 
#         val_dl, 
#         args,
#         test_dl=None
#         ):

#         super(FusionTrainer, self).__init__(args)
#         self.epoch = 0 
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         self.args = args
#         self.train_dl = train_dl
#         self.val_dl = val_dl
#         self.test_dl = test_dl

#         self.ehr_model = LSTM(input_dim=76, num_classes=args.num_classes, hidden_dim=args.dim, dropout=args.dropout, layers=args.layers).to(self.device)
#         self.cxr_model = CXRModels(self.args, self.device).to(self.device)


#         self.model = Fusion(args, self.ehr_model, self.cxr_model ).to(self.device)
#         self.init_fusion_method()

#         self.loss = nn.BCELoss()

#         self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
#         self.load_state()
#         print(self.ehr_model)
#         print(self.optimizer)
#         print(self.loss)
#         self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

#         self.best_auroc = 0
#         self.best_stats = None
#         # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99) 
#         self.epochs_stats = {'loss train': [], 'loss val': [], 'auroc val': [], 'loss align train': [], 'loss align val': []}
    
#     def init_fusion_method(self):

#         '''
#         for early fusion
#         load pretrained encoders and 
#         freeze both encoders
#         ''' 

#         if self.args.load_state_ehr is not None:
#             self.load_ehr_pheno(load_state=self.args.load_state_ehr)
#         if self.args.load_state_cxr is not None:
#             self.load_cxr_pheno(load_state=self.args.load_state_cxr)
        
#         if self.args.load_state is not None:
#             self.load_state()


#         if 'uni_ehr' in self.args.fusion_type:
#             self.freeze(self.model.cxr_model)
#         elif 'uni_cxr' in self.args.fusion_type:
#             self.freeze(self.model.ehr_model)
#         elif 'late' in self.args.fusion_type:
#             self.freeze(self.model)
#         elif 'early' in self.args.fusion_type:
#             self.freeze(self.model.cxr_model)
#             self.freeze(self.model.ehr_model)
#         elif 'lstm' in self.args.fusion_type:
#             # self.freeze(self.model.cxr_model)
#             # self.freeze(self.model.ehr_model)
#             pass

#     def train_epoch(self):
#         print(f'starting train epoch {self.epoch}')
#         epoch_loss = 0
#         epoch_loss_align = 0
#         outGT = torch.FloatTensor().to(self.device)
#         outPRED = torch.FloatTensor().to(self.device)
#         steps = len(self.train_dl)
#         for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (self.train_dl):
#             y = self.get_gt(y_ehr, y_cxr)
#             x = torch.from_numpy(x).float()
#             x = x.to(self.device)
#             y = y.to(self.device)
#             img = img.to(self.device)

#             output = self.model(x, seq_lengths, img, pairs)
            
#             pred = output[self.args.fusion_type].squeeze()
#             loss = self.loss(pred, y)
#             epoch_loss += loss.item()
#             if self.args.align > 0.0:
#                 loss = loss + self.args.align * output['align_loss']
#                 epoch_loss_align = epoch_loss_align + self.args.align * output['align_loss'].item()

#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#             outPRED = torch.cat((outPRED, pred), 0)
#             outGT = torch.cat((outGT, y), 0)

#             if i % 100 == 9:
#                 eta = self.get_eta(self.epoch, i)
#                 print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f} loss align {epoch_loss_align/i:0.4f}")
#         ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train')
#         self.epochs_stats['loss train'].append(epoch_loss/i)
#         self.epochs_stats['loss align train'].append(epoch_loss_align/i)
#         return ret
    
#     def validate(self, dl):
#         print(f'starting val epoch {self.epoch}')
#         epoch_loss = 0
#         epoch_loss_align = 0
#         # ehr_features = torch.FloatTensor()
#         # cxr_features = torch.FloatTensor()
#         outGT = torch.FloatTensor().to(self.device)
#         outGT = torch.FloatTensor().to(self.device)
#         outPRED = torch.FloatTensor().to(self.device)

#         with torch.no_grad():
#             for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (dl):
#                 y = self.get_gt(y_ehr, y_cxr)

#                 x = torch.from_numpy(x).float()
#                 x = Variable(x.to(self.device), requires_grad=False)
#                 y = Variable(y.to(self.device), requires_grad=False)
#                 img = img.to(self.device)
#                 output = self.model(x, seq_lengths, img, pairs)
                
#                 pred = output[self.args.fusion_type]
                
#                 if self.args.fusion_type != 'uni_cxr':
#                     if len(pred.shape) > 1:
#                          pred = pred.squeeze()
                           
#                 loss = self.loss(pred, y)
#                 epoch_loss += loss.item()
#                 if self.args.align > 0.0:

#                     epoch_loss_align +=  output['align_loss'].item()
#                 outPRED = torch.cat((outPRED, pred), 0)
#                 outGT = torch.cat((outGT, y), 0)
#                 # if 'ehr_feats' in output:
#                 #     ehr_features = torch.cat((ehr_features, output['ehr_feats'].data.cpu()), 0)
#                 # if 'cxr_feats' in output:
#                 #     cxr_features = torch.cat((cxr_features, output['cxr_feats'].data.cpu()), 0)
        
#         self.scheduler.step(epoch_loss/len(self.val_dl))

#         print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f} \t{epoch_loss_align/i:0.5f}")
#         ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
#         np.save(f'{self.args.save_dir}/pred.npy', outPRED.data.cpu().numpy()) 
#         np.save(f'{self.args.save_dir}/gt.npy', outGT.data.cpu().numpy()) 

#         # if 'ehr_feats' in output:
#         #     np.save(f'{self.args.save_dir}/ehr_features.npy', ehr_features.data.cpu().numpy())
#         # if 'cxr_feats' in output:
#         #     np.save(f'{self.args.save_dir}/cxr_features.npy', cxr_features.data.cpu().numpy())

#         self.epochs_stats['auroc val'].append(ret['auroc_mean'])

#         self.epochs_stats['loss val'].append(epoch_loss/i)
#         self.epochs_stats['loss align val'].append(epoch_loss_align/i)
#         # print(f'true {outGT.data.cpu().numpy().sum()}/{outGT.data.cpu().numpy().shape}')
#         # print(f'true {outGT.data.cpu().numpy().sum()/outGT.data.cpu().numpy().shape[0]} ({outGT.data.cpu().numpy().sum()}/{outGT.data.cpu().numpy().shape[0]})')

#         return ret

#     def compute_late_fusion(self, y_true, uniout_cxr, uniout_ehr):
#         y_true = np.array(y_true)
#         predictions_cxr = np.array(uniout_cxr)
#         predictions_ehr = np.array(uniout_ehr)
#         best_weights = np.ones(y_true.shape[-1])
#         best_auroc = 0.0
#         weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#         for class_idx in range(y_true.shape[-1]):
#             for weight in weights:
#                 predictions = (predictions_ehr * best_weights) + (predictions_cxr * (1-best_weights))
#                 predictions[:, class_idx] = (predictions_ehr[:, class_idx] * weight) + (predictions_cxr[:, class_idx] * 1-weight)
#                 auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
#                 auroc_mean = np.mean(np.array(auc_scores))
#                 if auroc_mean > best_auroc:
#                     best_auroc = auroc_mean
#                     best_weights[class_idx] = weight
#                 # predictions = weight * predictions_cxr[]


#         predictions = (predictions_ehr * best_weights) + (predictions_cxr * (1-best_weights))
#         print(best_weights)

#         auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
#         ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
#                                             average="micro")
#         ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
#                                             average="macro")
#         ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
#                                                 average="weighted")
        
#         # print(np.mean(np.array(auc_scores)

#         # print()
#         best_stats = {"auc_scores": auc_scores,
#                 "ave_auc_micro": ave_auc_micro,
#                 "ave_auc_macro": ave_auc_macro,
#                 "ave_auc_weighted": ave_auc_weighted,
#                 "auroc_mean": np.mean(np.array(auc_scores))
#                 }
#         self.print_and_write(best_stats , isbest=True, prefix='late fusion weighted average')

#         return best_stats 
#     def eval_age(self):

#         print('validating ... ')
           
#         patiens = pd.read_csv('data/physionet.org/files/mimic-iv-1.0/core/patients.csv')
#         subject_ids = np.array([int(item.split('_')[0]) for item in self.test_dl.dataset.ehr_files_paired])

#         selected = patiens[patiens.subject_id.isin(subject_ids)]
#         start = 18
#         copy_ehr = np.copy(self.test_dl.dataset.ehr_files_paired)
#         copy_cxr = np.copy(self.test_dl.dataset.cxr_files_paired)
#         self.model.eval()
#         step = 20
#         for i in range(20, 100, step):
#             subjects = selected.loc[((selected.anchor_age >= start) & (selected.anchor_age < i + step))].subject_id.values
#             indexes = [jj for (jj, subject) in enumerate(subject_ids) if  subject in subjects]
            
            
#             self.test_dl.dataset.ehr_files_paired = copy_ehr[indexes]
#             self.test_dl.dataset.cxr_files_paired = copy_cxr[indexes]

#             print(len(indexes))
#             ret = self.validate(self.test_dl)
#             print(f"{start}-{i + step} & {len(indexes)} & & & {ret['auroc_mean']:0.3f} & {ret['auprc_mean']:0.3f}")

#             self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} val', filename=f'results_test_{start}_{i + step}.txt')

#             # print(f"{start}-{i + step} & {len(indexes)} & & & {ret['auroc_mean']:0.3f} & {ret['auprc_mean']:0.3f}")
#             # print(f"{start}-{i + 10} & {len(indexes)} & & & {ret['auroc_mean']:0.3f} & {ret['auprc_mean']:0.3f}")
#             # self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} age_{start}_{i + 10}_{len(indexes)}', filename='results_test.txt')
#             start = i + step
#     def test(self):
#         print('validating ... ')
#         self.epoch = 0
#         self.model.eval()
#         ret = self.validate(self.val_dl)
#         self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} val', filename='results_val.txt')
#         self.model.eval()
#         ret = self.validate(self.test_dl)
#         self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename='results_test.txt')
#         return

#     def eval(self):
#         # self.eval_age()
#         print('validating ... ')
#         self.epoch = 0
#         self.model.eval()
#         # ret = self.validate(self.val_dl)
#         # self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} val', filename='results_val.txt')
#         # self.model.eval()
#         ret = self.validate(self.test_dl)
#         self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename='results_test.txt')
#         return
#     def train(self):
#         print(f'running for fusion_type {self.args.fusion_type}')
#         for self.epoch in range(self.start_epoch, self.args.epochs):
#             self.model.eval()
#             ret = self.validate(self.val_dl)
#             self.save_checkpoint(prefix='last')

#             if self.best_auroc < ret['auroc_mean']:
#                 self.best_auroc = ret['auroc_mean']
#                 self.best_stats = ret
#                 self.save_checkpoint()
#                 # print(f'saving best AUROC {ret["ave_auc_micro"]:0.4f} checkpoint')
#                 self.print_and_write(ret, isbest=True)
#                 self.patience = 0
#             else:
#                 self.print_and_write(ret, isbest=False)
#                 self.patience+=1

#             self.model.train()
#             self.train_epoch()
#             self.plot_stats(key='loss', filename='loss.pdf')
#             self.plot_stats(key='auroc', filename='auroc.pdf')
#             if self.patience >= self.args.patience:
#                 break
#         self.print_and_write(self.best_stats , isbest=True)

class FusionTrainer(Trainer):
    def __init__(self,
        train_dl,
        val_dl,
        args,
        test_dl=None
        ):
        print("--- [FusionTrainer Init] Starting Initialization ---")
        super(FusionTrainer, self).__init__(args)
        print("--- [FusionTrainer Init] Base Trainer Initialized ---")
        self.epoch = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"--- [FusionTrainer Init] Using device: {self.device} ---")

        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        print("--- [FusionTrainer Init] DataLoaders assigned ---")

        print("--- [FusionTrainer Init] Initializing EHR model ---")
        self.ehr_model = LSTM(input_dim=76, num_classes=args.num_classes, hidden_dim=args.dim, dropout=args.dropout, layers=args.layers).to(self.device)
        print("--- [FusionTrainer Init] Initializing CXR model ---")
        self.cxr_model = CXRModels(self.args, self.device).to(self.device)
        print("--- [FusionTrainer Init] Initializing Fusion model ---")
        self.model = Fusion(args, self.ehr_model, self.cxr_model ).to(self.device)

        print("--- [FusionTrainer Init] Initializing Fusion Method specific settings ---")
        self.init_fusion_method()
        print("--- [FusionTrainer Init] Fusion Method Initialized ---")


        self.loss = nn.BCELoss()
        print(f"--- [FusionTrainer Init] Loss function initialized: {type(self.loss).__name__} ---")

        self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
        print(f"--- [FusionTrainer Init] Optimizer initialized: {type(self.optimizer).__name__} ---")
        self.load_state() # Assuming this might print something if state is loaded
        print("--- [FusionTrainer Init] Attempted to load state ---")
        # print(self.ehr_model) # Optional: print model structures if needed
        # print(self.optimizer)
        # print(self.loss)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')
        print(f"--- [FusionTrainer Init] Scheduler initialized: {type(self.scheduler).__name__} ---")


        self.best_auroc = 0
        self.best_stats = None
        self.epochs_stats = {'loss train': [], 'loss val': [], 'auroc val': [], 'loss align train': [], 'loss align val': []}
        print("--- [FusionTrainer Init] Stats tracking initialized ---")
        print("--- [FusionTrainer Init] Initialization Complete ---")


    def init_fusion_method(self):
        print("--- [init_fusion_method] Starting ---")
        '''
        for early fusion
        load pretrained encoders and
        freeze both encoders
        '''

        if self.args.load_state_ehr is not None:
            print(f"--- [init_fusion_method] Loading EHR state from: {self.args.load_state_ehr} ---")
            self.load_ehr_pheno(load_state=self.args.load_state_ehr) # Assuming this method exists in base Trainer
        if self.args.load_state_cxr is not None:
            print(f"--- [init_fusion_method] Loading CXR state from: {self.args.load_state_cxr} ---")
            self.load_cxr_pheno(load_state=self.args.load_state_cxr) # Assuming this method exists in base Trainer

        if self.args.load_state is not None:
            print(f"--- [init_fusion_method] Loading overall Fusion state from: {self.args.load_state} ---")
            self.load_state() # Assuming this method exists in base Trainer


        if 'uni_ehr' in self.args.fusion_type:
            print("--- [init_fusion_method] Freezing CXR model for uni_ehr ---")
            self.freeze(self.model.cxr_model) # Assuming this method exists in base Trainer
        elif 'uni_cxr' in self.args.fusion_type:
            print("--- [init_fusion_method] Freezing EHR model for uni_cxr ---")
            self.freeze(self.model.ehr_model)
        elif 'late' in self.args.fusion_type:
            print("--- [init_fusion_method] Freezing entire Fusion model for late fusion ---")
            self.freeze(self.model)
        elif 'early' in self.args.fusion_type:
            print("--- [init_fusion_method] Freezing CXR and EHR models for early fusion ---")
            self.freeze(self.model.cxr_model)
            self.freeze(self.model.ehr_model)
        elif 'lstm' in self.args.fusion_type:
            print("--- [init_fusion_method] No freezing specified for lstm fusion type ---")
            # self.freeze(self.model.cxr_model)
            # self.freeze(self.model.ehr_model)
            pass
        else:
             print(f"--- [init_fusion_method] Unknown fusion type '{self.args.fusion_type}', no specific freezing applied ---")
        print("--- [init_fusion_method] Finished ---")

    def train_epoch(self):
        print(f"--- [Train Epoch {self.epoch}] Starting ---")
        self.model.train() # Ensure model is in training mode
        epoch_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        print(f"--- [Train Epoch {self.epoch}] Total steps: {steps} ---")
        for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (self.train_dl):
            # print(f"--- [Train Epoch {self.epoch}, Step {i}/{steps}] Data loaded ---") # Can be very verbose
            y = self.get_gt(y_ehr, y_cxr) # Assuming this method exists in base Trainer
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            img = img.to(self.device)
            # print(f"--- [Train Epoch {self.epoch}, Step {i}/{steps}] Data moved to device ---")

            # print(f"--- [Train Epoch {self.epoch}, Step {i}/{steps}] Calling model forward ---")
            output = self.model(x, seq_lengths, img, pairs)
            # print(f"--- [Train Epoch {self.epoch}, Step {i}/{steps}] Model forward complete ---")

            pred = output[self.args.fusion_type].squeeze()
            # print(f"--- [Train Epoch {self.epoch}, Step {i}/{steps}] Calculating loss ---")
            loss = self.loss(pred, y)
            # print(f"--- [Train Epoch {self.epoch}, Step {i}/{steps}] Loss calculated: {loss.item():.4f} ---")
            epoch_loss += loss.item()
            if self.args.align > 0.0:
                # print(f"--- [Train Epoch {self.epoch}, Step {i}/{steps}] Adding alignment loss ---")
                loss = loss + self.args.align * output['align_loss']
                epoch_loss_align = epoch_loss_align + self.args.align * output['align_loss'].item()
                # print(f"--- [Train Epoch {self.epoch}, Step {i}/{steps}] Alignment loss added: {self.args.align * output['align_loss'].item():.4f} ---")


            # print(f"--- [Train Epoch {self.epoch}, Step {i}/{steps}] Zeroing gradients ---")
            self.optimizer.zero_grad()
            # print(f"--- [Train Epoch {self.epoch}, Step {i}/{steps}] Performing backward pass ---")
            loss.backward()
            # print(f"--- [Train Epoch {self.epoch}, Step {i}/{steps}] Optimizer step ---")
            self.optimizer.step()
            # print(f"--- [Train Epoch {self.epoch}, Step {i}/{steps}] Step complete ---")


            # print(f"--- [Train Epoch {self.epoch}, Step {i}/{steps}] Concatenating results ---")
            outPRED = torch.cat((outPRED, pred.detach()), 0) # Detach to avoid holding onto graph
            outGT = torch.cat((outGT, y.detach()), 0)
            # print(f"--- [Train Epoch {self.epoch}, Step {i}/{steps}] Results concatenated ---")


            if i % 100 == 99: # Print every 100 steps (adjust frequency as needed)
                eta = self.get_eta(self.epoch, i) # Assuming this method exists in base Trainer
                current_loss = epoch_loss / (i + 1)
                current_align_loss = epoch_loss_align / (i + 1) if self.args.align > 0.0 else 0.0
                print(f"--- [Train Epoch {self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: {self.optimizer.param_groups[0]['lr']:0.4E} loss: {current_loss:0.5f} loss align {current_align_loss:0.4f} ---")

        print(f"--- [Train Epoch {self.epoch}] Calculating AUROC ---")
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train') # Assuming this method exists
        avg_epoch_loss = epoch_loss/len(self.train_dl) # Use len(dl) for average
        avg_epoch_align_loss = epoch_loss_align/len(self.train_dl) if self.args.align > 0.0 else 0.0
        self.epochs_stats['loss train'].append(avg_epoch_loss)
        self.epochs_stats['loss align train'].append(avg_epoch_align_loss)
        print(f"--- [Train Epoch {self.epoch}] Finished. Avg Loss: {avg_epoch_loss:.5f}, Avg Align Loss: {avg_epoch_align_loss:.4f} ---")
        return ret

    def validate(self, dl):
        print(f"--- [Validate Epoch {self.epoch}] Starting Validation ---")
        self.model.eval() # Ensure model is in evaluation mode
        epoch_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(dl)
        print(f"--- [Validate Epoch {self.epoch}] Total steps: {steps} ---")

        with torch.no_grad():
            print(f"--- [Validate Epoch {self.epoch}] Entering torch.no_grad() block ---")
            for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate (dl):
                print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] Batch loaded ---") # Print batch index
                y = self.get_gt(y_ehr, y_cxr) # Assuming this method exists in base Trainer
                # print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] Ground truth determined ---")

                x = torch.from_numpy(x).float()
                # print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] EHR data converted to tensor ---")
                # x = Variable(x.to(self.device), requires_grad=False) # Deprecated, just use .to(device)
                x = x.to(self.device)
                # y = Variable(y.to(self.device), requires_grad=False) # Deprecated
                y = y.to(self.device)
                img = img.to(self.device)
                print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] Data moved to device: {self.device} ---")


                print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] Calling model forward... ---")
                try:
                    output = self.model(x, seq_lengths, img, pairs)
                    print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] Model forward complete ---")
                except Exception as e:
                    print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] ERROR during model forward: {e} ---")
                    # Consider re-raising the exception or handling it appropriately
                    raise e # Re-raise to stop execution if there's a critical error

                pred = output[self.args.fusion_type]
                # print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] Prediction extracted ---")

                # Handle potential dimension issues
                if self.args.fusion_type != 'uni_cxr': # Why this specific condition? Check if necessary
                     if pred.dim() > 1 and pred.shape[1] == 1: # Check if it needs squeezing (e.g., shape [batch, 1])
                         print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] Squeezing prediction tensor from {pred.shape} ---")
                         pred = pred.squeeze(1)
                     elif pred.dim() == 0: # Handle scalar tensor if necessary
                         print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] Unsqueezing scalar prediction tensor ---")
                         pred = pred.unsqueeze(0)


                # Ensure y has the same shape as pred for BCELoss
                if pred.shape != y.shape:
                     print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] WARNING: Prediction shape {pred.shape} differs from Target shape {y.shape}. Reshaping target. ---")
                     # This might indicate an issue elsewhere, but let's try reshaping y
                     try:
                         y = y.view_as(pred)
                         print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] Target reshaped to {y.shape} ---")
                     except RuntimeError as reshape_err:
                          print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] ERROR: Could not reshape target to match prediction: {reshape_err} ---")
                          # Decide how to handle this: skip batch, raise error, etc.
                          continue # Skip this batch


                print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] Calculating validation loss... ---")
                try:
                    loss = self.loss(pred, y)
                    print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] Validation loss calculated: {loss.item():.4f} ---")
                    epoch_loss += loss.item()
                except Exception as e:
                    print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] ERROR during loss calculation: {e} ---")
                    print(f"--- Pred shape: {pred.shape}, Pred dtype: {pred.dtype}, Pred device: {pred.device}")
                    print(f"--- Target shape: {y.shape}, Target dtype: {y.dtype}, Target device: {y.device}")
                    # Handle or re-raise
                    raise e


                if self.args.align > 0.0:
                    align_loss_val = output['align_loss'].item()
                    print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] Adding alignment loss: {align_loss_val:.4f} ---")
                    epoch_loss_align += align_loss_val


                print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] Concatenating results... ---")
                outPRED = torch.cat((outPRED, pred), 0)
                outGT = torch.cat((outGT, y), 0)
                print(f"--- [Validate Epoch {self.epoch}, Step {i+1}/{steps}] Results concatenated. outPRED shape: {outPRED.shape}, outGT shape: {outGT.shape} ---")

            print(f"--- [Validate Epoch {self.epoch}] Exited torch.no_grad() block ---")

        avg_epoch_loss = epoch_loss / len(dl) # Use len(dl) for average
        avg_epoch_align_loss = epoch_loss_align / len(dl) if self.args.align > 0.0 else 0.0

        print(f"--- [Validate Epoch {self.epoch}] Stepping scheduler with loss: {avg_epoch_loss:.5f} ---")
        self.scheduler.step(avg_epoch_loss)
        print(f"--- [Validate Epoch {self.epoch}] Scheduler step complete ---")


        print(f"--- [Validate Epoch {self.epoch}] Final validation loss: {avg_epoch_loss:0.5f} \t Align loss: {avg_epoch_align_loss:0.5f} ---")
        print(f"--- [Validate Epoch {self.epoch}] Computing AUROC ---")
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation') # Assuming this method exists
        print(f"--- [Validate Epoch {self.epoch}] AUROC computed: {ret['auroc_mean']:.4f} ---")

        print(f"--- [Validate Epoch {self.epoch}] Saving predictions and ground truth to {self.args.save_dir} ---")
        np.save(f'{self.args.save_dir}/pred.npy', outPRED.data.cpu().numpy())
        np.save(f'{self.args.save_dir}/gt.npy', outGT.data.cpu().numpy())
        print(f"--- [Validate Epoch {self.epoch}] Files saved ---")

        self.epochs_stats['auroc val'].append(ret['auroc_mean'])
        self.epochs_stats['loss val'].append(avg_epoch_loss)
        self.epochs_stats['loss align val'].append(avg_epoch_align_loss)

        print(f"--- [Validate Epoch {self.epoch}] Finished Validation ---")
        return ret

    def compute_late_fusion(self, y_true, uniout_cxr, uniout_ehr):
        y_true = np.array(y_true)
        predictions_cxr = np.array(uniout_cxr)
        predictions_ehr = np.array(uniout_ehr)
        best_weights = np.ones(y_true.shape[-1])
        best_auroc = 0.0
        weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for class_idx in range(y_true.shape[-1]):
            for weight in weights:
                predictions = (predictions_ehr * best_weights) + (predictions_cxr * (1-best_weights))
                predictions[:, class_idx] = (predictions_ehr[:, class_idx] * weight) + (predictions_cxr[:, class_idx] * 1-weight)
                auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
                auroc_mean = np.mean(np.array(auc_scores))
                if auroc_mean > best_auroc:
                    best_auroc = auroc_mean
                    best_weights[class_idx] = weight
                # predictions = weight * predictions_cxr[]


        predictions = (predictions_ehr * best_weights) + (predictions_cxr * (1-best_weights))
        print(best_weights)

        auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
        ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                            average="micro")
        ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                            average="macro")
        ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                                average="weighted")
        
        # print(np.mean(np.array(auc_scores)

        # print()
        best_stats = {"auc_scores": auc_scores,
                "ave_auc_micro": ave_auc_micro,
                "ave_auc_macro": ave_auc_macro,
                "ave_auc_weighted": ave_auc_weighted,
                "auroc_mean": np.mean(np.array(auc_scores))
                }
        self.print_and_write(best_stats , isbest=True, prefix='late fusion weighted average')

        return best_stats 

    def eval_age(self):

        print('validating ... ')
           
        patiens = pd.read_csv('data/physionet.org/files/mimic-iv-1.0/core/patients.csv')
        subject_ids = np.array([int(item.split('_')[0]) for item in self.test_dl.dataset.ehr_files_paired])

        selected = patiens[patiens.subject_id.isin(subject_ids)]
        start = 18
        copy_ehr = np.copy(self.test_dl.dataset.ehr_files_paired)
        copy_cxr = np.copy(self.test_dl.dataset.cxr_files_paired)
        self.model.eval()
        step = 20
        for i in range(20, 100, step):
            subjects = selected.loc[((selected.anchor_age >= start) & (selected.anchor_age < i + step))].subject_id.values
            indexes = [jj for (jj, subject) in enumerate(subject_ids) if  subject in subjects]
            
            
            self.test_dl.dataset.ehr_files_paired = copy_ehr[indexes]
            self.test_dl.dataset.cxr_files_paired = copy_cxr[indexes]

            print(len(indexes))
            ret = self.validate(self.test_dl)
            print(f"{start}-{i + step} & {len(indexes)} & & & {ret['auroc_mean']:0.3f} & {ret['auprc_mean']:0.3f}")

            self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} val', filename=f'results_test_{start}_{i + step}.txt')

            # print(f"{start}-{i + step} & {len(indexes)} & & & {ret['auroc_mean']:0.3f} & {ret['auprc_mean']:0.3f}")
            # print(f"{start}-{i + 10} & {len(indexes)} & & & {ret['auroc_mean']:0.3f} & {ret['auprc_mean']:0.3f}")
            # self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} age_{start}_{i + 10}_{len(indexes)}', filename='results_test.txt')
            start = i + step


    def test(self):
        print("--- [Test] Starting Test Phase ---")
        self.epoch = "Test" # Indicate testing phase
        print("--- [Test] Setting model to eval mode ---")
        self.model.eval()
        print("--- [Test] Validating on Validation Set ---")
        ret_val = self.validate(self.val_dl)
        self.print_and_write(ret_val , isbest=True, prefix=f'{self.args.fusion_type} val', filename='results_val.txt') # Assuming print_and_write exists
        print("--- [Test] Setting model to eval mode again (just in case) ---")
        self.model.eval()
        print("--- [Test] Validating on Test Set ---")
        ret_test = self.validate(self.test_dl)
        self.print_and_write(ret_test , isbest=True, prefix=f'{self.args.fusion_type} test', filename='results_test.txt')
        print("--- [Test] Test Phase Finished ---")
        return

    def eval(self):
        # self.eval_age() # Uncomment if needed
        print("--- [Eval] Starting Evaluation Phase ---")
        self.epoch = "Eval" # Indicate evaluation phase
        print("--- [Eval] Setting model to eval mode ---")
        self.model.eval()
        # ret = self.validate(self.val_dl) # Uncomment if validation set eval is needed
        # self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} val', filename='results_val.txt')
        # self.model.eval() # Redundant if validate doesn't change mode
        print("--- [Eval] Evaluating on Test Set ---")
        ret = self.validate(self.test_dl)
        self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename='results_test.txt')
        print("--- [Eval] Evaluation Phase Finished ---")
        return

    def train(self):
        print(f"--- [Train Loop] Starting training process for fusion_type {self.args.fusion_type} ---")
        print(f"--- [Train Loop] Starting from epoch: {self.start_epoch}, Total epochs: {self.args.epochs} ---") # Assuming self.start_epoch exists from load_state
        for epoch_num in range(self.start_epoch, self.args.epochs):
            self.epoch = epoch_num # Set current epoch
            print(f"\n=============== Starting Epoch {self.epoch} =============== ")

            print(f"--- [Train Loop, Epoch {self.epoch}] Running Validation Phase ---")
            self.model.eval() # Set to eval mode for validation
            val_ret = self.validate(self.val_dl)
            print(f"--- [Train Loop, Epoch {self.epoch}] Validation Finished. AUROC: {val_ret['auroc_mean']:.4f} ---")


            print(f"--- [Train Loop, Epoch {self.epoch}] Saving last checkpoint ---")
            self.save_checkpoint(prefix='last') # Assuming this method exists in base Trainer

            if self.best_auroc < val_ret['auroc_mean']:
                print(f"--- [Train Loop, Epoch {self.epoch}] New best AUROC found: {val_ret['auroc_mean']:.4f} (previous best: {self.best_auroc:.4f}) ---")
                self.best_auroc = val_ret['auroc_mean']
                self.best_stats = val_ret
                print(f"--- [Train Loop, Epoch {self.epoch}] Saving best checkpoint ---")
                self.save_checkpoint() # Assuming this saves with a 'best' prefix by default
                self.print_and_write(val_ret, isbest=True) # Assuming this method exists
                self.patience = 0
                print(f"--- [Train Loop, Epoch {self.epoch}] Patience reset to 0 ---")
            else:
                print(f"--- [Train Loop, Epoch {self.epoch}] AUROC did not improve ({val_ret['auroc_mean']:.4f} vs best: {self.best_auroc:.4f}) ---")
                self.print_and_write(val_ret, isbest=False)
                self.patience += 1
                print(f"--- [Train Loop, Epoch {self.epoch}] Patience increased to {self.patience}/{self.args.patience} ---")

            if self.patience >= self.args.patience:
                print(f"--- [Train Loop, Epoch {self.epoch}] Early stopping triggered. Patience exceeded ({self.patience} >= {self.args.patience}) ---")
                break

            print(f"--- [Train Loop, Epoch {self.epoch}] Running Training Phase ---")
            self.model.train() # Set back to train mode
            train_ret = self.train_epoch()
            print(f"--- [Train Loop, Epoch {self.epoch}] Training Phase Finished ---")

            # Optional: Plot stats after each epoch if needed
            # print(f"--- [Train Loop, Epoch {self.epoch}] Plotting stats ---")
            # self.plot_stats(key='loss', filename='loss.pdf') # Assuming this method exists
            # self.plot_stats(key='auroc', filename='auroc.pdf') # Assuming this method exists

            print(f"=============== Finished Epoch {self.epoch} =============== ")


        print(f"--- [Train Loop] Training finished after {self.epoch + 1} epochs. ---")
        print(f"--- [Train Loop] Final Best Validation Stats ---")
        self.print_and_write(self.best_stats , isbest=True) # Log the best results at the end
        print("--- Training Complete ---")