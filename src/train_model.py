import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam,Adadelta,RMSprop,SGD
from torch.nn import BCELoss,BCEWithLogitsLoss,MSELoss
from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.nfm import NeuralFactorizationMachineModel 
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from utils.set_seed import setup_seed
from utils.summary_dat import cal_field_dims, make_feature
from utils.data_wrapper import Wrap_Dataset
from utils.early_stop import EarlyStopping
from utils.evaluate import cal_gauc, cal_group_metric

class Learner(object):
    
    def __init__(self, args):
        self.dat_name = args.dat_name
        self.model_name = args.model_name
        self.label_name = args.label_name

        self.group_num = args.group_num
        self.windows_size = args.windows_size
        self.alpha = args.alpha

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.patience = args.patience
        self.use_cuda = args.use_cuda
        self.epoch_num = args.epoch_num
        self.seed = args.randseed
        self.fout = args.fout


    def train(self):
        setup_seed(self.seed)
        self.all_dat, self.train_dat, self.vali_dat, self.test_dat = self._load_and_spilt_dat()
        self.train_loader, self.vali_loader, self.test_loader = self._wrap_dat()
        self.model, self.optim, self.early_stopping = self._init_train_env()
        self._train_iteration()
        self._test_and_save()


    def _load_and_spilt_dat(self):
        if self.dat_name == 'KuaiRand':
            all_dat = pd.read_json('../rec_datasets/Duration_KuaiRand/KuaiRand_subset.json')
            train_dat = all_dat[(all_dat['date']<=20220421) & (all_dat['date']>=20220408)]
            vali_dat = all_dat[(all_dat['date']<=20220428) & (all_dat['date']>=20220422)]
            test_dat = all_dat[(all_dat['date']<=20220508) & (all_dat['date']>=20220429)]
        elif self.dat_name == 'WeChat':
            all_dat = pd.read_json('../rec_datasets/Duration_WeChat/WeChat_subset.json')
            train_dat = all_dat[(all_dat['date']<=10) & (all_dat['date']>=1)]
            vali_dat = all_dat[(all_dat['date']<=12) & (all_dat['date']>=11)]
            test_dat = all_dat[(all_dat['date']<=14) & (all_dat['date']>=13)]
        elif self.dat_name == 'KuaiShou2018':
            all_dat = pd.read_json('../rec_datasets/Duration_KuaiShou2018/KuaiShou2018_subset.json')
            train_dat = all_dat[all_dat['date']==1]
            vali_dat = all_dat[all_dat['date']==2]
            test_dat = all_dat[all_dat['date']==3]

        return all_dat, train_dat, vali_dat, test_dat


    def _wrap_dat(self):
        input_train = Wrap_Dataset(make_feature(self.train_dat),
                                            self.train_dat[self.label_name].tolist())
        train_loader = DataLoader(input_train, 
                                        batch_size=self.batch_size, 
                                        shuffle=True)

        input_vali = Wrap_Dataset(make_feature(self.vali_dat),
                                            self.vali_dat[self.label_name].tolist())
        vali_loader = DataLoader(input_vali, 
                                        batch_size=2048, 
                                        shuffle=False)

        input_test = Wrap_Dataset(make_feature(self.test_dat),
                                            self.test_dat[self.label_name].tolist())
        test_loader = DataLoader(input_test, 
                                        batch_size=2048, 
                                        shuffle=False)
        return train_loader, vali_loader, test_loader

    
    def _init_train_env(self):
        if self.model_name == 'AFI':
            model = AutomaticFeatureInteractionModel(field_dims=cal_field_dims(self.all_dat), 
                                                     embed_dim=10, 
                                                     num_heads=8, 
                                                     num_layers=1,
                                                     atten_embed_dim=64,
                                                     mlp_dims=[64], 
                                                     dropouts=[0.2,0.2])
        elif self.model_name == 'NFM':
            model = NeuralFactorizationMachineModel(field_dims=cal_field_dims(self.all_dat), 
                                                    embed_dim=10, 
                                                    mlp_dims=[64,64,64], 
                                                    dropouts=[0.2,0.2])
        elif self.model_name == 'AFM':
            model = AttentionalFactorizationMachineModel(field_dims=cal_field_dims(self.all_dat), 
                                                         embed_dim=64, 
                                                         attn_size=128, 
                                                         dropouts=[0.2,0.2])
        elif self.model_name == 'DFM':
            model = DeepFactorizationMachineModel(field_dims=cal_field_dims(self.all_dat), embed_dim=10, mlp_dims=[64,64,64], dropout=0.2)
        elif self.model_name == 'FM':
            model = FactorizationMachineModel(field_dims=cal_field_dims(self.all_dat), embed_dim=10)

        if self.use_cuda:
            #model = nn.DataParallel(model)
            model = model.cuda()
        if self.model_name == 'FM' or self.model_name == 'DFM' or self.model_name == 'AFI':
            lr = 1e-3
            optim = Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.model_name == 'AFM' or self.model_name == 'NFM':
            lr = 1e-2
            optim = Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)

        # scheduler = ReduceLROnPlateau(optim, 
        #                               patience=10, 
        #                               mode='min',
        #                               threshold=1e-6,
        #                               verbose=True)

        early_stopping = EarlyStopping(self.fout + '_temp', patience=self.patience, verbose=True)

        print(model)

        return model, optim, early_stopping 


    def _train_iteration(self):
        dur=[]
        for epoch in range(self.epoch_num):
            if epoch >= 0:
                t0 = time.time()
            loss_log = []
            self.model.train()
            for _id, batch in enumerate(self.train_loader):
                self.model.train()
                self.optim.zero_grad()
                BCELossfunc = BCELoss()
                # BCELossfunc = MSELoss()
                output_score =  self.model(batch[0]).view(batch[0].size(0))
                target = batch[1]
                train_loss = BCELossfunc(output_score, target)
                train_loss.backward()
                self.optim.step()
                loss_log.append(train_loss.item())

            gauc_vali = cal_gauc(self.vali_dat, self.model, self.vali_loader)
            self.early_stopping(gauc_vali*(-1), self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break 

            if epoch >= 0:
                dur.append(time.time() - t0)

            print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Vali_GAUC {:.4f} | "
                    "Test_Recall {:.4f}| Test_AUC {:.4f}| Test_Log_Loss {:.4f}|". format(epoch, np.mean(dur), np.mean(loss_log),gauc_vali,
                                                    0,0,0))


    def _test_and_save(self):
        if self.model_name == 'AFI':
            model = AutomaticFeatureInteractionModel(field_dims=cal_field_dims(self.all_dat), 
                                                     embed_dim=10, 
                                                     num_heads=8, 
                                                     num_layers=1,
                                                     atten_embed_dim=64,
                                                     mlp_dims=[64], 
                                                     dropouts=[0.2,0.2])
        elif self.model_name == 'NFM':
            model = NeuralFactorizationMachineModel(field_dims=cal_field_dims(self.all_dat), 
                                                    embed_dim=10, 
                                                    mlp_dims=[64,64,64], 
                                                    dropouts=[0.2,0.2])
        elif self.model_name == 'AFM':
            model = AttentionalFactorizationMachineModel(field_dims=cal_field_dims(self.all_dat), 
                                                         embed_dim=64, 
                                                         attn_size=128, 
                                                         dropouts=[0.2,0.2])
        elif self.model_name == 'DFM':
            model = DeepFactorizationMachineModel(field_dims=cal_field_dims(self.all_dat), embed_dim=10, mlp_dims=[64,64,64], dropout=0.2)
        elif self.model_name == 'FM':
            model = FactorizationMachineModel(field_dims=cal_field_dims(self.all_dat), embed_dim=10)

        model = model.cuda()

        model.load_state_dict(torch.load(self.fout + '_temp_checkpoint.pt'))

        ndcg_ls, gauc_val = cal_group_metric(self.test_dat ,model,[1,3,5], self.test_loader)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("{}_{} | Log_loss {:.4f} | AUC {:.4f} | GAUC {:.4f} | Recall {:.4f} | "
                    "nDCG@1 {:.4f}| nDCG@3 {:.4f}| nDCG@5 {:.4f}|". format(self.model_name, self.label_name, 0,0, gauc_val, 0,
                                                    ndcg_ls[0],ndcg_ls[1],ndcg_ls[2]))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        df_result = pd.DataFrame([],columns=['GAUC','nDCG@1','nDCG@3','nDCG@5'])
        df_result.loc[1] =  [gauc_val] + ndcg_ls


        df_result.to_json('{}_result.json'.format(self.fout), indent=4)
        torch.save(model.state_dict(), '{}_model.pt'.format(self.fout))


        
if __name__=="__main__":
    pass

        