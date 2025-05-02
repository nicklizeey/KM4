from datasets import *
from PreBuildModel import *
from GrokfastAlgorithm.grokfast import *
from GrokkingReproducer import *
from Visulization import *
from load_objs import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import datetime
import os




class Trainer:
    def __init__(self, config):
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     #创建设备

        self.train_data_loader, self.valid_data_loader = load_objs(config['datasets']) #创建数据迭代器以供使用,创建数据迭代器以供使用    

        self.model = load_objs(config['model']).to(self.device)                        #创建模型

        self.optimizer = load_objs(config['optimizer'], self.model)                    #创建优化器

        self.loss_fn = load_objs(config['loss'])                                       #创建损失函数

        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        self.log_dir = f"{config['train']['log_dir']}/{self.timestamp}"                #创建日志目录 

        self.writer = SummaryWriter(self.log_dir)                                      #创建日志记录器

        self.model_dir = os.path.join(get_original_cwd(), config['train']['checkpoint_path']) if self.config['train']['checkpoint_path'] != None else None #创建模型目录

        self.save_every = config['train']['save_every']                                 #创建保存间隔  save_every: 1000

        self.grads = None                                                              #创建grads 给grokfast算法使用

        self.using_grokfast = config['train']['using_grokfast']                        #创建grokfast算法开关

        self.max_epoch = int(config['train']['max_steps'])                             #创建训练轮数  max_steps: 1e6

        self.eval_every = config['train']['eval_every']                                #创建评估间隔  eval_every: 10

        self.stop_acc = config['train']['stop_condi']                                  #创建停止精度  stop_acc: 0.999

        self.after_reach_epoch = config['train']['stop_epochs']                        #创建达到验证精确度连续到达-轮数后停止  after_reach_epoch: 1000

        self.train_acc = 0.0                                                           #创建训练精度

        self.valid_acc = 0.0                                                           #创建验证精度

        self.train_loss = 0.0                                                          #创建训练损失

        self.valid_loss = 0.0                                                          #创建验证损失


    def T_D_train_epoch(self, epoch):
        self.model.train()
        for X, Y, T in self.train_data_loader:
            X, Y= X.to(self.device), Y.to(self.device)
            self.optimizer.zero_grad()
            Y_hat = self.model(X)
            loss = self.loss_fn(Y_hat, Y)
            loss.backward()
            # 检查是否使用以及使用什么grokfast算法
            if self.using_grokfast != 0:
                if self.config['grokfast']['name'] == 'grokfast_ma':
                    self.grads = gradfilter_ma(self.model, grads=self.grads, window_size=self.config['grokfast']['window_size'], lamb=self.config['grokfast']['lamb'], filter_type=self.config['grokfast']['filter_type'])
                elif self.config['grokfast']['name'] == 'grokfast_ema':
                    self.grads = gradfilter_ema(self.model, grads=self.grads, alpha=self.config['grokfast']['alpha'], lamb=self.config['grokfast']['lamb'])

            self.optimizer.step()

            self.train_loss = loss.item()           #记录训练损失
            self.writer.add_scalar('train_loss', self.train_loss, epoch)  #记录训练损失到日志

        self.model.eval()                                 #评估模型
        with torch.no_grad():
            train_correct = 0
            train_total = 0
            val_correct = 0
            val_total = 0
            for X, Y, T in self.valid_data_loader:
                X, Y = X.to(self.device), Y.to(self.device)
                Y_hat = self.model(X)
                preds = Y_hat[:, -1, :].argmax(dim=-1)
                labels = Y.reshape(-1)
                val_correct += (preds == labels).sum().item()
                val_total += Y.size(0)
            self.valid_acc = val_correct / val_total
            self.writer.add_scalar('valid_acc', self.valid_acc, epoch)  #记录验证精度到日志

            for X, Y, T in self.train_data_loader:
                X, Y = X.to(self.device), Y.to(self.device)
                Y_hat = self.model(X)
                preds = Y_hat[:, -1, :].argmax(dim=-1)
                labels = Y.reshape(-1)
                train_correct += (preds == labels).sum().item()
                train_total += Y.size(0)
            self.train_acc = train_correct / train_total
            self.writer.add_scalar('train_acc', self.train_acc, epoch)  #记录训练精度到日志
            self.writer.flush()

    def T_train_epoch(self, epoch):                                #实现 transformer的train_epoch函数
        pass

    def MLP_train_epoch(self, epoch):                              #实现 MLP的train_epoch函数
        pass

    def CNN_train_epoch(self, epoch):                              #实现 CNN的train_epoch函数
        pass

    def RNN_train_epoch(self, epoch):                              #实现 RNN的train_epoch函数
        pass




    def train(self):
        good_count = 0
        for epoch in range(self.max_epoch):

            #------------------------------------------------------------------------------------------------------------------------------------

            if self.config['model']['name'] == 'TransformerDecodeOnly':             #示例，已经实现transformer-decoder-only 的train_epoch函数
                self.T_D_train_epoch(epoch)
            elif self.config['model']['name'] == 'Transformer':                     #根据不同的模型选用不同的train_epoch函数
                pass
            elif self.config['model']['name'] == 'MLP':                             #根据不同的模型选用不同的train_epoch函数
                pass
            elif self.config['model']['name'] == 'CNN':                             #根据不同的模型选用不同的train_epoch函数
                pass
            elif self.config['model']['name'] == 'RNN':                             #根据不同的模型选用不同的train_epoch函数
                pass

            #------------------------------------------------------------------------------------------------------------------------------------


            if epoch % self.eval_every == 0:
                print(f'Epoch {epoch}, Train Loss: {self.train_loss}, Train Acc: {self.train_acc}, Valid Acc: {self.valid_acc}, have reach good acc {good_count} times.')

            if epoch > 0 and epoch % self.save_every == 0:
                if self.config['train']['checkpoint_path'] != None:
                    torch.save(self.model.state_dict(), self.model_dir)

            if self.valid_acc > self.stop_acc and good_count >= self.after_reach_epoch:
                print(f'Epoch {epoch}, Train Loss: {self.train_loss}, Train Acc: {self.train_acc}, Valid Acc: {self.valid_acc}, have reach good acc {good_count} times.')
                break
            elif self.valid_acc > self.stop_acc:
                good_count += 1
            else:
                good_count = 0



