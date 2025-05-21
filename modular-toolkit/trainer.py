from datasets import *
from PreBuildModel import *
from GrokfastAlgorithm.grokfast import *
from load_objs import *
import torch
from torch.utils.tensorboard import SummaryWriter
from hydra.utils import get_original_cwd
import datetime
import os




class Trainer:
    def __init__(self, config): #以下到self.valid_loss都不要动
        self.config = config

        if torch.backends.mps.is_available():                                           #创建设备
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.train_data_loader, self.valid_data_loader = load_objs(config['datasets']) #创建数据迭代器以供使用,创建数据迭代器以供使用    

        self.model = load_objs(config['model']).to(self.device)                        #用YAML文件创建模型

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

        self.loss_recoder = []
        self.acc_recoder = []


    def T_D_train_epoch(self, epoch):
        self.model.train()
        for X, Y, T in self.train_data_loader:
            X, Y= X.to(self.device), Y.to(self.device)
            self.optimizer.zero_grad()
            Y_hat = self.model(X)
            loss = self.loss_fn(Y_hat, Y)
            loss.backward() #计算梯度
            # 检查是否使用以及使用什么grokfast算法
            if self.using_grokfast != 0:
                if self.config['grokfast']['name'] == 'grokfast_ma':
                    self.grads = gradfilter_ma(self.model, grads=self.grads, window_size=self.config['grokfast']['window_size'], lamb=self.config['grokfast']['lamb'], filter_type=self.config['grokfast']['filter_type'])
                elif self.config['grokfast']['name'] == 'grokfast_ema':
                    self.grads = gradfilter_ema(self.model, grads=self.grads, alpha=self.config['grokfast']['alpha'], lamb=self.config['grokfast']['lamb'])

            self.optimizer.step()

            self.train_loss = loss.item()           #记录训练损失
            self.writer.add_scalar('Loss/Train', self.train_loss, epoch)  #记录训练损失到日志

#---------------------------------------------------------------------------------------------------------------
            self.loss_recoder.append(self.train_loss)  #记录训练损失到列表
            if len(self.loss_recoder) > 3:
                self.loss_recoder.pop(0)  #如果列表长度大于3，则删除第一个元素

#---------------------------------------------------------------------------------------------------------------

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
            self.writer.add_scalar('Accuracy/Validation', self.valid_acc, epoch)  #记录验证精度到日志

#--------------------------------------------------------------------------------------------------------------- 为自适应grokfast算法添加验证准确率参考数据
            self.acc_recoder.append(self.valid_acc)  #记录验证精度到列表
            if len(self.acc_recoder) > 3:
                self.acc_recoder.pop(0)  #如果列表长度大于3，则删除第一个元素
            if self.config['grokfast']['self_adoptive'] == True:    
                self_adoptive(self.config, self.loss_recoder, self.acc_recoder)    #如果使用自适应算法，则调用自适应算法
#---------------------------------------------------------------------------------------------------------------


            for X, Y, T in self.train_data_loader:
                X, Y = X.to(self.device), Y.to(self.device)
                Y_hat = self.model(X)
                preds = Y_hat[:, -1, :].argmax(dim=-1)
                labels = Y.reshape(-1)
                train_correct += (preds == labels).sum().item()
                train_total += Y.size(0)
            self.train_acc = train_correct / train_total
            self.writer.add_scalar('Accuracy/Train', self.train_acc, epoch)  #记录训练精度到日志
            self.writer.flush()

    def T_train_epoch(self, epoch):                                #实现 transformer的train_epoch函数
        self.model.train() # Set model to training mode
        running_loss = 0.0
        num_batches = 0

        for X, Y, Tgt in self.train_data_loader: # Tgt is needed for decoder input
            # Move data to the configured device
            src = X.to(self.device)
            labels = Y.to(self.device)
            tgt_input = Tgt.to(self.device) # Use Tgt as decoder input

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            # The Transformer model takes src and tgt
            Y_hat = self.model(src, tgt_input)

            # Calculate loss
            # Assuming the loss function expects the last token's prediction vs the label Y
            loss = self.loss_fn(Y_hat, labels) # Use the loaded loss function

            # Backward pass
            loss.backward()

            # Optional: Apply Grokfast gradient modification
            if self.using_grokfast:
                if self.config['grokfast']['name'] == 'grokfast_ma':
                    self.grads = gradfilter_ma(self.model, grads=self.grads, window_size=self.config['grokfast']['window_size'], lamb=self.config['grokfast']['lamb'], filter_type=self.config['grokfast']['filter_type'])
                elif self.config['grokfast']['name'] == 'grokfast_ema':
                    self.grads = gradfilter_ema(self.model, grads=self.grads, alpha=self.config['grokfast']['alpha'], lamb=self.config['grokfast']['lamb'])

            # Optimize
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        # Log average training loss for the epoch
        self.train_loss = running_loss / num_batches if num_batches > 0 else 0.0
        self.writer.add_scalar('Loss/Train', self.train_loss, epoch)

        # Evaluation phase (similar to T_D_train_epoch)
        self.model.eval() # Set model to evaluation mode
        train_correct = 0
        train_total = 0
        val_correct = 0
        val_total = 0
        running_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            # Validation loop
            for X, Y, Tgt in self.valid_data_loader:
                src = X.to(self.device)
                labels = Y.to(self.device)
                tgt_input = Tgt.to(self.device)

                Y_hat = self.model(src, tgt_input)
                val_loss = self.loss_fn(Y_hat, labels) # Calculate validation loss
                running_val_loss += val_loss.item()
                num_val_batches += 1

                # Accuracy calculation (assuming loss_fn aligns with taking last token prediction)
                preds = Y_hat[:, -1, :].argmax(dim=-1)
                labels_flat = labels.reshape(-1)
                val_correct += (preds == labels_flat).sum().item()
                val_total += labels.size(0)

            # Calculate and log validation metrics
            self.valid_acc = val_correct / val_total if val_total > 0 else 0.0
            self.valid_loss = running_val_loss / num_val_batches if num_val_batches > 0 else 0.0
            self.writer.add_scalar('Accuracy/Validation', self.valid_acc, epoch)
            #self.writer.add_scalar('Loss/validation', self.valid_loss, epoch)


            # Training accuracy loop (optional but good practice)
            for X, Y, Tgt in self.train_data_loader:
                src = X.to(self.device)
                labels = Y.to(self.device)
                tgt_input = Tgt.to(self.device)

                Y_hat = self.model(src, tgt_input)

                # Accuracy calculation
                preds = Y_hat[:, -1, :].argmax(dim=-1)
                labels_flat = labels.reshape(-1)
                train_correct += (preds == labels_flat).sum().item()
                train_total += labels.size(0)

            # Calculate and log training accuracy
            self.train_acc = train_correct / train_total if train_total > 0 else 0.0
            self.writer.add_scalar('Accuracy/Train', self.train_acc, epoch)

        self.writer.flush() # Ensure logs are written

    def MLP_train_epoch(self, epoch):                              #实现 MLP的train_epoch函数
        self.model.train()
        train_correct = 0
        train_total = 0


        for X, Y, T in self.train_data_loader:
            X, Y = X.to(self.device), Y.to(self.device)
            self.optimizer.zero_grad()
            Y_hat = self.model(X)
            loss = self.loss_fn(Y_hat, Y)
            loss.backward()

            # 添加 grokfast 支持
            if self.using_grokfast != 0:
                if self.config['grokfast']['name'] == 'grokfast_ma':
                    self.grads = gradfilter_ma(self.model, grads=self.grads, window_size=self.config['grokfast']['window_size'], lamb=self.config['grokfast']['lamb'], filter_type=self.config['grokfast']['filter_type'])
                elif self.config['grokfast']['name'] == 'grokfast_ema':
                    self.grads = gradfilter_ema(self.model, grads=self.grads, alpha=self.config['grokfast']['alpha'], lamb=self.config['grokfast']['lamb'])

            self.optimizer.step()

            preds = Y_hat.argmax(dim=-1)
            train_correct += (preds == Y.reshape(-1)).sum().item()
            train_total += Y.size(0)

            self.train_loss = loss.item()
            self.writer.add_scalar('train_loss', self.train_loss, epoch)

        self.train_acc = train_correct / train_total
        self.writer.add_scalar('train_acc', self.train_acc, epoch)

        # 评估阶段
        self.model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, Y, T in self.valid_data_loader:
                X, Y = X.to(self.device), Y.to(self.device)
                Y_hat = self.model(X)
                preds = Y_hat.argmax(dim=-1)
                val_correct += (preds == Y.reshape(-1)).sum().item()
                val_total += Y.size(0)

        self.valid_acc = val_correct / val_total
        self.writer.add_scalar('valid_acc', self.valid_acc, epoch)

    def CNN_train_epoch(self, epoch):                              #实现 CNN的train_epoch函数
        # --- 训练 ---
        self.model.train()
        for X, Y, *rest in self.train_data_loader:
            # X: (batch, C, H, W), Y: (batch,) 或 (batch,1)
            X, Y = X.to(self.device), Y.to(self.device).view(-1)
            self.optimizer.zero_grad()
            logits = self.model(X)                     # → (batch, num_classes)
            loss   = self.loss_fn(logits, Y)           # e.g. CrossEntropy
            loss.backward()

            # 如果开启了 GrokFast，还可以在此处插入 grads 过滤：
            if self.using_grokfast:
                cfg = self.config['grokfast']
                if cfg['name'] == 'grokfast_ma':
                    self.grads = gradfilter_ma(self.model, grads=self.grads,
                                               window_size=cfg['window_size'],
                                               lamb=cfg['lamb'],
                                               filter_type=cfg['filter_type'])
                elif cfg['name'] == 'grokfast_ema':
                    self.grads = gradfilter_ema(self.model, grads=self.grads,
                                                alpha=cfg['alpha'],
                                                lamb=cfg['lamb'])
            self.optimizer.step()

            # 记录训练损失
            self.train_loss = loss.item()
            self.writer.add_scalar('train/loss', self.train_loss, epoch)

        # --- 验证 ---
        self.model.eval()
        with torch.no_grad():
            # 计算验证集准确率
            val_correct = 0
            val_total   = 0
            for X, Y, *rest in self.valid_data_loader:
                X, Y = X.to(self.device), Y.to(self.device).view(-1)
                logits = self.model(X)
                preds  = logits.argmax(dim=1)
                val_correct += (preds == Y).sum().item()
                val_total   += Y.size(0)
            self.valid_acc = val_correct / val_total
            self.writer.add_scalar('valid/acc', self.valid_acc, epoch)

            # 计算训练集准确率（完整遍历）
            train_correct = 0
            train_total   = 0
            for X, Y, *rest in self.train_data_loader:
                X, Y = X.to(self.device), Y.to(self.device).view(-1)
                logits = self.model(X)
                preds  = logits.argmax(dim=1)
                train_correct += (preds == Y).sum().item()
                train_total   += Y.size(0)
            self.train_acc = train_correct / train_total
            self.writer.add_scalar('train/acc', self.train_acc, epoch)

        # 每次评估后 flush 日志
        self.writer.flush()

    def RNN_train_epoch(self, epoch):                              # 实现 RNN的train_epoch函数
        # 训练
        self.model.train()
        running_loss = 0.0
        num_batches  = 0

        for batch in self.train_data_loader:         
            X, Y = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X)                  
            labels = Y[:, -1]                       
            loss   = self.loss_fn(logits, labels)
            loss.backward()

            # 可选 · Grokfast
            if self.config['train']['using_grokfast']:
                cfg = self.config['grokfast']
                if cfg['name'] == 'grokfast_ma':
                    self.grads = gradfilter_ma(self.model, self.grads,
                                                cfg['window_size'], cfg['lamb'],
                                                cfg['filter_type'])
                elif cfg['name'] == 'grokfast_ema':
                    self.grads = gradfilter_ema(self.model, self.grads,
                                                 cfg['alpha'], cfg['lamb'])
                for p, g in zip(self.model.parameters(), self.grads):
                    p.grad = g
            self.optimizer.step()

            running_loss += loss.item()
            num_batches  += 1

        self.train_loss = running_loss / num_batches
        self.writer.add_scalar('Loss/Train', self.train_loss, epoch)

        # 训练集准确率
        train_correct = train_total = 0
        with torch.no_grad():
            for batch in self.train_data_loader:
                X, Y = batch[0].to(self.device), batch[1].to(self.device)
                preds = self.model(X).argmax(dim=-1)
                labels = Y[:, -1]
                train_correct += (preds == labels).sum().item()
                train_total   += labels.size(0)
        self.train_acc = train_correct / train_total
        self.writer.add_scalar('Accuracy/Train', self.train_acc, epoch)

        # 验证
        self.model.eval()
        val_loss = 0.0
        val_correct = val_total = 0
        with torch.no_grad():
            for batch in self.valid_data_loader:
                X, Y = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(X)
                labels = Y[:, -1]

                val_loss += self.loss_fn(logits, labels).item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        self.valid_loss = val_loss / len(self.valid_data_loader)
        self.valid_acc  = val_correct / val_total
        self.writer.add_scalar('Loss/Validation',     self.valid_loss, epoch)
        self.writer.add_scalar('Accuracy/Validation', self.valid_acc,  epoch)
        self.writer.flush()

    def Image_train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        correct_train = 0
        total_train = 0

        for images, labels in self.train_data_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass - Model should handle image input (e.g., flatten internally if MLP)
            outputs = self.model(images) 

            # Loss calculation - Use the standard CrossEntropyLoss instance
            loss = self.loss_fn(outputs, labels) 

            loss.backward()

            # Optional Grokfast application (might need adjustment for image gradients)
            if self.using_grokfast:
                if self.config['grokfast']['name'] == 'grokfast_ma':
                    self.grads = gradfilter_ma(self.model, grads=self.grads, window_size=self.config['grokfast']['window_size'], lamb=self.config['grokfast']['lamb'], filter_type=self.config['grokfast']['filter_type'])
                elif self.config['grokfast']['name'] == 'grokfast_ema':
                    self.grads = gradfilter_ema(self.model, grads=self.grads, alpha=self.config['grokfast']['alpha'], lamb=self.config['grokfast']['lamb'])

            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            # Training accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        self.train_loss = running_loss / num_batches if num_batches > 0 else 0.0
        self.train_acc = correct_train / total_train if total_train > 0 else 0.0
        self.writer.add_scalar('Loss/Train', self.train_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', self.train_acc, epoch)

        # --- Validation ---
        self.model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        num_val_batches = 0
        with torch.no_grad():
            for images, labels in self.valid_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()
                num_val_batches += 1

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        self.valid_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
        self.valid_acc = correct_val / total_val if total_val > 0 else 0.0
        #self.writer.add_scalar('Loss/validation', self.valid_loss, epoch)
        self.writer.add_scalar('Accuracy/Validation', self.valid_acc, epoch)
        self.writer.flush() # Ensure logs are written
    



    def train(self):
        good_count = 0
        print(f"Starting training with model: {self.config['model']['name']}") # Added print statement
        print(f"Using dataset: {self.config['datasets']['name']}") # Add dataset name
        for epoch in range(self.max_epoch):

            #------------------------------------------------------------------------------------------------------------------------------------

            if self.config['model']['name'] == 'TransformerDecodeOnly':             #示例，已经实现transformer-decoder-only 的train_epoch函数
                self.T_D_train_epoch(epoch)
            elif self.config['model']['name'] == 'Transformer':                     #根据不同的模型选用不同的train_epoch函数
                self.T_train_epoch(epoch)
            elif self.config['model']['name'] == 'MLP':                             #根据不同的模型选用不同的train_epoch函数
                self.MLP_train_epoch(epoch)
            elif self.config['model']['name'] == 'CNN':                             #根据不同的模型选用不同的train_epoch函数
                self.CNN_train_epoch(epoch)
            elif self.config['model']['name'] == 'RNN':                             #根据不同的模型选用不同的train_epoch函数
                self.RNN_train_epoch(epoch)
            elif self.config['model']['name'] == 'MNIST_MLP': 
                self.Image_train_epoch(epoch)                                        # <--- CALL IMAGE TRAIN EPOCH
            else:
                print(f"Error: Unknown model name '{self.config['model']['name']}' in config.")
                break # Stop if model name is not recognized

            #------------------------------------------------------------------------------------------------------------------------------------


            if epoch % self.eval_every == 0:
                print(f'Epoch {epoch}, Train Loss: {self.train_loss:.4f}, Train Acc: {self.train_acc:.4f}, Valid Acc: {self.valid_acc:.4f}, have reach good acc {good_count} times.')
                print(f"The alpha value is {self.config['grokfast']['alpha']}, lamb value is {self.config['grokfast']['lamb']}")
            if epoch > 0 and epoch % self.save_every == 0:
                if self.config['train']['checkpoint_path'] != None:
                    torch.save(self.model.state_dict(), self.model_dir)

            if self.valid_acc > self.stop_acc and good_count >= self.after_reach_epoch:
                print(f'Stop condition met at Epoch {epoch}. Train Loss: {self.train_loss:.4f}, Train Acc: {self.train_acc:.4f}, Valid Acc: {self.valid_acc:.4f}, have reach good acc {good_count} times.')
                break
            elif self.valid_acc > self.stop_acc:
                good_count += 1
            else:
                good_count = 0
        
        print("Training finished.") # Add confirmation
        self.writer.close() # Close writer at the end



