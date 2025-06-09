# File: KM4/modular_toolkit/trainer.py
from datasets import *
from PreBuildModel import *
from GrokfastAlgorithm.grokfast import *
from load_objs import *
import torch
from torch.utils.tensorboard import SummaryWriter
import os # Make sure os is imported


class Trainer:
    # MODIFIED: Added hydra_run_path to the constructor
    def __init__(self, config, hydra_run_path):
        self.config = config
        self.hydra_run_path = hydra_run_path # Store the Hydra run path

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.train_data_loader, self.valid_data_loader = load_objs(config['datasets'])
        self.model = load_objs(config['model']).to(self.device)
        self.optimizer = load_objs(config['optimizer'], self.model)
        self.loss_fn = load_objs(config['loss'])

        # MODIFIED: Construct log_dir for SummaryWriter using the passed hydra_run_path
        # config['train']['log_dir'] should still be a relative path like './logs' or 'logs'
        relative_log_subdir = config['train']['log_dir']
        self.log_dir = os.path.join(self.hydra_run_path, relative_log_subdir)

        # Ensure the specific log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir) # SummaryWriter now uses the correctly constructed absolute path

        # This print statement will now show the correct path inside the Hydra run directory
        print(f"INFO: TensorBoard event files for this run will be written to: {self.log_dir}")

        # MODIFIED: Adjust model_dir (checkpoint saving) to also be inside hydra_run_path
        if config['train']['checkpoint_path'] is not None and config['train']['checkpoint_path'].strip() != "":
            # Assuming config['train']['checkpoint_path'] is a relative subdirectory name e.g., 'checkpoints'
            relative_checkpoint_subdir = config['train']['checkpoint_path']
            self.model_dir = os.path.join(self.hydra_run_path, relative_checkpoint_subdir)
            if not os.path.exists(self.model_dir): # Ensure model_dir exists before trying to save to it
                os.makedirs(self.model_dir, exist_ok=True)
            print(f"INFO: Model checkpoints will be saved to: {self.model_dir}")
        else:
            self.model_dir = None # Or set to a default within hydra_run_path if you always want to save them
            print("INFO: Model checkpoints will not be saved (checkpoint_path is None or empty in config).")


        self.save_every = config['train']['save_every']
        self.grads = None
        self.using_grokfast = config['train']['using_grokfast']
        self.max_epoch = int(config['train']['max_steps'])
        self.eval_every = config['train']['eval_every']
        self.stop_acc = config['train']['stop_condi']
        self.after_reach_epoch = config['train']['stop_epochs']
        self.train_acc = 0.0
        self.valid_acc = 0.0
        self.train_loss = 0.0
        self.valid_loss = 0.0
        self.total_param_norm = 0.0 # Initialize parameter norm
        self.loss_recoder = []
        self.acc_recoder = []

    def _log_parameter_norms(self, epoch):
        """Calculates and logs the total L2 norm of model parameters."""
        total_param_norm = 0.0
        for p in self.model.parameters():
            if p.requires_grad: # Consider only trainable parameters
                param_norm = p.data.norm(2)
                total_param_norm += param_norm.item() ** 2
        total_param_norm = total_param_norm ** 0.5
        self.total_param_norm = total_param_norm # Store for terminal output
        self.writer.add_scalar('Norms/Total_Parameter_L2_Norm', self.total_param_norm, epoch)

    def T_D_train_epoch(self, epoch):
        self.model.train()
        for X, Y, T in self.train_data_loader:
            X, Y= X.to(self.device), Y.to(self.device)
            self.optimizer.zero_grad()
            Y_hat = self.model(X)
            loss = self.loss_fn(Y_hat, Y)
            loss.backward()
            if self.using_grokfast != 0:
                if self.config['grokfast']['name'] == 'grokfast_ma':
                    self.grads = gradfilter_ma(self.model, grads=self.grads, window_size=self.config['grokfast']['window_size'], lamb=self.config['grokfast']['lamb'], filter_type=self.config['grokfast']['filter_type'])
                elif self.config['grokfast']['name'] == 'grokfast_ema':
                    self.grads = gradfilter_ema(self.model, grads=self.grads, alpha=self.config['grokfast']['alpha'], lamb=self.config['grokfast']['lamb'])
            self.optimizer.step()
            self.train_loss = loss.item()
        # Log training loss per epoch (average or last batch) - current code logs last batch loss
        self.writer.add_scalar('Loss/Train', self.train_loss, epoch)
        self.loss_recoder.append(self.train_loss)
        if len(self.loss_recoder) > 3:
            self.loss_recoder.pop(0)

        self.model.eval()
        with torch.no_grad():
            train_correct = 0
            train_total = 0
            val_correct = 0
            val_total = 0
            running_val_loss = 0.0
            num_val_batches = 0
            for X, Y, T in self.valid_data_loader:
                X, Y = X.to(self.device), Y.to(self.device)
                Y_hat = self.model(X)
                
                loss = self.loss_fn(Y_hat, Y)
                running_val_loss += loss.item()
                num_val_batches += 1

                preds = Y_hat[:, -1, :].argmax(dim=-1)
                labels = Y.reshape(-1)
                val_correct += (preds == labels).sum().item()
                val_total += Y.size(0)

            self.valid_loss = running_val_loss / num_val_batches if num_val_batches > 0 else 0.0
            self.writer.add_scalar('Loss/Validation', self.valid_loss, epoch)

            self.valid_acc = val_correct / val_total if val_total > 0 else 0.0 # Avoid division by zero
            self.writer.add_scalar('Accuracy/Validation', self.valid_acc, epoch)
            self.acc_recoder.append(self.valid_acc)
            if len(self.acc_recoder) > 3:
                self.acc_recoder.pop(0)
            if self.config['grokfast'].get('self_adoptive', False) == True: # Safely get self_adoptive
                self_adoptive(self.config, self.loss_recoder, self.acc_recoder)
            for X, Y, T in self.train_data_loader: # Re-iterate for train accuracy on current model state
                X, Y = X.to(self.device), Y.to(self.device)
                Y_hat = self.model(X)
                preds = Y_hat[:, -1, :].argmax(dim=-1)
                labels = Y.reshape(-1)
                train_correct += (preds == labels).sum().item()
                train_total += Y.size(0)
            self.train_acc = train_correct / train_total if train_total > 0 else 0.0 # Avoid division by zero
            self.writer.add_scalar('Accuracy/Train', self.train_acc, epoch)
            
            # Log parameter norms
            self._log_parameter_norms(epoch)

        self.writer.flush()


    def T_train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        for X, Y, Tgt in self.train_data_loader:
            src = X.to(self.device)
            labels = Y.to(self.device)

            # Create a shifted version of the labels for the decoder input
            tgt_input = torch.cat([torch.zeros((labels.size(0), 1), dtype=torch.long, device=self.device), labels[:, :-1]], dim=1)


            self.optimizer.zero_grad()
            Y_hat = self.model(src, tgt_input)
            
            loss = self.loss_fn(Y_hat.view(-1, Y_hat.size(-1)), labels.view(-1))
            loss.backward()
            if self.using_grokfast:
                if self.config['grokfast']['name'] == 'grokfast_ma':
                    self.grads = gradfilter_ma(self.model, grads=self.grads, window_size=self.config['grokfast']['window_size'], lamb=self.config['grokfast']['lamb'], filter_type=self.config['grokfast']['filter_type'])
                elif self.config['grokfast']['name'] == 'grokfast_ema':
                    self.grads = gradfilter_ema(self.model, grads=self.grads, alpha=self.config['grokfast']['alpha'], lamb=self.config['grokfast']['lamb'])
            self.optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        self.train_loss = running_loss / num_batches if num_batches > 0 else 0.0
        self.writer.add_scalar('Loss/Train', self.train_loss, epoch)

        self.model.eval()
        train_correct = 0
        train_total = 0
        val_correct = 0
        val_total = 0
        running_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for X, Y, Tgt in self.valid_data_loader:
                src = X.to(self.device)
                labels = Y.to(self.device)
                
                # Autoregressive decoding for validation
                batch_size = src.size(0)
                # Initialize with start-of-sequence token (e.g., 0)
                tgt_input = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
                max_len = labels.size(1)
                
                for _ in range(max_len):
                    Y_hat = self.model(src, tgt_input)
                    # Get the last token prediction
                    next_token_logits = Y_hat[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                    # Append the predicted token to the input for the next step
                    tgt_input = torch.cat([tgt_input, next_token], dim=1)

                # now tgt_input contains the generated sequence, of shape (batch_size, max_len + 1)
                # remove the start token for comparison
                generated_sequence = tgt_input[:, 1:] 

                val_loss = self.loss_fn(Y_hat.view(-1, Y_hat.size(-1)), labels.view(-1))
                running_val_loss += val_loss.item()
                num_val_batches += 1
                
                labels_flat = labels.view(-1)
                val_correct += (generated_sequence.view(-1) == labels_flat).sum().item()
                val_total += labels.numel()

            self.valid_acc = val_correct / val_total if val_total > 0 else 0.0
            self.valid_loss = running_val_loss / num_val_batches if num_val_batches > 0 else 0.0
            self.writer.add_scalar('Accuracy/Validation', self.valid_acc, epoch)
            self.writer.add_scalar('Loss/Validation', self.valid_loss, epoch)

            # Train accuracy
            with torch.no_grad():
                for X, Y, Tgt in self.train_data_loader:
                    src = X.to(self.device)
                    labels = Y.to(self.device)
                    tgt_input = torch.cat([torch.zeros((labels.size(0), 1), dtype=torch.long, device=self.device), labels[:, :-1]], dim=1)

                    Y_hat = self.model(src, tgt_input)
                    preds = Y_hat.argmax(dim=-1)
                    labels_flat = labels.view(-1)
                    train_correct += (preds.view(-1) == labels_flat).sum().item()
                    train_total += labels.numel()
            self.train_acc = train_correct / train_total if train_total > 0 else 0.0
            self.writer.add_scalar('Accuracy/Train', self.train_acc, epoch)

            # Log parameter norms
            self._log_parameter_norms(epoch)

        self.writer.flush()

    def MLP_train_epoch(self, epoch):
        running_loss  = 0.0
        num_batches   = 0
        correct_train_epoch = 0 # Use different name for epoch accumulation
        total_train_epoch   = 0 # Use different name for epoch accumulation
        self.model.train()
        for batch in self.train_data_loader:
            if isinstance(batch, (list, tuple)): X, Y = batch[0], batch[1]
            elif isinstance(batch, dict): X, Y = batch["inputs"], batch["labels"]
            else: raise TypeError(f"Unexpected batch type: {type(batch)}")
            X, Y = X.to(self.device), Y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss   = self.loss_fn(logits, Y)
            loss.backward()
            if self.using_grokfast:
                gconf = self.config["grokfast"]
                if gconf["name"] == "grokfast_ma": self.grads = gradfilter_ma(self.model, grads=self.grads, window_size=gconf["window_size"], lamb=gconf["lamb"], filter_type=gconf["filter_type"])
                elif gconf["name"] == "grokfast_ema": self.grads = gradfilter_ema(self.model, grads=self.grads, alpha=gconf["alpha"], lamb=gconf["lamb"])
            self.optimizer.step()
            running_loss += loss.item()
            num_batches  += 1
            
        self.train_loss = running_loss / max(num_batches, 1)
        # Calculate training accuracy after the epoch
        self.model.eval() # Switch to eval mode for accuracy calculation
        with torch.no_grad():
            for batch in self.train_data_loader:
                if isinstance(batch, (list, tuple)): X, Y = batch[0], batch[1]
                elif isinstance(batch, dict): X, Y = batch["inputs"], batch["labels"]
                X, Y = X.to(self.device), Y.to(self.device)
                logits = self.model(X)
                if logits.dim() > 2: preds = logits.view(-1, logits.size(-1)).argmax(dim=1)
                else: preds  = logits.argmax(dim=1)
                Y_flat = Y.view(-1)
                correct_train_epoch += (preds == Y_flat).sum().item()
                total_train_epoch   += Y_flat.numel()
        self.train_acc  = correct_train_epoch / max(total_train_epoch, 1)
        self.writer.add_scalar("Loss/Train", self.train_loss, epoch)
        self.writer.add_scalar("Accuracy/Train", self.train_acc,  epoch)
        
        # Validation
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        num_val_batches = 0
        with torch.no_grad():
            for batch in self.valid_data_loader:
                if isinstance(batch, (list, tuple)): X_val, Y_val = batch[0], batch[1]
                elif isinstance(batch, dict): X_val, Y_val = batch["inputs"], batch["labels"]
                else: raise TypeError(f"Unexpected batch type: {type(batch)}")
                X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)
                logits_val = self.model(X_val)
                loss_val   = self.loss_fn(logits_val, Y_val)
                val_loss  += loss_val.item()
                num_val_batches +=1
                if logits_val.dim() > 2: preds_val = logits_val.view(-1, logits_val.size(-1)).argmax(dim=1)
                else: preds_val = logits_val.argmax(dim=1)
                Y_val_flat = Y_val.view(-1)
                correct_val += (preds_val == Y_val_flat).sum().item()
                total_val += Y_val_flat.numel()
        self.valid_loss = val_loss / max(num_val_batches, 1)
        self.valid_acc  = correct_val / max(total_val, 1)
        self.writer.add_scalar("Loss/Validation", self.valid_loss, epoch)
        self.writer.add_scalar("Accuracy/Validation", self.valid_acc,  epoch)
        # Log parameter norms
        self._log_parameter_norms(epoch)
        self.writer.flush()


    def CNN_train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        for X, Y, *rest in self.train_data_loader:
            X, Y = X.to(self.device), Y.to(self.device).view(-1)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.loss_fn(logits, Y)
            loss.backward()
            if self.using_grokfast:
                cfg = self.config['grokfast']
                if cfg['name'] == 'grokfast_ma': self.grads = gradfilter_ma(self.model, grads=self.grads, window_size=cfg['window_size'], lamb=cfg['lamb'], filter_type=cfg['filter_type'])
                elif cfg['name'] == 'grokfast_ema': self.grads = gradfilter_ema(self.model, grads=self.grads, alpha=cfg['alpha'], lamb=cfg['lamb'])
            self.optimizer.step()
            running_loss += loss.item()
            num_batches +=1
        self.train_loss = running_loss / num_batches if num_batches > 0 else 0.0
        self.writer.add_scalar('Loss/Train', self.train_loss, epoch)

        self.model.eval()
        with torch.no_grad():
            val_correct = 0
            val_total   = 0
            running_val_loss = 0.0
            num_val_batches = 0
            for X, Y, *rest in self.valid_data_loader:
                X, Y = X.to(self.device), Y.to(self.device).view(-1)
                logits = self.model(X)
                loss_v = self.loss_fn(logits, Y)
                running_val_loss += loss_v.item()
                num_val_batches += 1
                preds  = logits.argmax(dim=1)
                val_correct += (preds == Y).sum().item()
                val_total   += Y.size(0)
            self.valid_acc = val_correct / val_total if val_total > 0 else 0.0
            self.valid_loss = running_val_loss / num_val_batches if num_val_batches > 0 else 0.0
            self.writer.add_scalar('Accuracy/Validation', self.valid_acc, epoch)
            self.writer.add_scalar('Loss/Validation', self.valid_loss, epoch)

            train_correct = 0
            train_total   = 0
            for X, Y, *rest in self.train_data_loader: # Re-iterate for train accuracy
                X, Y = X.to(self.device), Y.to(self.device).view(-1)
                logits = self.model(X)
                preds  = logits.argmax(dim=1)
                train_correct += (preds == Y).sum().item()
                train_total   += Y.size(0)
            self.train_acc = train_correct / train_total if train_total > 0 else 0.0
            self.writer.add_scalar('Accuracy/Train', self.train_acc, epoch)
            
            # Log parameter norms
            self._log_parameter_norms(epoch)
            
        self.writer.flush()

    def RNN_train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        num_batches  = 0
        for batch in self.train_data_loader:
            X, Y = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X)
            labels = Y[:, -1] # Assuming labels are for the last step in sequence
            loss   = self.loss_fn(logits, labels)
            loss.backward()
            if self.using_grokfast:
                cfg = self.config['grokfast']
                if cfg['name'] == 'grokfast_ma': self.grads = gradfilter_ma(self.model, grads=self.grads, window_size=cfg['window_size'], lamb=cfg['lamb'], filter_type=cfg['filter_type'])
                elif cfg['name'] == 'grokfast_ema': self.grads = gradfilter_ema(self.model, grads=self.grads, alpha=cfg['alpha'], lamb=cfg['lamb'])
            self.optimizer.step()
            running_loss += loss.item()
            num_batches  += 1
        self.train_loss = running_loss / num_batches if num_batches > 0 else 0.0
        self.writer.add_scalar('Loss/Train', self.train_loss, epoch)
        
        self.model.eval()
        train_correct = train_total = 0
        with torch.no_grad():
            for batch in self.train_data_loader: # Re-iterate for train accuracy
                X, Y = batch[0].to(self.device), batch[1].to(self.device)
                preds = self.model(X).argmax(dim=-1)
                labels = Y[:, -1]
                train_correct += (preds == labels).sum().item()
                train_total   += labels.size(0)
        self.train_acc = train_correct / train_total if train_total > 0 else 0.0
        self.writer.add_scalar('Accuracy/Train', self.train_acc, epoch)
        
        val_loss = 0.0
        val_correct = val_total = 0
        num_val_batches = 0
        with torch.no_grad():
            for batch in self.valid_data_loader:
                X, Y = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(X)
                labels = Y[:, -1]
                val_loss += self.loss_fn(logits, labels).item()
                num_val_batches +=1
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
        self.valid_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
        self.valid_acc  = val_correct / val_total if val_total > 0 else 0.0
        self.writer.add_scalar('Loss/Valid', self.valid_loss, epoch)
        self.writer.add_scalar('Accuracy/Valid', self.valid_acc,  epoch)

        # Log parameter norms
        self._log_parameter_norms(epoch)
        
        self.writer.flush()

    def Image_train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        correct_train_epoch = 0 
        total_train_epoch = 0   
        for images, labels in self.train_data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            if self.using_grokfast:
                if self.config['grokfast']['name'] == 'grokfast_ma': self.grads = gradfilter_ma(self.model, grads=self.grads, window_size=self.config['grokfast']['window_size'], lamb=self.config['grokfast']['lamb'], filter_type=self.config['grokfast']['filter_type'])
                elif self.config['grokfast']['name'] == 'grokfast_ema': self.grads = gradfilter_ema(self.model, grads=self.grads, alpha=self.config['grokfast']['alpha'], lamb=self.config['grokfast']['lamb'])
            self.optimizer.step()
            running_loss += loss.item()
            num_batches += 1
            _, predicted = torch.max(outputs.data, 1)
            total_train_epoch += labels.size(0)
            correct_train_epoch += (predicted == labels).sum().item()

        self.train_loss = running_loss / num_batches if num_batches > 0 else 0.0
        self.train_acc = correct_train_epoch / total_train_epoch if total_train_epoch > 0 else 0.0 
        self.writer.add_scalar('Loss/Train', self.train_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', self.train_acc, epoch)
        
        self.model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        num_val_batches = 0
        with torch.no_grad():
            for images, labels in self.valid_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss_val_item = self.loss_fn(outputs, labels) 
                val_loss += loss_val_item.item()
                num_val_batches += 1
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        self.valid_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
        self.valid_acc = correct_val / total_val if total_val > 0 else 0.0
        self.writer.add_scalar('Loss/Validation', self.valid_loss, epoch)
        self.writer.add_scalar('Accuracy/Validation', self.valid_acc, epoch)

        # Log parameter norms
        self._log_parameter_norms(epoch)
        
        self.writer.flush()


    def train(self): # This is the main training loop in run.py, not used directly from here
        # This method is effectively superseded by the loop in run.py
        # good_count logic and print statements are handled in run.py
        # Checkpoint saving and stop condition logic are also in run.py
        
        # Minimal placeholder if it were ever called, though it shouldn't be in current setup
        print(f"Starting training with model: {self.config['model']['name']} from Trainer.train() (should be unused)")
        # ... (loop would go here but is in run.py) ...
        print("Training finished from Trainer.train() (should be unused).")
        if self.writer:
            self.writer.close()