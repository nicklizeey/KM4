import torch
from datasets.mathop import *
from datasets.dataloader import *
from PreBuildModel.TransformerDecodeOnly import *
from PreBuildModel.Transformer import *
from PreBuildModel.MLP import *
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf


registry = {}

def register(name):
    def decorator(f):
        registry[name] = f
        return f
    return decorator

def load_objs(config, *args):
    return registry[config['name']](config, *args)



#加载训练和验证数据集 (x + y) % p
@register('AddModTransformerData')
def load_AddModTransformerData(config, *args):
    figs = ModTestdata(config['p'])
    data = AddModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#加载训练和验证数据集 (x - y) % p 
@register('SubModTransformerData')
def load_SubModTransformerData(config, *args):
    figs = ModTestdata(config['p'])
    data = SubModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#加载训练和验证数据集 (x * y) % p
@register('MulModTransformerData')
def load_MulModTransformerData(config, *args):
    figs = ModTestdata(config['p'])
    data = MulModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#加载训练和验证数据集 (x / y) % p
@register('DivModTransformerData')
def load_DivModTransformerData(config, *args):
    figs = ModTestdata(config['p'])
    data = DivModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#加载训练和验证数据集 (x^2 + y^2) % p
@register('Pow2ModTransformerData')
def load_Pow2ModTransformerData(config, *args):
    figs = ModTestdata(config['p'])
    data = Pow2ModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#加载训练和验证数据集 (x^3 + y^3) % p
@register('Pow3ModTransformerData')
def load_Pow3ModTransformerData(config, *args):
    figs = ModTestdata(config['p'])
    data = Pow3ModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#加载训练和验证数据集 (x^3 + xy) % p
@register('Powx3xyModTransformerData')
def load_Powx3xyModTransformerData(config, *args):
    figs = ModTestdata(config['p'])
    data = Powx3xyModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#加载训练和验证数据集 (x^3 + xy^2 + y) % p
@register('Powx3xy2yModTransformerData')
def load_Powx3xy2yModTransformerData(config, *args):
    figs = ModTestdata(config['p'])
    data = Powx3xy2yModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#-----------------------------


#加载其他训练数据


#-----------------------------



#加载模型 transformer-decoder-only
@register('TransformerDecodeOnly')
def load_TransformerDecodeOnly(config, *args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = config['transformer_config']
    model = TransformerDecoderOnly(model_config['vocab_size'], model_config['embed_dim'], model_config['norm_shape'], model_config['ffn_num_hiddens'], model_config['num_heads'], model_config['num_layers'], model_config['dropout']).to(device)

    if config['checkpoint_path'] is not None:
        model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device), strict=config['strict_load'])
    return model


#加载模型 transformer
@register('Transformer')
def load_Transformer(config, *args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = config['transformer_config']
    model = Transformer(model_config['vocab_size'], model_config['embed_dim'], model_config['ffn_num_hiddens'], model_config['num_heads'], model_config['num_layers'], model_config['dropout'], device)

    if config['checkpoint_path'] is not None:
        model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device), strict=config['strict_load'])
    return model


#加载模型 transformer
@register('MLP')
def load_MLP(config, *args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = config['model_config']
    model = MLP(model_config['vocab_size'], model_config['embed_dim'], model_config['num_hiddens']).to(device)

    if config['checkpoint_path'] is not None:
        model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device), strict=config['strict_load'])
    return model

#---------------------------


#加载模型 CNN



#---------------------------


#---------------------------


#加载模型 RNN



#---------------------------




#加载损失函数 CrossEntropy
@register('CrossEntropy')
def load_CrossEntropy(config, *args):
    def get_loss(X, Y):
        loss = nn.CrossEntropyLoss()
        Y = Y.reshape(-1)
        X = X[:, -1, :]
        return loss(X, Y)
    return get_loss


#-----------------------


#加载不同模型需要使用的损失函数      目前已经有一种，用作Transformer Decoder Only  预计还需要四个


#-----------------------

#加载优化器 AdamW
@register('AdamW')
def load_AdamW(config, model, *args):
    optim =torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    return optim



#-----------------------


#加载不同模型需要使用的优化器      目前已经有一种，用作Transformer Decoder Only  预计还需要四个


#-----------------------





#@hydra.main(config_path="./config", config_name="train")
#def main(cfg : DictConfig):
#    cfg = OmegaConf.to_container(cfg)
#    model = load_objs(cfg['model'])

#if __name__ == '__main__':
#    main()

