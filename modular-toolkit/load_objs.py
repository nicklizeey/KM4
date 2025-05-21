from PreBuildModel.MNIST_MLP import MNIST_MLP # Or MNIST_CNN etc.
from datasets.mnist_data import create_mnist_dataloaders
import torch
from datasets.mathop import *
from datasets.dataloader import *
from PreBuildModel.TransformerDecodeOnly import *
from PreBuildModel.Transformer import *
from PreBuildModel.MLP import *
from PreBuildModel.RNN import *
from PreBuildModel.CNN import *


registry = {}

def register(name):
    def decorator(f):
        registry[name] = f
        return f
    return decorator  #装饰器

def load_objs(config_or_name, *args):
    """
    Loads an object (dataset, model, loss, optimizer) based on the provided
    configuration dictionary or name string.
    """
    if isinstance(config_or_name, dict):
        # If a dictionary is provided, extract the name
        name = config_or_name.get('name')
        if name and name in registry:
            # Call the registered function with the original config dictionary
            return registry[name](config_or_name, *args)
        else:
            # Handle error: name not found in dict or registry not found
            if not name:
                 raise ValueError(f"Configuration dictionary missing 'name' key: {config_or_name}")
            else:
                 raise ValueError(f"Object name '{name}' from config dictionary not found in registry.")

    elif isinstance(config_or_name, str):
        # If just the name string is provided (e.g., from command-line override)
        name = config_or_name
        if name in registry:
            # Call the registered function.
            # Pass an empty dictionary as the config argument, assuming the specific
            # loader function (like load_ImageCrossEntropy) doesn't strictly need it.
            # Modify this if other loaders require the actual config dict even when called by name.
            return registry[name]({}, *args)
        else:
            # Handle error: name not found in registry
            raise ValueError(f"Object name '{name}' (as string) not found in registry.")
    else:
        # Handle unexpected input type
        raise TypeError(f"load_objs expects a dictionary or string, but received {type(config_or_name)}")



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


@register("AddModImageCNN")
def load_AddModImageCNN(config, *args):
    p = config['p']
    frac_train = config.get('frac_train', 0.1)
    batch_size = config.get('batch_size', 32)

    dataset = AddModCNNData(p, frac_train)
    train_dataset = MathOPImageDataset(dataset.train)
    valid_dataset = MathOPImageDataset(dataset.valid)

    train_loader = data_loader(train_dataset, batch_size)
    valid_loader = data_loader(valid_dataset, batch_size)
    return train_loader, valid_loader


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


#加载模型 MLP
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
@register('CNN')
def load_CNN(config, *args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = config['CNNmodel_config']

    model = CNN(
        emb_dim=model_config['emb_dim'],
        p=model_config['p'],
        num_classes=model_config['num_classes'],
        c1=model_config['c1'],
        c2=model_config['c2'],
        hidden_dim=model_config['hidden_dim'],
        dropout_rate=model_config['dropout_rate']
    ).to(device)

    if config.get('checkpoint_path'):
        model.load_state_dict(
            torch.load(config['checkpoint_path'], map_location=device),
            strict=config.get('strict', True)
        )

    return model




#---------------------------


#---------------------------


#加载模型 RNN
@register('RNN')
def load_RNN(config, *args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = config['model_config']
    model = RNN(model_config['p'], model_config['embedding_dim'], model_config['hidden_size'], model_config['num_layers']).to(device)

    if config['checkpoint_path'] is not None:
        model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device), strict=config['strict_load'])
    return model

#---------------------------




#加载损失函数 CrossEntropy
'''
@register('CrossEntropy')
def load_CrossEntropy(config, *args):
    def get_loss(X, Y):
        loss = nn.CrossEntropyLoss()
        Y = Y.reshape(-1)
        X = X[:, -1, :]
        return loss(X, Y)
    return get_loss
'''


@register('CrossEntropy')
def load_CrossEntropy(config, *args):
    # ... (implementation as in your file) ...
    def get_loss(X, Y):
        loss = nn.CrossEntropyLoss()
        Y = Y.reshape(-1)
        # Check if X needs slicing (depends on model output shape)
        if X.dim() > 2: # <<<<< CORRECTED LINE
            X = X[:, -1, :] # Assume prediction is based on the last token
        elif X.dim() == 2: # e.g., MLP output (batch, classes)
            pass # No slicing needed
        else:
             print(f"Warning: Unexpected output shape {X.shape} for CrossEntropyLoss.")
        return loss(X, Y)
    return get_loss


@register("CrossEntropyCNN")
def get_loss_fn(config, *args):
    return torch.nn.CrossEntropyLoss()


@register("Adam")
def get_optimizer(config, model):
    return torch.optim.Adam(
        model.parameters(),
        lr=config.get('lr', 1e-5),
        weight_decay=config.get('weight_decay', 1e-4)
    )

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

#加载数据集 MNIST
@register('ImageCrossEntropy')
def load_ImageCrossEntropy(config, *args):
    # This function correctly ignores the config argument
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn

@register('MNIST')
def load_MNIST_data(config, *args):
    # ... (implementation as in your file) ...
    print(f"Loading MNIST dataset with config: {config}") # Add print for debugging
    train_loader, valid_loader, _ = create_mnist_dataloaders( # Ignore test_loader for now
        batch_size=config.get('batch_size', 64), # Use .get for defaults
        data_path=config.get('data_path', './data'),
        train_fraction=config.get('train_fraction', 0.8),
        num_workers=config.get('num_workers', 0)
    )
    return train_loader, valid_loader


@register('MNIST_MLP')
def load_MNIST_MLP(config, *args):
    # ... (implementation as in your file) ...
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Also check for MPS (Apple Silicon) if needed
    # if torch.backends.mps.is_available(): device = torch.device('mps')

    # Ensure model_config is accessed correctly within the provided config dict
    if 'model_config' not in config:
        raise ValueError(f"Missing 'model_config' in configuration for MNIST_MLP: {config}")

    model_config = config['model_config']
    model = MNIST_MLP(
        input_size=model_config['input_size'],
        hidden_sizes=model_config['hidden_sizes'],
        output_size=model_config['output_size']
    ).to(device)

    if config.get('checkpoint_path') is not None: # Use .get safely
        model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device), strict=config.get('strict_load', True))
    return model


#@hydra.main(config_path="./config", config_name="train")
#def main(cfg : DictConfig):
#    cfg = OmegaConf.to_container(cfg)
#    model = load_objs(cfg['model'])

#if __name__ == '__main__':
#    main()

@register('SGD')
def load_SGD(config, model, *args):
    return torch.optim.SGD(
        model.parameters(), 
        lr=config.get('lr', 0.01),
        momentum=config.get('momentum', 0.9)
    )
