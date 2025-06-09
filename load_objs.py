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
import os
from hydra.utils import get_original_cwd


registry = {}

def register(name):   #For each registered name, register a dictionary of the corresponding function.
    def decorator(f):
        registry[name] = f
        return f
    return decorator

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



#Load training and validation dataset (x + y) % p
@register('AddModTransformerData')
def load_AddModTransformerData(config, *args):
    p = config['p']
    frac_train = config['frac_train']
    batch_size = config['batch_size']

    print("\n" + "="*60)
    print("You are now using the AddModTransformerData dataset")
    print(f"Modulo (p):              {p}")
    print(f"Train / Valid Split:    {frac_train*100:.1f}% / {(1-frac_train)*100:.1f}%")
    print(f"Batch Size:             {batch_size}")
    print("="*60 + "\n")
    figs = ModTestdata(config['p'])
    data = AddModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#Load training and validation dataset (x - y) % p 
@register('SubModTransformerData')
def load_SubModTransformerData(config, *args):
    p = config['p']
    frac_train = config['frac_train']
    batch_size = config['batch_size']

    print("\n" + "="*60)
    print("You are now using the SubModTransformerData dataset")
    print(f"Modulo (p):              {p}")
    print(f"Train / Valid Split:    {frac_train*100:.1f}% / {(1-frac_train)*100:.1f}%")
    print(f"Batch Size:             {batch_size}")
    print("="*60 + "\n")
    figs = ModTestdata(config['p'])
    data = SubModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#Load training and validation dataset (x * y) % p
@register('MulModTransformerData')
def load_MulModTransformerData(config, *args):
    p = config['p']
    frac_train = config['frac_train']
    batch_size = config['batch_size']

    print("\n" + "="*60)
    print("You are now using the MulModTransformerData dataset")
    print(f"Modulo (p):              {p}")
    print(f"Train / Valid Split:    {frac_train*100:.1f}% / {(1-frac_train)*100:.1f}%")
    print(f"Batch Size:             {batch_size}")
    print("="*60 + "\n")
    figs = ModTestdata(config['p'])
    data = MulModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#Load training and validation datasets (x / y) % p
@register('DivModTransformerData')
def load_DivModTransformerData(config, *args):
    p = config['p']
    frac_train = config['frac_train']
    batch_size = config['batch_size']

    print("\n" + "="*60)
    print("You are now using the DivModTransformerData dataset")
    print(f"Modulo (p):              {p}")
    print(f"Train / Valid Split:    {frac_train*100:.1f}% / {(1-frac_train)*100:.1f}%")
    print(f"Batch Size:             {batch_size}")
    print("="*60 + "\n")
    figs = ModTestdata(config['p'])
    data = DivModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#Load training and validation dataset (x^2 + y^2) % p
@register('Pow2ModTransformerData')
def load_Pow2ModTransformerData(config, *args):
    p = config['p']
    frac_train = config['frac_train']
    batch_size = config['batch_size']

    print("\n" + "="*60)
    print("You are now using the Pow2ModTransformerData dataset")
    print(f"Modulo (p):              {p}")
    print(f"Train / Valid Split:    {frac_train*100:.1f}% / {(1-frac_train)*100:.1f}%")
    print(f"Batch Size:             {batch_size}")
    print("="*60 + "\n")
    figs = ModTestdata(config['p'])
    data = Pow2ModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#Load training and validation datasets (x^3 + y^3) % p
@register('Pow3ModTransformerData')
def load_Pow3ModTransformerData(config, *args):
    p = config['p']
    frac_train = config['frac_train']
    batch_size = config['batch_size']

    print("\n" + "="*60)
    print("You are now using the Pow3ModTransformerData dataset")
    print(f"Modulo (p):              {p}")
    print(f"Train / Valid Split:    {frac_train*100:.1f}% / {(1-frac_train)*100:.1f}%")
    print(f"Batch Size:             {batch_size}")
    print("="*60 + "\n")
    figs = ModTestdata(config['p'])
    data = Pow3ModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#Load training and validation datasets (x^3 + xy) % p
@register('Powx3xyModTransformerData')
def load_Powx3xyModTransformerData(config, *args):
    p = config['p']
    frac_train = config['frac_train']
    batch_size = config['batch_size']

    print("\n" + "="*60)
    print("You are now using the Powx3xyModTransformerData dataset")
    print(f"Modulo (p):              {p}")
    print(f"Train / Valid Split:    {frac_train*100:.1f}% / {(1-frac_train)*100:.1f}%")
    print(f"Batch Size:             {batch_size}")
    print("="*60 + "\n")
    figs = ModTestdata(config['p'])
    data = Powx3xyModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter


#Load training and validation datasets (x^3 + xy^2 + y) % p
@register('Powx3xy2yModTransformerData')
def load_Powx3xy2yModTransformerData(config, *args):
    p = config['p']
    frac_train = config['frac_train']
    batch_size = config['batch_size']

    print("\n" + "="*60)
    print("You are now using the Powx3xy2yModTransformerData dataset")
    print(f"Modulo (p):              {p}")
    print(f"Train / Valid Split:    {frac_train*100:.1f}% / {(1-frac_train)*100:.1f}%")
    print(f"Batch Size:             {batch_size}")
    print("="*60 + "\n")
    figs = ModTestdata(config['p'])
    data = Powx3xy2yModTransformerData(figs.x_list(), figs.y_list(), config['frac_train'])
    train_dataset = MathOPDataset(data.train)
    valid_dataset = MathOPDataset(data.valid)
    train_data_iter = data_loader(train_dataset, config['batch_size'])
    valid_data_iter = data_loader(valid_dataset, config['batch_size'])
    return train_data_iter, valid_data_iter

@register('MNIST')
def load_MNIST_data(config, *args):
    print("\n" + "="*60)
    print("You are now using the MNIST dataset")
    print(f"Train / Valid Split:    {config.get('train_fraction', 0.8) * 100:.1f}% / {(1 - config.get('train_fraction', 0.8)) * 100:.1f}%")
    print(f"Batch Size:             {config.get('batch_size', 64)}")
    print("="*60 + "\n")
    print(f"Loading MNIST dataset with config: {config}") # Add print for debugging
    train_loader, valid_loader, _ = create_mnist_dataloaders( # Ignore test_loader for now
        batch_size=config.get('batch_size', 64), # Use .get for defaults
        data_path=config.get('data_path', './data'),
        train_fraction=config.get('train_fraction', 0.8),
        num_workers=config.get('num_workers', 0)
    )
    return train_loader, valid_loader

@register("AddModImageCNN")
def load_AddModImageCNN(config, *args):
    p = config['p']
    frac_train = config.get('frac_train', 0.1)
    batch_size = config.get('batch_size', 32)

    # 📢 User Alerts
    print("\n" + "="*60)
    print("🔢 You are now using the Modular Addition CNN Dataset")
    print(f"🧮 Modulo (p):              {p}")
    print(f"📊 Train / Valid Split:    {frac_train*100:.1f}% / {(1-frac_train)*100:.1f}%")
    print(f"📦 Batch Size:             {batch_size}")
    print("="*60 + "\n")

    dataset = AddModCNNData(p, frac_train)
    train_dataset = MathOPImageDataset(dataset.train)
    valid_dataset = MathOPImageDataset(dataset.valid)

    train_loader = data_loader(train_dataset, batch_size)
    valid_loader = data_loader(valid_dataset, batch_size)
    return train_loader, valid_loader


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



#Loading Models transformer-decoder-only
@register('TransformerDecodeOnly')
def load_TransformerDecodeOnly(config, *args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = config['transformer_config']

    print("\n" + "="*60)
    print("🧠 You are now initializing the Transformer Decoder-Only model")
    print(f"🔤 Vocab Size:              {model_config['vocab_size']}")
    print(f"📐 Embedding Dimension:     {model_config['embed_dim']}")
    print(f"📏 Norm Shape:              {model_config['norm_shape']}")
    print(f"⚙️  FFN Hidden Size:         {model_config['ffn_num_hiddens']}")
    print(f"🧠 Attention Heads:         {model_config['num_heads']}")
    print(f"📚 Decoder Layers:          {model_config['num_layers']}")
    print(f"💧 Dropout Rate:            {model_config['dropout']}")
    print(f"🖥️ Device:                   {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    model = TransformerDecoderOnly(
        model_config['vocab_size'], model_config['embed_dim'], model_config['norm_shape'],
        model_config['ffn_num_hiddens'], model_config['num_heads'], model_config['num_layers'],
        model_config['dropout']
    ).to(device)

    if config['checkpoint_path'] is not None:
        print(f"📥 Loading checkpoint from:  {config['checkpoint_path']}")
        model.load_state_dict(torch.load(os.path.join(get_original_cwd(), config['checkpoint_path']), map_location=device), strict=config['strict_load'])
        print("✅ Checkpoint loaded successfully.")
    else:
        print("🚫 No checkpoint specified. Model will be randomly initialized.")
    print("="*60 + "\n")

    return model


#Loading Models transformer
@register('Transformer')
def load_Transformer(config, *args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = config['transformer_config']

    print("\n" + "="*60)
    print("🧠 You are now initializing the Full Transformer model")
    print(f"🔤 Vocab Size:              {model_config['vocab_size']}")
    print(f"📐 Embedding Dimension:     {model_config['embed_dim']}")
    print(f"⚙️  FFN Hidden Size:         {model_config['ffn_num_hiddens']}")
    print(f"🧠 Attention Heads:         {model_config['num_heads']}")
    print(f"📚 Transformer Layers:      {model_config['num_layers']}")
    print(f"💧 Dropout Rate:            {model_config['dropout']}")
    print(f"🖥️ Device:                   {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    model = Transformer(
        model_config['vocab_size'], model_config['embed_dim'],
        model_config['ffn_num_hiddens'], model_config['num_heads'],
        model_config['num_layers'], model_config['dropout'], device
    )

    if config['checkpoint_path'] is not None:
        print(f"📥 Loading checkpoint from:  {config['checkpoint_path']}")
        model.load_state_dict(torch.load(os.path.join(get_original_cwd(), config['checkpoint_path']), map_location=device), strict=config['strict_load'])
        print("✅ Checkpoint loaded successfully.")
    else:
        print("🚫 No checkpoint specified. Model will be randomly initialized.")
    print("="*60 + "\n")

    return model


#Loading Models MLP
@register("MLP")
def load_MLP(config, *args):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    mc = config["model_config"]
    model = MLP(
        p             = mc["p"],
        embedding_dim = mc["embedding_dim"],
        hidden_sizes  = mc["hidden_sizes"]
    ).to(device)

    if config["checkpoint_path"] is not None:
        model.load_state_dict(
            torch.load(os.path.join(get_original_cwd(), config['checkpoint_path']), map_location=device),
            strict=config["strict_load"]
        )
    return model


@register('MNIST')
def load_MNIST_data(config, *args):
    print("\n" + "="*60)
    print("You are now using the MNIST dataset")
    print(f"Train / Valid Split:    {config.get('train_fraction', 0.8) * 100:.1f}% / {(1 - config.get('train_fraction', 0.8)) * 100:.1f}%")
    print(f"Batch Size:             {config.get('batch_size', 64)}")
    print("="*60 + "\n")
    # ... (implementation as in your file) ...
    print(f"Loading MNIST dataset with config: {config}") # Add print for debugging
    train_loader, valid_loader, _ = create_mnist_dataloaders( # Ignore test_loader for now
        batch_size=config.get('batch_size', 64), # Use .get for defaults
        data_path=config.get('data_path', './data'),
        train_fraction=config.get('train_fraction', 0.8),
        num_workers=config.get('num_workers', 0)
    )
    return train_loader, valid_loader

#Loading Models MNIST_MLP
@register('MNIST_MLP')
def load_MNIST_MLP(config, *args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'model_config' not in config:
        raise ValueError(f"Missing 'model_config' in configuration for MNIST_MLP: {config}")

    model_config = config['model_config']

    print("\n" + "="*60)
    print("🧠 You are now initializing the MNIST MLP model")
    print(f"🖼️ Input Size:              {model_config['input_size']}")
    print(f"🧮 Hidden Sizes:            {model_config['hidden_sizes']}")
    print(f"🎯 Output Size:             {model_config['output_size']}")
    print(f"🖥️ Device:                   {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    model = MNIST_MLP(
        input_size=model_config['input_size'],
        hidden_sizes=model_config['hidden_sizes'],
        output_size=model_config['output_size']
    ).to(device)

    if config.get('checkpoint_path') is not None:
        print(f"📥 Loading checkpoint from:  {config['checkpoint_path']}")
        model.load_state_dict(torch.load(os.path.join(get_original_cwd(), config['checkpoint_path']), map_location=device), strict=config.get('strict_load', True))
        print("✅ Checkpoint loaded successfully.")
    else:
        print("🚫 No checkpoint specified. Model will be randomly initialized.")
    print("="*60 + "\n")

    return model

#Loading Models CNN
@register('CNN')
def load_CNN(config, *args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = config['CNNmodel_config']

    # 📢 Printing Model Configuration Information
    print("\n" + "="*60)
    print("🧠 You are now initializing the CNN model")
    print(f"🧩 Embedding Dimension:     {model_config['emb_dim']}")
    print(f"🔢 Modulo (p):              {model_config['p']}")
    print(f"🎯 Number of Classes:       {model_config['num_classes']}")
    print(f"🧱 Conv Channels (c1/c2):   {model_config['c1']} / {model_config['c2']}")
    print(f"🏗️ Hidden Dimension:         {model_config['hidden_dim']}")
    print(f"💧 Dropout Rate:            {model_config['dropout_rate']}")
    print(f"🖥️ Device:                   {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    if config.get('checkpoint_path'):
        print(f"📥 Loading checkpoint from:  {config['checkpoint_path']}")
        model = CNN(
            emb_dim=model_config['emb_dim'],
            p=model_config['p'],
            num_classes=model_config['num_classes'],
            c1=model_config['c1'],
            c2=model_config['c2'],
            hidden_dim=model_config['hidden_dim'],
            dropout_rate=model_config['dropout_rate']
        ).to(device)

        model.load_state_dict(
            torch.load(os.path.join(get_original_cwd(), config['checkpoint_path']), map_location=device),
            strict=config.get('strict', True)
        )
        print("✅ Checkpoint loaded successfully.")
    else:
        model = CNN(
            emb_dim=model_config['emb_dim'],
            p=model_config['p'],
            num_classes=model_config['num_classes'],
            c1=model_config['c1'],
            c2=model_config['c2'],
            hidden_dim=model_config['hidden_dim'],
            dropout_rate=model_config['dropout_rate']
        ).to(device)
        print("🚫 No checkpoint specified. Model will be randomly initialized.")

    print("="*60 + "\n")

    return model

#Loading Models RNN
@register('RNN')
def load_RNN(config, *args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = config['model_config']

    print("\n" + "="*60)
    print("🧠 You are now initializing the RNN model")
    print(f"🔢 Modulo (p):              {model_config['p']}")
    print(f"📐 Embedding Dimension:     {model_config['embedding_dim']}")
    print(f"🧮 Hidden Size:             {model_config['hidden_size']}")
    print(f"📚 RNN Layers:              {model_config['num_layers']}")
    print(f"🖥️ Device:                   {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    model = RNN(
        model_config['p'],
        model_config['embedding_dim'],
        model_config['hidden_size'],
        model_config['num_layers']
    ).to(device)

    if config['checkpoint_path'] is not None:
        print(f"📥 Loading checkpoint from:  {config['checkpoint_path']}")
        model.load_state_dict(torch.load(os.path.join(get_original_cwd(), config['checkpoint_path']), map_location=device), strict=config['strict_load'])
        print("✅ Checkpoint loaded successfully.")
    else:
        print("🚫 No checkpoint specified. Model will be randomly initialized.")
    print("="*60 + "\n")

    return model

#---------------------------




#---------------------------
#Load the loss functions that need to be used for different models

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

@register('ImageCrossEntropy')
def load_ImageCrossEntropy(config, *args):
    # This function correctly ignores the config argument
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn


#-----------------------


#Load Optimiser
@register('AdamW')
def load_AdamW(config, model, *args):
    optim =torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    return optim

@register("AdamCNN")
def get_optimizer(config, model, *args):
    return torch.optim.Adam(
        model.parameters(),
        lr=config.get('lr', 1e-5),
        weight_decay=config.get('weight_decay', 1e-4)
    )

@register("Adam")
def get_optimizer(config, model):
    return torch.optim.Adam(
        model.parameters(),
        lr=config.get('lr', 1e-5),
        weight_decay=config.get('weight_decay', 1e-4)
    )

@register('SGD')
def load_SGD(config, model, *args):
    return torch.optim.SGD(
        model.parameters(), 
        lr=config.get('lr', 0.01),
        momentum=config.get('momentum', 0.9)
    )
#-----------------------






#@register('MNIST')
#def load_MNIST_data(config, *args):
#   # ... (implementation as in your file) ...
 #   print(f"Loading MNIST dataset with config: {config}") # Add print for debugging
 #   train_loader, valid_loader, _ = create_mnist_dataloaders( # Ignore test_loader for now
  #      batch_size=config.get('batch_size', 64), # Use .get for defaults
 #       data_path=config.get('data_path', './data'),
#        train_fraction=config.get('train_fraction', 0.8),
##        num_workers=config.get('num_workers', 0)
 #   )
 #   return train_loader, valid_loader


#@register('MNIST_MLP')
#def load_MNIST_MLP(config, *args):
 #   # ... (implementation as in your file) ...
  #  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Also check for MPS (Apple Silicon) if needed
    # if torch.backends.mps.is_available(): device = torch.device('mps')

    # Ensure model_config is accessed correctly within the provided config dict
  #  if 'model_config' not in config:
  #      raise ValueError(f"Missing 'model_config' in configuration for MNIST_MLP: {config}")

  #  model_config = config['model_config']
    #model = MNIST_MLP(
 #       input_size=model_config['input_size'],
#        hidden_sizes=model_config['hidden_sizes'],
 #       output_size=model_config['output_size']
  #  ).to(device)

 #   if config.get('checkpoint_path') is not None: # Use .get safely
   #     model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device), strict=config.get('strict_load', True))
  #  return model