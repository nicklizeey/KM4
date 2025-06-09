from .GrokfastAlgorithm import grokfast 

from types import SimpleNamespace

from .PreBuildModel.MNIST_MLP import MNIST_MLP
from .PreBuildModel.TransformerDecodeOnly import TransformerDecoderOnly
from .PreBuildModel.Transformer import Transformer
from .PreBuildModel.MLP import MLP
from .PreBuildModel.RNN import RNN
from .PreBuildModel.CNN import CNN

prebuildmodel = SimpleNamespace(
    CNN=CNN,
    RNN=RNN,
    Transformer=Transformer,
    TransformerDecoderOnly=TransformerDecoderOnly,
    MLP=MLP,
    MNIST_MLP=MNIST_MLP
)
