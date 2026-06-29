from .attention_unet import AttentionResNetUNet
from .cnn_baseline import CNNBaseline
from .resnet_unet import ResNetUNet
from .unet import UNet
from .vgg_unet import VGGUNet

MODEL_BUILDERS = {
    "cnn": lambda pretrained: CNNBaseline(),
    "unet": lambda pretrained: UNet(),
    "vgg_unet": lambda pretrained: VGGUNet(pretrained=pretrained),
    "resnet_unet": lambda pretrained: ResNetUNet(pretrained=pretrained),
    "attention_unet": lambda pretrained: AttentionResNetUNet(pretrained=pretrained),
}

MODEL_NAMES = list(MODEL_BUILDERS.keys())


def get_model(name: str, pretrained: bool = True):
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")
    return MODEL_BUILDERS[name](pretrained)
