from clip.clip import BICUBIC, _convert_image_to_rgb
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def export_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor()
    ])
