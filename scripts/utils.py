import os
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
_to_tensor = transforms.ToTensor()
_to_pil    = transforms.ToPILImage()

def load_image(path, size=None, crop=False, keep_aspect=True):
    img = Image.open(path).convert('RGB')
    if size:
        if isinstance(size, int):
            if keep_aspect:
                w, h = img.size
                scale = size / min(w, h)
                img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            else:
                img = img.resize((size, size), Image.LANCZOS)
            if crop:
                w, h = img.size
                left, top = (w-size)//2, (h-size)//2
                img = img.crop((left, top, left+size, top+size))
        else:
            img = img.resize(size, Image.LANCZOS)
    return img

def tensor_to_image(t, denormalize=False):
    t = t.detach().cpu().clone()
    if t.dim() == 4:
        t = t.squeeze(0)
    if denormalize:
        mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
        std  = torch.tensor(IMAGENET_STD).view(3,1,1)
        t = t * std + mean
    t = t.clamp(0,1)
    return _to_pil(t)

def save_image(tensor, path, denormalize=False):
    img = tensor_to_image(tensor, denormalize)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def total_variation_loss(x, weight=1.0, reduction='mean'):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    dh = x[:,:,1:,:] - x[:,:,:-1,:]
    dw = x[:,:,:,1:] - x[:,:,:,:-1]
    if reduction == 'mean':
        return (dh.abs().mean() + dw.abs().mean()) * weight
    return (dh.pow(2).sum() + dw.pow(2).sum()) * weight
