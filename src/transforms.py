"""
Data augmentation / transform pipelines used for support, query, and eval sets.
"""

from torchvision import transforms


def build_transforms(config):
    """
    Return a dict of transforms keyed by role:
      'train'   – heavy augmentation for query images during training
      'support' – light augmentation for stable prototypes
      'eval'    – minimal transforms for validation / test
    """
    train_transform = transforms.Compose([
        transforms.Resize((config.RESIZE_SIZE, config.RESIZE_SIZE)),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        # Hue is the one channel to be careful with here. On a fresh/rotten task
        # the discriminative signal IS colour — browning, darkening, bruising —
        # so shifting hue augments away the label. configs/base.yaml:41-43 already
        # settled on 0.02 with exactly that reasoning; the executed run used 0.05
        # because src/ and configs/ were never wired together. Follow the config.
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=config.HUE_JITTER),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ])

    support_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((config.RESIZE_SIZE, config.RESIZE_SIZE)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return {
        "train": train_transform,
        "support": support_transform,
        "eval": eval_transform,
    }
