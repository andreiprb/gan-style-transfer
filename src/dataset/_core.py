from typing import Literal
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset


class StyleTransferDataset(Dataset):
    def __init__(
        self,
        domain: Literal['photos', 'monet'],
        split: Literal['train', 'val'] = 'train',
        image_size: int = 256,
        augment: bool | None = None,
    ):
        self.domain = domain
        self.split = split
        self.image_size = image_size
        self.augment = augment if augment is not None else (split == 'train')

        self.column = 'imageA' if domain == 'monet' else 'imageB'

        hf_split = 'train' if split == 'train' else 'test'

        print(f"Loading {self.domain} ({self.column}) from HuggingFace...")
        self.dataset = load_dataset(
            "huggan/monet2photo",
            split=hf_split,
            trust_remote_code=True,
        )
        print(f"Loaded {len(self.dataset)} images for {hf_split} split.")

        self.transform = self._build_transforms()

    def _build_transforms(self) -> transforms.Compose:
        transform_list = []

        if self.augment:
            transform_list.extend([
                transforms.Resize(int(self.image_size * 1.1)),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            transform_list.extend([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
            ])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and transform an image."""
        image = self.dataset[idx][self.column]

        if image.mode != 'RGB':
            image = image.convert('RGB')

        return self.transform(image)


class PairedDataset(Dataset):
    def __init__(
        self,
        split: Literal['train', 'val'] = 'train',
        image_size: int = 256,
        augment: bool | None = None,
    ):
        self.split = split
        self.image_size = image_size
        self.augment = augment if augment is not None else (split == 'train')

        hf_split = 'train' if split == 'train' else 'test'

        print(f"Loading paired dataset from HuggingFace...")
        self.dataset = load_dataset(
            "huggan/monet2photo",
            split=hf_split,
            trust_remote_code=True,
        )
        print(f"Loaded {len(self.dataset)} pairs for {hf_split} split.")

        self.transform = self._build_transforms()

    def _build_transforms(self) -> transforms.Compose:
        transform_list = []

        if self.augment:
            transform_list.extend([
                transforms.Resize(int(self.image_size * 1.1)),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            transform_list.extend([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
            ])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataset[idx]

        photo = row['imageB']
        monet = row['imageA']

        if photo.mode != 'RGB':
            photo = photo.convert('RGB')
        if monet.mode != 'RGB':
            monet = monet.convert('RGB')

        return self.transform(photo), self.transform(monet)


def get_dataloaders(
    domain: Literal['photos', 'monet'] | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 256,
    paired: bool = False,
) -> dict[str, DataLoader]:
    dataloaders = {}

    for split in ['train', 'val']:
        if paired:
            dataset = PairedDataset(
                split=split,
                image_size=image_size,
            )
        else:
            if domain is None:
                raise ValueError("Must specify domain when paired=False")
            dataset = StyleTransferDataset(
                domain=domain,
                split=split,
                image_size=image_size,
            )

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
        )

    return dataloaders