import json
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class ToolDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split:str,
        use_aug=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split

        self.get_files()
        if use_aug:
            self.T = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomGrayscale(p=0.2),
                # transforms.RandomApply([transforms.ColorJitter(0.3, 0.1, 0.05, 0.05)], p=0.5),
                # transforms.RandomApply([transforms.GaussianBlur(31, 2)], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.T = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __getitem__(self, index: int):
        """
        return :
            shot: [B, 2, 6, 3, 128, 128]
            query: [B, 2, 1, 3, 128, 128]
        """
        out = {}
        img_paths = self.img_files[index]
        concept = self.concepts[index]
        imgs = [self.T(Image.open(img_path).convert('RGB')) for img_path in img_paths]
        pos_shot = torch.stack(imgs[:6])
        neg_shot = torch.stack(imgs[6:12])
        shot = torch.stack([pos_shot, neg_shot]) # 6pos + 6neg [2, 6, 3, 128, 128]
        query = torch.stack(imgs[12:]) # 1pos + 1neg [2, 1, 3, 128, 128]
        out['shot'] = shot
        out['query'] = query
        out['concept'] = concept
        return out

    def __len__(self):
        return len(self.img_files)
    
    def get_files(self):
        with open(f'{self.data_root}/{self.split}.json', 'r') as f:
            files = json.load(f)
        keys = list(files.keys())
        self.img_files = [files[key] for key in keys]
        self.concepts = keys


class ToolDataModule():
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_root: str,
        split_type: str,
        use_aug: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        if split_type == 'NS':
            train_spilt = 'train'
            test_split = 'test'
        elif split_type == 'CGS':
            train_spilt = 'train_func'
            test_split = 'test_func'
        else:
            raise ValueError('split_type must be NS or CGS')

        self.train_dataset = ToolDataset(data_root, train_spilt, use_aug)
        self.test_dataset = ToolDataset(data_root, test_split)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


'''test'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    data_root = '/code/Dataset'
    use_aug = False
    batch_size = 10
    num_workers = 0
    split_type = 'concept'

    datamodule = ToolDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        data_root=data_root,
        split_type=split_type,
        use_aug=use_aug,
    )
    dl = datamodule.val_dataloader()
    it = iter(dl)
    batch = next(it)
    print(batch['shot'].shape)
    print(batch['query'].shape)
    # idx = 4
    # shot = batch['shot'][idx] # [2, 6, 3, 128, 128]
    # query = batch['query'][idx:idx+1] # [1, 2, 3, 128, 128]
    # concept = batch['concept'][idx]
    # print(concept)
    # T = transforms.ToPILImage()
    # import torchvision.utils as vutils
    # images = vutils.make_grid(
    #         shot.reshape(-1, 3, 128, 128), normalize=False, nrow=6,
    #         padding=3, pad_value=0,
    #     )
    # T(images).save('shot.png')
    # images = vutils.make_grid(
    #         query.reshape(-1, 3, 128, 128), normalize=False, nrow=6,
    #         padding=3, pad_value=0,
    #     )
    # T(images).save('query.png')