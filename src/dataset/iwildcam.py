import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class IWildCamDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_name = str(row['filename'])
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224)) # Ảnh đen thay thế nếu file lỗi
            
        label = int(row['label_idx'])
        
        class_name = str(row['name'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, class_name

def get_dataloaders(csv_path, img_dir, batch_size, train_transform, val_transform, test_transform):
    df_merged = pd.read_csv(csv_path)
    
    unique_cats = sorted(df_merged['category_id'].unique())
    id_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_cats)}
    df_merged['label_idx'] = df_merged['category_id'].map(id_to_idx)
    
    print(f"Total number of classes: {len(unique_cats)}")
    
    df_merged['split'] = df_merged['split'].astype(str).str.strip()
    
    df_train = df_merged[df_merged['split'] == 'train'].reset_index(drop=True)
    df_val = df_merged[df_merged['split'] == 'val'].reset_index(drop=True)
    df_id_val = df_merged[df_merged['split'] == 'id_val'].reset_index(drop=True)
    df_test = df_merged[df_merged['split'] == 'test'].reset_index(drop=True)
    df_id_test = df_merged[df_merged['split'] == 'id_test'].reset_index(drop=True)
    
    print("=== Dataset partitioning statistics ===")
    print(f" Train set          : {len(df_train)} ")
    print(f" Val set (OOD)      : {len(df_val)} ")
    print(f" ID Val set (ID)    : {len(df_id_val)} ")
    print(f" Test set (OOD)     : {len(df_test)} ")
    print(f" ID Test set (ID)   : {len(df_id_test)} ")
    
    train_dataset = IWildCamDataset(df_train, img_dir, transform=train_transform)
    val_dataset = IWildCamDataset(df_val, img_dir, transform=val_transform)
    id_val_dataset = IWildCamDataset(df_id_val, img_dir, transform=val_transform)
    test_dataset = IWildCamDataset(df_test, img_dir, transform=test_transform)
    id_test_dataset = IWildCamDataset(df_id_test, img_dir, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
    id_val_loader = DataLoader(id_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
    id_test_loader = DataLoader(id_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'id_val': id_val_loader,
        'test': test_loader,
        'id_test': id_test_loader
    }, id_to_idx
