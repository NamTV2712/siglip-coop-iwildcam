import os
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim import AdamW
import wandb

# Absolute imports from the root directory
from src.dataset.iwildcam import get_dataloaders
from src.models.coop import CustomCLIP
from src.engine.train import train_one_epoch
from src.engine.evaluate import evaluate

def main():
    # 1. Load Training Configurations
    with open("configs/train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Initialize Weights & Biases for experiment tracking
    wandb.init(
        project=cfg['wandb']['project'],
        name=cfg['wandb']['run_name'],
        config=cfg 
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using runtime device: {device}")

    # 2. Define Image Transformations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # 3. Initialize DataLoaders
    # The get_dataloaders function returns a dictionary with keys: 'train', 'val', 'id_val', 'test', 'id_test'
    loaders, id_to_idx = get_dataloaders(
        csv_path=cfg['data']['csv_path'],
        img_dir=cfg['data']['img_dir'],
        batch_size=cfg['data']['batch_size'],
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=val_transform
    )

    # Construct classname array aligned with mapped indices
    classnames = [None] * len(id_to_idx)
    for cat_id, idx in id_to_idx.items():
        classnames[idx] = f"category_{cat_id}" 

    # 4. Instantiate the CoOp + LoRA Model Architecture
    model = CustomCLIP(
        classnames=classnames,
        model_name="google/siglip2-base-patch32-256", 
        n_ctx=cfg['model']['n_ctx'],
        ctx_init=cfg['model']['ctx_init'],
        apply_lora=True
    ).to(device)

    # Watch the model gradients and parameters with WandB
    wandb.watch(model, criterion=None, log="all", log_freq=100)

    # Filter parameters requiring gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg['training']['lr'])
    criterion = nn.CrossEntropyLoss() 

    # Create output directory
    os.makedirs(cfg['training']['output_dir'], exist_ok=True)

    # 5. Core Training and Validation Loop
    epochs = cfg['training']['epochs']
    best_ood_val_acc = 0.0 # Standard WILDS benchmark: model selection based on OOD validation
    best_model_path = os.path.join(cfg['training']['output_dir'], "best_model.pth")

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*20} Epoch {epoch}/{epochs} {'='*20}")
        
        # 5.1 Training Phase
        train_loss, train_acc = train_one_epoch(model, loaders['train'], optimizer, criterion, device)
        
        # 5.2 Validation Phase: In-Distribution (Same cameras as train set)
        id_val_loss, id_val_acc = evaluate(model, loaders['id_val'], criterion, device, desc="ID Val")
        
        # 5.3 Validation Phase: Out-of-Distribution (New cameras - CRUCIAL FOR GENERALIZATION)
        ood_val_loss, ood_val_acc = evaluate(model, loaders['val'], criterion, device, desc="OOD Val")
        
        print(f"-> Train      | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"-> ID Val     | Loss: {id_val_loss:.4f} | Acc: {id_val_acc:.4f}")
        print(f"-> OOD Val    | Loss: {ood_val_loss:.4f} | Acc: {ood_val_acc:.4f}")
        
        # Log all metrics to WandB dashboard
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val_ID/loss": id_val_loss,
            "val_ID/accuracy": id_val_acc,
            "val_OOD/loss": ood_val_loss,
            "val_OOD/accuracy": ood_val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Save the optimal model checkpoint based on OOD validation accuracy
        if ood_val_acc > best_ood_val_acc:
            best_ood_val_acc = ood_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f" Saved new best model! (OOD Acc: {best_ood_val_acc:.4f})")
            
            # Sync the saved checkpoint file directly to WandB Cloud
            wandb.save(best_model_path, base_path=cfg['training']['output_dir'])

    # =================================================================
    # 6. Final Evaluation on Test Sets (Post-Training)
    # =================================================================
    print("\n" + "="*50)
    print("TRAINING COMPLETE. RUNNING FINAL EVALUATION...")
    print("="*50)
    
    # Load the best weights acquired during training
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # Evaluate on ID Test Set
    id_test_loss, id_test_acc = evaluate(model, loaders['id_test'], criterion, device, desc="Final ID Test")
    print(f"[FINAL] ID Test  | Loss: {id_test_loss:.4f} | Acc: {id_test_acc:.4f}")
    
    # Evaluate on OOD Test Set
    ood_test_loss, ood_test_acc = evaluate(model, loaders['test'], criterion, device, desc="Final OOD Test")
    print(f"[FINAL] OOD Test | Loss: {ood_test_loss:.4f} | Acc: {ood_test_acc:.4f}")
    
    # Log final benchmark results to WandB summary
    wandb.summary["final_test_ID_loss"] = id_test_loss
    wandb.summary["final_test_ID_accuracy"] = id_test_acc
    wandb.summary["final_test_OOD_loss"] = ood_test_loss
    wandb.summary["final_test_OOD_accuracy"] = ood_test_acc

    # Explicitly finish the WandB run
    wandb.finish()

if __name__ == '__main__':
    main()