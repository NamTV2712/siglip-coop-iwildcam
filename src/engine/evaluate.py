import torch
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=desc, leave=False)
    
    for step, (images, labels, _) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': total_loss / (step + 1), 
            'acc': correct / total
        })
        
    return total_loss / len(dataloader), correct / total