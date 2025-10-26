"""
Training script for Vision Transformer on CIFAR-10
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import os
from pathlib import Path

from model import create_model
from data import get_dataloaders, get_class_names
from utils import (
    set_seed, load_config, save_config, count_parameters,
    save_checkpoint, load_checkpoint, plot_training_curves,
    plot_confusion_matrix, save_classification_report,
    visualize_predictions, AverageMeter, accuracy
)


def get_optimizer(model, config):
    """Create optimizer based on config."""
    opt_config = config['optimizer']
    opt_type = opt_config['type'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if opt_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        momentum = opt_config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
    
    return optimizer


def get_scheduler(optimizer, config):
    """Create learning rate scheduler based on config."""
    sched_config = config['scheduler']
    sched_type = sched_config['type'].lower()
    
    if sched_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['training']['num_epochs']
        )
    elif sched_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config['step_size'],
            gamma=sched_config['gamma']
        )
    elif sched_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=sched_config['gamma'],
            patience=sched_config['patience']
        )
    elif sched_type == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")
    
    return scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, scaler, config, epoch):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if config['training']['mixed_precision'] and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config['training'].get('gradient_clip', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['training']['gradient_clip']
                )
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['training']['gradient_clip']
                )
            
            optimizer.step()
        
        # Measure accuracy
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        
        losses.update(loss.item(), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{top1.avg:.2f}%',
            'top5': f'{top5.avg:.2f}%'
        })
    
    return losses.avg, top1.avg, top5.avg


def validate(model, val_loader, criterion, device, config):
    """Validate the model."""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Measure accuracy
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            
            losses.update(loss.item(), batch_size)
            top1.update(acc1, batch_size)
            top5.update(acc5, batch_size)
            
            # Store predictions
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{top1.avg:.2f}%',
                'top5': f'{top5.avg:.2f}%'
            })
    
    return losses.avg, top1.avg, top5.avg, all_preds, all_labels


def main(args):
    """Main training function."""
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['plot_dir'], exist_ok=True)
    if config['logging']['tensorboard']:
        os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # Save config
    save_config(config, os.path.join(config['checkpoint']['save_dir'], 'config.yaml'))
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
    class_names = get_class_names()
    
    # Create model
    print("\nCreating model...")
    model = create_model(config).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = get_optimizer(model, config)
    
    # Scheduler
    scheduler = get_scheduler(optimizer, config)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
    
    # TensorBoard writer
    writer = None
    if config['logging']['tensorboard']:
        writer = SummaryWriter(config['logging']['log_dir'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(args.resume, model, optimizer)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    print("\nStarting training...")
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc, train_top5 = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config, epoch+1
        )
        
        # Validate
        val_loss, val_acc, val_top5, val_preds, val_labels = validate(
            model, val_loader, criterion, device, config
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Top-5: {train_top5:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Top-5: {val_top5:.2f}%")
        
        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Top5_Accuracy/train', train_top5, epoch)
            writer.add_scalar('Top5_Accuracy/val', val_top5, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        checkpoint_path = os.path.join(
            config['checkpoint']['save_dir'],
            f'checkpoint_epoch_{epoch+1}.pth'
        )
        
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_acc': best_acc,
            'config': config
        }, checkpoint_path, is_best=is_best)
        
        # Plot training curves periodically
        if (epoch + 1) % 10 == 0 or is_best:
            plot_path = os.path.join(config['logging']['plot_dir'], 'training_curves.png')
            plot_training_curves(history, plot_path)
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    # Load best model
    best_model_path = os.path.join(config['checkpoint']['save_dir'], 'checkpoint_epoch_1_best.pth')
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model)
    
    test_loss, test_acc, test_top5, test_preds, test_labels = validate(
        model, test_loader, criterion, device, config
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Top-5 Accuracy: {test_top5:.2f}%")
    
    # Save final plots and reports
    plot_path = os.path.join(config['logging']['plot_dir'], 'final_training_curves.png')
    plot_training_curves(history, plot_path)
    
    cm_path = os.path.join(config['logging']['plot_dir'], 'confusion_matrix.png')
    plot_confusion_matrix(test_labels, test_preds, class_names, cm_path)
    
    report_path = os.path.join(config['logging']['plot_dir'], 'classification_report.txt')
    save_classification_report(test_labels, test_preds, class_names, report_path)
    
    pred_path = os.path.join(config['logging']['plot_dir'], 'predictions.png')
    visualize_predictions(model, test_loader, class_names, device, num_images=16, save_path=pred_path)
    
    if writer is not None:
        writer.close()
    
    print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Vision Transformer on CIFAR-10')
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)