import os  
import torch  
import torch.nn as nn  
import numpy as np  
from tqdm import tqdm  
import argparse  
from collections import defaultdict  

from pet_age_estimation.losses.custom_loss import ProbabilisticAgeLoss  
from pet_age_estimation.data.pet_dataset import get_pet_age_dataloaders  
from pet_age_estimation.utils.training_utils import TrainingConfig, TrainingManager  
from pet_age_estimation.utils.ema import ModelEMA  
from pet_age_estimation.models.dual_branch_model import DualBranchAgeEstimator  


class AgeEstimationTrainer:  
    def __init__(self, config):  
        """  
        Comprehensive Training Framework for Pet Age Estimation  
        
        Args:  
        - config: Training configuration dictionary  
        """  
        self.config = config  
        
        # Device Configuration  
        self.device = config.get('device', 'cuda:1')
        
        # Model Initialization  
        self.model = self._initialize_model()  
        
        # Loss Function  
        self.criterion = ProbabilisticAgeLoss().to(self.device)  
        
        # Training Utilities  
        self.training_config = TrainingConfig(  
            learning_rate=config.get('learning_rate', 1e-3),  
            weight_decay=config.get('weight_decay', 1e-5),  
            mixed_precision=config.get('mixed_precision', True)  
        )  
        self.trainer = TrainingManager(self.model, self.training_config)  
        
        # Data Loaders  
        self.train_loader, self.val_loader = get_pet_age_dataloaders(  
            base_dir=config.get('dataset_dir', 'pet_age_estimation/dataset/'),  
            batch_size=config.get('batch_size', 32),
            image_size=config.get('image_size', 384)
        )  
        
        # Initialize EMA
        self.ema = ModelEMA(
            model=self.model,
            decay=config.get('ema_decay', 0.9999),
            device=self.device
        )
    
    def _initialize_model(self):  
        """  
        Initialize the Dual Branch Age Estimation Model  
        """  
        model = DualBranchAgeEstimator(
            num_branches=self.config.get('num_branches', 8),
            dropout_rate=self.config.get('dropout_rate', 0.3)
        ).to(self.device)
        
        return model  
    
    def train_epoch(self, epoch):  
        """  
        Single training epoch  
        """  
        self.model.train()  
        total_loss = 0  
        loss_components_sum = defaultdict(float)  
        
        progress_bar = tqdm(  
            self.train_loader,   
            desc=f'Epoch {epoch}',   
            unit='batch'  
        )  
        
        for batch_idx, (images, targets) in enumerate(progress_bar):  
            # Handle the dictionary of images
            if isinstance(images, dict):
                images = {k: v.to(self.device) for k, v in images.items()}
                # Use the larger size (384) as input, let model handle resizing
                images = images['image_convnext']
            else:
                images = images.to(self.device)
            
            targets = targets.to(self.device)  
            
            # Zero gradients  
            self.trainer.optimizer.zero_grad()  
            
            # Forward pass  
            outputs = self.model(images)  
            
            # Get loss and components
            loss, loss_components = self.criterion(outputs, targets)  
            
            # Backward pass  
            loss.backward()  
            
            # Optimizer step  
            self.trainer.optimizer.step()  
            
            # Update EMA model
            self.ema.update(self.model)  
            
            # Accumulate losses
            total_loss += loss.item()  
            for name, value in loss_components.items():  
                loss_components_sum[name] += value  
            
            # Progress bar update  
            progress_bar.set_postfix({  
                'loss': loss.item(),  
                'avg_loss': total_loss / (batch_idx + 1)  
            })  
        
        # Compute averages
        num_batches = len(self.train_loader)  
        avg_loss = total_loss / num_batches  
        avg_components = {  
            name: value / num_batches 
            for name, value in loss_components_sum.items()  
        }  
        
        return avg_loss  
    
    def validate(self, use_ema=True):  
        """  
        Validation epoch  
        """  
        model_to_evaluate = self.ema.ema if use_ema else self.model
        model_to_evaluate.eval()
        total_loss = 0
        predictions = []
        true_ages = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                # Handle the dictionary of images
                if isinstance(images, dict):
                    images = {k: v.to(self.device) for k, v in images.items()}
                    # Use the larger size (384) as input, let model handle resizing
                    images = images['image_convnext']
                else:
                    images = images.to(self.device)
                    
                targets = targets.to(self.device)
                
                outputs = model_to_evaluate(images)
                loss, _ = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions.append(outputs['prediction'].cpu().numpy())
                true_ages.append(targets.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        true_ages = np.concatenate(true_ages)
        
        mae = np.mean(np.abs(predictions - true_ages))
        mse = np.mean((predictions - true_ages) ** 2)
        
        return mae
    
    def train(self, num_epochs=100):  
        """  
        Full training process with gradual unfreezing  
        """  
        best_mae = float('inf')  
        best_ema_mae = float('inf')  
        
        # Calculate epochs per phase based on total epochs
        phase_ratios = [0.1, 0.3, 0.4, 0.2]  # Must sum to 1.0
        phase_epochs = [int(ratio * num_epochs) for ratio in phase_ratios]
        # Adjust for any rounding errors to match num_epochs exactly
        phase_epochs[-1] += (num_epochs - sum(phase_epochs))
        
        # Training phases with dynamic epoch allocation
        phases = [  
            {'epochs': phase_epochs[0], 'description': 'Training with frozen backbone'},  
            {'epochs': phase_epochs[1], 'description': 'Fine-tuning last 4 layers', 'unfreeze_layers': 4},  
            {'epochs': phase_epochs[2], 'description': 'Fine-tuning last 8 layers', 'unfreeze_layers': 8},  
            {'epochs': phase_epochs[3], 'description': 'Fine-tuning last 12 layers', 'unfreeze_layers': 12}  
        ]  
        
        current_epoch = 1  
        for phase in phases:  
            print(f"\nStarting phase: {phase['description']}")  
            
            # Configure backbone freezing  
            if 'unfreeze_all' in phase and phase['unfreeze_all']:  
                self.model.backbone.unfreeze_backbone()  
                # Use smaller learning rate for full fine-tuning  
                for param_group in self.trainer.optimizer.param_groups:  
                    param_group['lr'] = self.config.get('learning_rate') * 0.3 
            elif 'unfreeze_layers' in phase:  
                self.model.backbone.partial_unfreeze(num_layers=phase['unfreeze_layers'])  
                # Use intermediate learning rate for partial fine-tuning  
                for param_group in self.trainer.optimizer.param_groups:  
                    param_group['lr'] = self.config.get('learning_rate') * 0.8  
            
            # Train for specified number of epochs in this phase  
            for _ in range(phase['epochs']):  
                # Training epoch  
                train_loss = self.train_epoch(current_epoch)  
                
                # Validate with both regular and EMA models
                val_mae = self.validate(use_ema=False)
                val_ema_mae = self.validate(use_ema=True)
                
                # Learning rate scheduler  
                self.trainer.scheduler.step()  
                
                # Print current metrics
                print(f"\nEpoch {current_epoch}:")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Validation MAE (Regular): {val_mae:.4f}")
                print(f"Validation MAE (EMA): {val_ema_mae:.4f}")
                print(f"Learning Rate: {self.trainer.optimizer.param_groups[0]['lr']:.6f}")
                
                # Save checkpoints for both models if they improve
                if val_mae < best_mae:
                    best_mae = val_mae
                    checkpoint_name = os.path.join(
                        '/mnt/sdc/pet_age_estimation_ckpts',
                        f"pet_age_estimator_",
                        'dual_branch',
                        f"phase_{phase['description'].replace(' ', '_')}_"
                        f"epoch_{current_epoch}_"
                        f"mae_{best_mae:.3f}_"
                        f"lr_{self.config.get('learning_rate')}_"
                        f"bs_{self.config.get('batch_size')}_"
                        f"img_{self.config.get('image_size')}.pth"
                    )
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                        'best_mae': best_mae,
                        'epoch': current_epoch
                    }, checkpoint_name)
                
                if val_ema_mae < best_ema_mae:
                    best_ema_mae = val_ema_mae
                    ema_checkpoint_name = os.path.join(
                        '/mnt/sdc/pet_age_estimation_ckpts',
                        f"pet_age_estimator_ema_"
                        'dual_branch',
                        f"phase_{phase['description'].replace(' ', '_')}_"
                        f"epoch_{current_epoch}_"
                        f"mae_{best_ema_mae:.3f}_"
                        f"lr_{self.config.get('learning_rate')}_"
                        f"bs_{self.config.get('batch_size')}_"
                        f"img_{self.config.get('image_size')}.pth"
                    )
                    torch.save({
                        'model_state_dict': self.ema.state_dict(),
                        'best_mae': best_ema_mae,
                        'epoch': current_epoch
                    }, ema_checkpoint_name)
                
                current_epoch += 1  
    
    def _print_trainable_parameters(self):
        """
        Print the trainable status of model parameters
        """
        print("\nTrainable parameters:")
        for name, param in self.model.named_parameters():
            print(f"{name}: {param.requires_grad}")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTrainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")

def parse_arguments():  
    """  
    Parse command-line arguments  
    """  
    parser = argparse.ArgumentParser(description='Pet Age Estimation Training')  
    parser.add_argument('--backbone', type=str, default='convnextv2_huge',   
                        choices=['convnextv2_huge', 'convnextv2_tiny', 'vit'])  
    parser.add_argument('--batch_size', type=int, default=32)  
    parser.add_argument('--learning_rate', type=float, default=1e-4)  
    parser.add_argument('--epochs', type=int, default=100)  
    parser.add_argument('--weight_decay', type=float, default=1e-5)  
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                      help='Decay rate for EMA model')
    return parser.parse_args()  

def main():  
    # Parse arguments  
    args = parse_arguments()  
    
    # Configuration  
    config = {  
        'backbone': args.backbone,  
        'batch_size': args.batch_size,  
        'learning_rate': args.learning_rate,  
        'weight_decay': args.weight_decay,
        'num_branches': 8,  
        'dataset_dir': 'pet_age_estimation/dataset/',  
        'mixed_precision': True,
        'device': args.device,
        'image_size': args.image_size,
        'ema_decay': args.ema_decay,
    }  
    
    # Initialize trainer  
    trainer = AgeEstimationTrainer(config)  
    
    # Train model  
    trainer.train(num_epochs=args.epochs)  

if __name__ == '__main__':  
    main()