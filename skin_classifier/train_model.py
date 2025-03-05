import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import os
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import json
import argparse

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = (targets_one_hot * probs).sum(dim=1)  # Probability of target class
        focal_weight = (1 - pt) ** self.gamma
        

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.gather(0, targets)
                focal_weight = alpha_t * focal_weight
            else:
                alpha_t = self.alpha * targets_one_hot + (1 - self.alpha) * (1 - targets_one_hot)
                alpha_t = alpha_t.gather(1, targets.unsqueeze(1)).squeeze(1)
                focal_weight = alpha_t * focal_weight
        
       
        loss = F.nll_loss(log_probs, targets, reduction='none')
        focal_loss = focal_weight * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        


class SkinCancerClassifier:
    def __init__(self, root_dir, model_type='efficientnet_b0', batch_size=16, num_workers=2, seed=42):
        """
        Initialize the skin cancer classifier.
        
        Args:
            root_dir (str): Root directory containing 'train' and 'val' folders
            model_type (str): Type of model to use (default: 'efficientnet_b0')
            batch_size (int): Batch size for dataloaders
            num_workers (int): Number of workers for dataloaders
            seed (int): Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Set seeds for reproducibility
        self._set_seed()
        
        # Initialize empty attributes
        self.data_transforms = None
        self.image_datasets = None
        self.dataloaders = None
        self.dataset_sizes = None
        self.class_names = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.metrics = None
        
        # Setup components
        self._setup_transforms()
        self._load_datasets()
        self._setup_model()
        self._check_gpu()
        
    def _set_seed(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {self.seed} for reproducibility")
    
    def _setup_transforms(self):
        """Set up data transformations for training and validation."""
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(10),
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1),  # Random translation up to 10% in each direction
                    scale=(0.9, 1.1),      # Random scaling between 90% and 110%
                ),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.Resize(232),  # Slightly larger for crop
                transforms.CenterCrop(224),  # Standard size for EfficientNet-B0
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(232),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    
    def _load_datasets(self):
        """Load datasets and create dataloaders."""
        # Load datasets
        self.image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(self.root_dir, x),
                self.data_transforms[x]
            ) for x in ['train', 'val']
        }
        
        # Create dataloaders with worker init function for seed
        def seed_worker(worker_id):
            worker_seed = self.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.image_datasets[x],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                worker_init_fn=seed_worker,
                generator=g,
                pin_memory=True
            ) for x in ['train', 'val']
        }
        
        # Get dataset sizes and class names
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes
        print(f"Classes: {self.class_names}")
        print(f"Dataset sizes - Train: {self.dataset_sizes['train']}, Val: {self.dataset_sizes['val']}")
    
    def _check_gpu(self):
        """Check GPU availability and print CUDA information."""
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("WARNING: CUDA is not available. Training will be slow on CPU!")
    
    def _setup_model(self):
        """Set up the model architecture."""
        # Load pre-trained model
        if self.model_type == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            
            # Freeze early layers to prevent overfitting
            freeze_layers = 5  # Freeze first 5 layers
            for i, (name, param) in enumerate(self.model.features.named_parameters()):
                if i < freeze_layers:
                    param.requires_grad = False
            
            # Replace the classifier
            num_classes = len(self.class_names)
            self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 1024), 
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported")
        
        # Move model to device
        self.model = self.model.to(self.device)
        print(f"Model moved to {self.device}")
    
    def setup_training(self, learning_rate_base=5e-5, learning_rate_classifier=1e-3, 
                      weight_decay=1e-4, class_weighting=True, use_focal_loss=False, gamma=2.0):
        """
        Set up training components: criterion, optimizer, and scheduler.
        
        Args:
            learning_rate_base (float): Learning rate for the base model
            learning_rate_classifier (float): Learning rate for the classifier
            weight_decay (float): Weight decay for optimizer
            class_weighting (bool): Whether to use class weighting for loss function
            use_focal_loss (bool): Whether to use Focal Loss instead of Cross Entropy
            gamma (float): Focusing parameter for Focal Loss (only used if use_focal_loss=True)
        """
        # Prepare class weights if needed
        if class_weighting:
            class_counts = [0] * len(self.class_names)
            for _, label in self.image_datasets['train']:
                class_counts[label] += 1
            total = sum(class_counts)
            class_weights = [total/count for count in class_counts]
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        else:
            class_weights = None
        
        # Choose loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=class_weights, gamma=gamma)
            print(f"Using Focal Loss with gamma={gamma}" + 
                  (f" and class weights" if class_weighting else ""))
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights if class_weighting else None
            )
            print("Using Cross Entropy Loss" + 
                  (f" with class weights" if class_weighting else ""))
        
        # Optimizer with different learning rates for base model and classifier
        self.optimizer = optim.AdamW([
            {'params': [p for n, p in self.model.named_parameters() 
                       if 'classifier' not in n and p.requires_grad], 
             'lr': learning_rate_base},
            {'params': self.model.classifier.parameters(), 
             'lr': learning_rate_classifier}
        ], weight_decay=weight_decay)
        
        # Scheduler (OneCycleLR policy)
        steps_per_epoch = len(self.dataloaders['train'])
        self.scheduler = lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[learning_rate_base*10, learning_rate_classifier*5],
            steps_per_epoch=steps_per_epoch,
            epochs=40,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000
        )
    
    def train(self, num_epochs=30):
        """
        Train the model.
        
        Args:
            num_epochs (int): Number of epochs to train
            
        Returns:
            model: Trained model
            metrics (dict): Training metrics
        """
        
        if self.criterion is None or self.optimizer is None or self.scheduler is None:
            self.setup_training()
        
    
    
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        # Verify model is on GPU before training
        if torch.cuda.is_available():
            current_device = next(self.model.parameters()).device
            print(f"Training on: {current_device}")
            if 'cuda' not in str(current_device):
                print("WARNING: Model is not on GPU! Moving model to GPU now...")
                self.model = self.model.to(self.device)
        
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs-1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  
                else:
                    self.model.eval() 

                running_loss = 0.0
                running_corrects = 0
                
                pbar = tqdm(self.dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch}')
    
                for inputs, labels in pbar:
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    current_loss = loss.item()
                    pbar.set_postfix({'loss': f'{current_loss:.4f}'})
                
                
                pbar.close()
                
                
                if phase == 'train' and self.scheduler is not None:
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                if phase == 'train':
                    self.metrics['train_loss'].append(float(epoch_loss))
                    self.metrics['train_acc'].append(float(epoch_acc))
                else:
                    self.metrics['val_loss'].append(float(epoch_loss))
                    self.metrics['val_acc'].append(float(epoch_acc))

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
              
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        self.model.load_state_dict(best_model_wts)    
        return self.model, self.metrics
    
    
    def plot_training_progress(self):
        """Plot training and validation metrics."""
        if self.metrics is None:
            print("No metrics available. Please train the model first.")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['train_loss'], label='Training Loss')
        plt.plot(self.metrics['val_loss'], label='Validation Loss')
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['train_acc'], label='Training Accuracy')
        plt.plot(self.metrics['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()
    
    def save_model(self, filepath='best_model.pth'):
        """Save the model to a file."""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the model from a file."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {filepath}")



def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a skin cancer classification model with configurable parameters')
    
    # Dataset parameters
    parser.add_argument('--root_dir', type=str, required=True, 
                        help='Path to the processed dataset directory')
    
    # Model parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--lr_base', type=float, default=5e-5,
                        help='Base learning rate for the model')
    parser.add_argument('--lr_classifier', type=float, default=1e-3,
                        help='Learning rate for the classifier head')
    parser.add_argument('--class_weighting', action='store_true',
                        help='Enable class weighting to handle imbalanced classes')
    
    # Focal Loss parameters
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use Focal Loss instead of Cross Entropy Loss')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='Gamma parameter for Focal Loss')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--samples_for_calibration', type=int, default=200,
                        help='Number of samples to use for calibration analysis')
    
    # Output parameters
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to save the trained model')
    parser.add_argument('--metrics_path', type=str, default='training_metrics.json',
                        help='Path to save the training metrics')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    classifier = SkinCancerClassifier(args.root_dir, seed=args.seed)
    classifier.setup_training(
        learning_rate_base=args.lr_base,
        learning_rate_classifier=args.lr_classifier,
        class_weighting=args.class_weighting,
        use_focal_loss=args.use_focal_loss,
        gamma=args.gamma
    )
    model, metrics = classifier.train(num_epochs=args.epochs)
    classifier.analyze_prediction_confidence(num_samples=args.samples_for_calibration)
    classifier.plot_training_progress()
    classifier.save_model(args.model_path)
    with open(args.metrics_path, 'w') as f:
        json.dump(classifier.metrics, f)
    
    print(f"Training completed. Model saved at {args.model_path}")
    print(f"Training metrics saved at {args.metrics_path}")

if __name__ == "__main__":
    main()