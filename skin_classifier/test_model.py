import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from tqdm.auto import tqdm
import pandas as pd
import random
import argparse
from PIL import Image
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = None


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")


def print_device_info():
    """Print information about the device being used"""
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA is not available. Evaluation will run on CPU!")


def get_data_transforms():
    """Create data transformations for EfficientNet"""
    global data_transform
    
    if data_transform is None:
        data_transform = transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(224),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    return data_transform


def load_test_data(test_dir, batch_size=16):
    """Load and prepare test dataset"""
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found!")
        return None, None
    
    transform = get_data_transforms()
    test_dataset = datasets.ImageFolder(test_dir, transform)
    class_names = test_dataset.classes
    
    print(f"Found {len(test_dataset)} test images across {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        multiprocessing_context='spawn'  
    )
    
    return test_loader, class_names


def load_model(model_path, num_classes=2):
    """Load the EfficientNet model with saved weights"""
    print(f"Loading model from: {model_path}")      
    model = models.efficientnet_b0(weights=None)  
    
    
    model.classifier = nn.Sequential(
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
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  
    print("Model loaded successfully!")
    
    return model


def evaluate_model(model, test_loader, class_names, output_prefix='efficientnet'):
    """Evaluate model with detailed metrics"""
    if model is None or test_loader is None:
        print("Cannot evaluate: model or test data not properly loaded.")
        return None
    
    # Verify model is on correct device
    current_device = next(model.parameters()).device
    if torch.cuda.is_available() and 'cuda' not in str(current_device):
        print("Moving model to GPU for evaluation...")
        model = model.to(device)
    
    model.eval()  # Set model to evaluation mode
    
    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    
    
    all_preds = []
    all_labels = []
    all_probs = []  
    pbar = tqdm(test_loader, desc="Evaluating")
    
    with torch.no_grad(): 
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        
            
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            
            current_acc = (predicted == labels).sum().item() / labels.size(0)
            pbar.set_postfix({'batch_acc': f'{100 * current_acc:.1f}%'})
            
           
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            
           
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    pbar.close()
    
    
    overall_accuracy = 100 * correct / total
    print(f'\nTest Results:')
    print(f'Overall accuracy: {overall_accuracy:.2f}%')
    
    class_accuracy = []
    for i in range(len(class_names)):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            class_accuracy.append({
                'Class': class_names[i],
                'Accuracy': f'{acc:.2f}%',
                'Samples': int(class_total[i]),
                'Correct': int(class_correct[i])
            })
            print(f'Accuracy of {class_names[i]}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
        else:
            class_accuracy.append({
                'Class': class_names[i],
                'Accuracy': 'N/A',
                'Samples': 0,
                'Correct': 0
            })
            print(f'No samples for class: {class_names[i]}')
    save_evaluation_results(overall_accuracy, class_accuracy, all_preds, all_labels, all_probs, class_names, output_prefix)
    
    return overall_accuracy, all_probs, all_labels


def save_evaluation_results(overall_accuracy, class_accuracy, all_preds, all_labels, all_probs, class_names, output_prefix):
    """Save all evaluation results to files and generate visualizations"""
    cm = confusion_matrix(all_labels, all_preds)
    create_confusion_matrix(cm, class_names, output_prefix)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(pd.DataFrame(report_df).round(3))
    pd.DataFrame(class_accuracy).to_csv(f'{output_prefix}_class_accuracy.csv', index=False)
    report_df.to_csv(f'{output_prefix}_classification_report.csv')
    results_df = pd.DataFrame({
        'True_Label': [class_names[i] for i in all_labels],
        'Predicted': [class_names[i] for i in all_preds],
        'Correct': [1 if all_labels[i] == all_preds[i] else 0 for i in range(len(all_labels))]
    })
    results_df.to_csv(f'{output_prefix}_prediction_results.csv', index=False)
    
    print(f"Results saved with prefix: {output_prefix}")


def create_confusion_matrix(cm, class_names, output_prefix):
    """Create and save confusion matrix visualization"""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix for percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  
    
    # Plot with seaborn
    ax = sns.heatmap(
        cm_norm, 
        annot=cm, 
        fmt='d',
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (values: counts, colors: percentages)')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as '{output_prefix}_confusion_matrix.png'")
    plt.close()  


def predict_single_image(model, image, class_names):
    
    """
    Predict the class of a single image.
    
    Args:
        model: Loaded PyTorch model
        class_names (list): List of class names
        
    Returns:
        dict: Dictionary containing prediction results
    """
    
    if model is None:
        print("Error: Model not loaded. Cannot make prediction.")
        return None

        
    try:
        # Open and preprocess image
        transform = get_data_transforms()
        image = Image.fromarray(image)
        img_tensor = transform(image).unsqueeze(0)  
        img_tensor = img_tensor.to(device)
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            predicted_class_idx = torch.argmax(probabilities).item()
            
        # Format results
        class_probs = {
            class_names[i]: probabilities[i].item() * 100 
            for i in range(len(class_names))
        }
        
        sorted_probs = sorted(
            class_probs.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        result = {
            'predicted_class': class_names[predicted_class_idx],
            'confidence': probabilities[predicted_class_idx].item() * 100,
            'all_probabilities': sorted_probs,
        }
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("\nClass Probabilities:")
        for cls, prob in sorted_probs:
            print(f"  {cls}: {prob:.2f}%")
            
        return result
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Skin Cancer Classification Evaluation')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing processed test images')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--output_prefix', type=str, default='efficientnet',
                        help='Prefix for output files')
    
    
    args = parser.parse_args()
    set_seed(args.seed)
    print_device_info()
    
    try:
        test_loader, class_names = load_test_data(args.test_dir, args.batch_size)
        model = load_model(args.model_path, len(class_names))
        
        evaluate_model(model, test_loader, class_names, args.output_prefix)
        print("\nEvaluation completed successfully!")
    
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print("See traceback for details:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()