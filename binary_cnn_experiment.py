import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision import transforms, datasets, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.isotonic import IsotonicRegression
from estimators import accuracy, precision, recall, f1, calculate_expectation
from utils import calculate_ese

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sampling settings
TRIALS = 1000     # number of random samples
WINDOW = 100      # number of instances per trial

# Dataset configurations
DATASETS = [
    {
        'name': 'MNIST_0_vs_rest',
        'class_labels': (0, None),  # digit 0 vs all other digits
        'dataset_cls': datasets.MNIST,
        'input_size': (1, 28, 28)
    },
    {
        'name': 'FashionMNIST_0_vs_rest',
        'class_labels': (0, None),  # class 0 (T-shirt) vs all others
        'dataset_cls': datasets.FashionMNIST,
        'input_size': (1, 28, 28)
    },
    {
        'name': 'CIFAR10_cat_vs_dog',
        'class_labels': (3, 5),  # cat=3, dog=5
        'dataset_cls': datasets.CIFAR10,
        'input_size': (3, 32, 32)
    }
]

# Transform pipelines
ID_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
OOD_SHIFT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.GaussianBlur(kernel_size=5),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Utility: filter to two classes
def make_binary_subset(dataset, class_a, class_b):
    """
    Return a BinaryDataset wrapping 'dataset' filtered to class_a vs class_b (or rest if class_b is None).
    """
    # Determine indices to include and class mapping
    if class_b is None:
        # class_a vs rest
        indices = [i for i, (_, label) in enumerate(dataset)
                   if True]  # include all
        # Build class_map: class_a->0, others->1
        class_map = {int(class_a): 0}
        # Ensure dataset.targets is a list of ints
        if hasattr(dataset, 'targets'):
            labels = dataset.targets.tolist() if hasattr(dataset.targets, 'tolist') else [int(l) for l in dataset.targets]
        else:
            labels = [int(l) for _, l in dataset]
        unique_labels = set(labels)
        for lbl in unique_labels:
            if lbl != class_a:
                class_map[int(lbl)] = 1
    else:
        # specific binary classes
        indices = [i for i, (_, label) in enumerate(dataset) if label in (class_a, class_b)]
        class_map = {int(class_a): 0, int(class_b): 1}
    # Create BinaryDataset with correct mapping
    binary = BinaryDataset(dataset, indices)
    binary.class_map = class_map
    return binary


class BinaryDataset(Subset):
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, self.class_map[y]

# Training loop
def train_model(model, criterion, optimizer, dataloaders, num_epochs=10, patience=3):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE, dtype=torch.float)
                outputs = model(inputs).view(-1)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
        epoch_val_loss = val_loss / len(dataloaders['val'].dataset)
        print(f"Epoch {epoch}/{num_epochs-1} - Val Loss: {epoch_val_loss:.4f}")
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
    model.load_state_dict(best_model_wts)
    return model

# Evaluation metrics
def evaluate_metrics(model, calibrator, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs).view(-1)
            u_probs = torch.sigmoid(logits).cpu().numpy()
            c_probs = calibrator.predict(u_probs)
            preds = (u_probs >= 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        ese = calculate_ese(np.array(all_labels), np.array(c_probs), scheme='dynamic')
        print("Adapted Expected Calibration error is", ese)
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }


# Estimated metrics
def estimate_metrics(model, calibrator, dataloader):
    """
    Estimates perfomance for four metrics
    """
    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs).view(-1)
            u_probs = torch.sigmoid(logits).cpu().numpy()
            c_probs = calibrator.predict(u_probs)
    return {
        'accuracy': calculate_expectation(accuracy(c_probs, zero_division=0)),
        'precision': calculate_expectation(precision(c_probs, zero_division=0)),
        'recall': calculate_expectation(recall(c_probs, zero_division=0)),
        'f1': calculate_expectation(f1(c_probs, zero_division=0))
    }


# Performance estimation with sampling
def estimate_performance(model, calibrator, dataset, split_name, dataset_name):
    """
    Compute true metrics on full dataset, then perform TRIALS samples
    of size WINDOW, estimate metrics, and compute MAE and std of errors.
    """
    # True metrics on entire dataset
    full_loader = DataLoader(dataset, batch_size=WINDOW, shuffle=False)
    true_metrics = evaluate_metrics(model, calibrator, full_loader)

    # Prepare containers for absolute errors
    error_lists = {k: [] for k in true_metrics}

    # Sampling trials
    for t in range(TRIALS):
        print(f"Trial number: {t}", end='\r', flush=True)
        idxs = random.sample(range(len(dataset)), WINDOW)
        sample_subset = Subset(dataset, idxs)
        loader = DataLoader(sample_subset, batch_size=WINDOW, shuffle=False)
        est_metrics = estimate_metrics(model, calibrator, loader)
        for k in true_metrics:
            error_lists[k].append(abs(est_metrics[k] - true_metrics[k]))

    # Compute MAE and std for each metric
    mae = {k: sum(v for v in error_lists[k]) / len(error_lists[k]) for k in error_lists}
    std = {k: np.std(error_lists[k]) for k in error_lists}

    print(f"{dataset_name} - {split_name} True metrics: {true_metrics}")
    print(f"{dataset_name} - {split_name} MAE over {TRIALS} trials: {mae}")
    print(f"{dataset_name} - {split_name} Std of errors: {std}")


def train_calibrator(model, dataset, dataset_name):
    """
    Train a calibration mapping
    """
    # The whole dataset
    model.eval()
    dataloader = DataLoader(dataset, batch_size=WINDOW, shuffle=False)
    
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')   
        calibrator.fit(all_probs, all_labels)
    
    return calibrator


# Main experiment
def run_experiment():
    for cfg in DATASETS:
        name = cfg['name']
        a, b = cfg['class_labels']
        ds_cls = cfg['dataset_cls']

        print(f"\n=== Dataset: {name} ===")
        full_train = ds_cls(root='./data', train=True, download=True, transform=ID_TRANSFORM)
        full_test  = ds_cls(root='./data', train=False, download=True, transform=ID_TRANSFORM)

        bin_trainval = make_binary_subset(full_train, a, b)
        bin_test     = make_binary_subset(full_test, a, b)
        if b is not None:
            bin_trainval.class_map = {a: 0, b: 1}
            bin_test.class_map     = {a: 0, b: 1}

        # Split ID-train, ID-val
        n_total = len(bin_trainval)
        n_test  = min(1000, len(bin_test))
        n_val   = int(0.1 * (n_total - n_test))
        n_train = n_total - n_val

        train_subset, val_unused = random_split(bin_trainval, [n_train, n_val])
        test_id_dataset = bin_test
        test_ood_dataset = make_binary_subset(
            ds_cls(root='./data', train=False, download=True, transform=OOD_SHIFT_TRANSFORM),
            a, b)
        if b is not None:
            test_ood_dataset.class_map = {a:0, b:1}

        # Model setup: LeNet-5 architecture
        model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 224x224 -> 110x110
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 110x110 -> 53x53
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(120 * 49 * 49, 84),
            nn.Tanh(),
            nn.Linear(84, 1)
        )
        model = model.to(DEVICE)

        # model = models.resnet18(pretrained=True)
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, 1)
        # model = model.to(DEVICE)

        # Train
        print("Begin training")
        loaders = {
            'train': DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4),
            'val':   DataLoader(val_unused,   batch_size=64, shuffle=False, num_workers=4)
        }
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        model = train_model(model, criterion, optimizer, loaders, num_epochs=10, patience=3)

        # Train calibrator
        calibrator = train_calibrator(model, val_unused, name)

        # Evaluate ID and OoD with sampling
        print("Estimating ID performance")
        estimate_performance(model, calibrator, test_id_dataset, 'ID-test', name)
        print("Estimating OoD performance")
        estimate_performance(model, calibrator, test_ood_dataset, 'OoD-test', name)

if __name__ == '__main__':
    run_experiment()
