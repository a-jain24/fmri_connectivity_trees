import torch
import torch.nn as nn
from torchvision.models import vgg16, resnet18
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# BrainNetCNN architecture for connectivity matrices
class BrainNetCNN(nn.Module):
    def __init__(self, num_nodes, n_classes=2):
        super().__init__()
        # E2E: edge-to-edge convolution layers (Kawahara et al.)
        self.e2econv1 = nn.Conv2d(1, 16, kernel_size=(1,num_nodes))
        self.e2econv2 = nn.Conv2d(16, 32, kernel_size=(num_nodes,1))
        # E2N: edge-to-node
        self.e2nconv = nn.Conv2d(32, 64, kernel_size=(1,num_nodes))
        # N2G: node-to-graph
        self.n2gconv = nn.Conv2d(64, 128, kernel_size=(num_nodes,1))
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        # x shape: [batch, 1, num_nodes, num_nodes]
        x = F.relu(self.e2econv1(x))
        x = F.relu(self.e2econv2(x))
        x = F.relu(self.e2nconv(x))
        x = F.relu(self.n2gconv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Load VGG16, adjust input channels and output classes
class VGG16_Connectivity(nn.Module):
    def __init__(self, n_classes, input_size):
        super().__init__()
        self.vgg = vgg16(pretrained=True)
        # For single-channel connectivity maps
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.vgg.classifier[-1] = nn.Linear(4096, n_classes)
    def forward(self, x):
        return self.vgg(x)

# Load ResNet18, adjust input channels and output classes
class ResNet_Connectivity(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes)
    def forward(self, x):
        return self.resnet(x)

def get_model(n_classes=2, input_size=116, name='vgg16'):

    if name == 'vgg16':
        return VGG16_Connectivity(n_classes=n_classes, input_size=input_size).to(device)  # Replace 116 with your ROI count
    elif name == 'resnet18':
        return ResNet_Connectivity(n_classes=n_classes).to(device)
    elif name == 'brainnetcnn':
        return BrainNetCNN(num_nodes=input_size, n_classes=n_classes).to(device) 
    else:
        raise ValueError(f"Model {name} not supported")

def evaluate(y_true, y_pred, y_pred_prob):

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'roc_auc': roc_auc
    }

def main():
    model = get_model(n_classes=2, input_size=116, name='brainnetcnn')
    print(model)

if __name__ == "__main__":
    main()