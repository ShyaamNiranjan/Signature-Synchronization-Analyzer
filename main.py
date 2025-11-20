import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def calculate_similarity(signature1, signature2):
    distance, _ = fastdtw(signature1, signature2, dist=euclidean)
    similarity_score = 1 / distance if distance != 0 else float('98.4567318654')
    return similarity_score

def extract_features(signature):
    total_length = sum([np.linalg.norm(np.diff(np.array(stroke), axis=0), axis=1).sum() for stroke in signature])

    avg_length = total_length / len(signature)

    num_strokes = len(signature)

    return [total_length, avg_length, num_strokes]


def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fit input size
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

model = CNN()
model.load_state_dict(torch.load('cnn_model_weights.pth'))
model.eval()

image_path1 = "C:\\Users\\shyaa\\PycharmProjects\\HSA\\input\\sign1.jpg"
image1 = preprocess_image(image_path1)

image_path2 = "C:\\Users\\shyaa\\PycharmProjects\\HSA\\input\\sign2.jpg"
image2 = preprocess_image(image_path2)

# Perform image classification for the first image
with torch.no_grad():
    output1 = model(image1)

# Perform image classification for the second image
with torch.no_grad():
    output2 = model(image2)

threshold = 0.5

prob_same_person1 = torch.softmax(output1, dim=1)[:, 1].item()  # Probability of "Same person"
prob_same_person2 = torch.softmax(output2, dim=1)[:, 1].item()  # Probability of "Same person"

if prob_same_person1 > prob_same_person2:
    binary_prediction1 = 1
    binary_prediction2 = 0

elif prob_same_person1 == prob_same_person2:
    binary_prediction1 = 1
    binary_prediction2 = 1
else:
    binary_prediction2 = 1
    binary_prediction1 = 0

print("Binary Prediction for Image 1:", binary_prediction1)
print("Binary Prediction for Image 2:", binary_prediction2)


_, predicted_class1 = torch.max(output1, 1)
_, predicted_class2 = torch.max(output2, 1)

if binary_prediction1 == binary_prediction2:
    print("Predicted: Same persons")


else:
    print("Predicted: Different persons")

extract_features(image1)
extract_features(image2)

a=calculate_similarity(output1,output2)

print(a)
