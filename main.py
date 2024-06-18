import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
from torchvision.transforms.functional import invert

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# Fonction d'entraînement
def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# Fonction de test/évaluation
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Fonction pour convertir une image en tenseur
def img_to_tensor(path):
    img = read_image(path, mode=ImageReadMode.GRAY)  # img est un tenseur en nuance de gris
    img = invert(img)  # mon image paint est en noire sur fond blanc, il faut inverser les couleurs
    transform = v2.Compose([
        v2.Resize((28, 28)),
        v2.ConvertImageDtype(torch.float32),
        v2.Normalize((0.1307,), (0.3081,))
    ])
    out = transform(img).unsqueeze(0)
    return out


# Fonction pour prédire la classe d'une image donnée
def predict(img_path):
    device = get_device()
    model = CNN().to(device)
    model.load_state_dict(torch.load('./model/CNN.pth'))
    model.eval()  # Mettre le modèle en mode évaluation

    image_tensor = img_to_tensor(img_path).to(device)
    with torch.no_grad():  # Désactiver le calcul des gradients
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)  # Obtenir la classe prédite
    print(f"Prédiction: {predicted.item()}")  # Afficher la classe prédite


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Définir l'appareil (CPU ou GPU)
    device = get_device()

    # Hyperparamètres (paramètres que l'on fixe avant l'entraînement et qui influent sur la performance du modèle)
    n_epochs = 14
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 1.0
    log_interval = 10
    gamma = 0.7

    # Définir les transformations des données (normalisation, conversion en tenseur, etc.)
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.1307,), (0.3081,))  # Normaliser les images (moyenne et écart-type de l'ensemble MNIST)
    ])

    # Charger les données d'entraînement avec les transformations définies
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Charger les données de test avec les transformations définies
    test_data = datasets.MNIST(root='./data', train=False, transform=transform)

    # Créer un DataLoader pour les données d'entraînement
    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True, num_workers=4)

    # Créer un DataLoader pour les données de test
    test_loader = DataLoader(test_data, batch_size=batch_size_test, num_workers=4)

    # Initialiser le modèle et le déplacer sur le device (GPU ou CPU)
    model = CNN().to(device)

    # Définir l'optimiseur (algorithme pour ajuster les poids du modèle)
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Entraîner et évaluer le modèle pour chaque époque
    for epoch in range(1, n_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), './model/CNN.pth')


# Main function
if __name__ == '__main__':
    # Train the model by calling main()
    # main()

    # Prédire la classe d'une image donnée
    predict("./images/4.jpg")  # Remplacer par le chemin de votre image