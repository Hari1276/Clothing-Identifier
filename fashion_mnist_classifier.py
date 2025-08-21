import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    if training:
        train_set=datasets.FashionMNIST("./data",train=True, download=True, transform=transform)
        return torch.utils.data.DataLoader(train_set, batch_size = 64)
    else:
        test_set=datasets.FashionMNIST("./data", train=False, transform=transform)
        return torch.utils.data.DataLoader(test_set, batch_size = 64)




def build_model():
    """

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 10))
    return model





def train_model(model, train_loader, criterion, T):
    """

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """

    ## used chatGPT

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):
        model.train()  # Set the model to training mode
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
    
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradients
        
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
        
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
        
            total_loss += loss.item()
    
        accuracy = (total_correct / total_samples) * 100
        avg_loss = total_loss / len(train_loader)
    
        # Print training status
        print(f'Train Epoch: {epoch} Accuracy: {total_correct}/{total_samples} ({accuracy:.2f}%) Loss: {avg_loss:.2f}')


    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """

    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
            total_loss += loss.item()

    accuracy = (total_correct / total_samples) * 100
    avg_loss = total_loss / len(test_loader)

    if show_loss:
        print(f'Average loss: {avg_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')



def predict_label(model, test_images, index):
    """

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']


    logits = model(test_images[index])
    prob = F.softmax(logits, dim=1)


    ## use chatGPT
    top3_probabilities, top3_classes = torch.topk(prob, 3)
    for i in range(3):
        class_idx = top3_classes[0][i].item()
        probability = top3_probabilities[0][i].item() * 100
        class_name = class_names[class_idx]
        print(f"{class_name}: {probability:.2f}%")


if __name__ == '__main__':
    '''
    Test code
    '''

    train_loader = get_data_loader()

    test_loader = get_data_loader(False)
    model = build_model()

    criterion = nn.CrossEntropyLoss()

    
    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion, show_loss = True)
