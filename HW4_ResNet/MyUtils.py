import time
import pandas as pd
import torch

def calculate_accuracy(model, data_loader, device):
    """
        Calculate the model's predict accuracy in certain dataset
    """

    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(data_loader):
            X_test_batch, Y_test_batch= X_test_batch.to(device),Y_test_batch.to(device)
            
            outputs = model(X_test_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += Y_test_batch.size(0)
            correct += (predicted == Y_test_batch).sum().item()

    return correct/total

def train(epoch, model, train_loader, test_loader, device, optimizer, criterion,
            num_epochs, total_step):
    """
        TODO: 
        1. Train the model, print the loss every 100 batch
        2. Calculate the accuracy on the training set and test set
    """
    
    # Train the model
    model.train()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(train_loader):
        X_train_batch, Y_train_batch = X_train_batch.to(device), Y_train_batch.to(device)

        # Forward Pass
        outputs = model(X_train_batch)
        loss = criterion(outputs, Y_train_batch)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        if (epoch >= 6):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if 'state' in state.keys():
                        if (state['step'] >= 1024):
                            state['step'] = 1000

        optimizer.step()
        
        if (batch_idx+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, batch_idx+1, total_step, loss.item()))

    # Calculate the accuracy
    # on training set
    train_acc = calculate_accuracy(model, train_loader, device)

    # on testing set
    test_acc = calculate_accuracy(model, test_loader, device)

    print('Accuracy on train: {:.2f} %, on test: {:.2f} %'
            .format(100 * train_acc, 100 * test_acc))
    
    return (train_acc, test_acc)