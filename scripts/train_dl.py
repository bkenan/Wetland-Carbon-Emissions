import numpy as np
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error,r2_score
from data_pipeline import X_train_scaled, X_test_scaled, y_train, y_test
from dl_model import net


def main():
    """
    Performs model training.
    Output: The saved model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batchsize = 64
    epochs = 200
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.005
    LR = 0.001

    def dataloaders(X_train_scaled,y_train,X_test_scaled,y_test,batch_size):
        
        # Converting training and test data to TensorDatasets
        trainset = TensorDataset(torch.from_numpy(np.array(X_train_scaled).astype('float32')), 
                                torch.from_numpy(np.array(y_train).astype('float32')).view(-1,1))
        testset = TensorDataset(torch.from_numpy(np.array(X_test_scaled).astype('float32')), 
                                torch.from_numpy(np.array(y_test).astype('float32')).view(-1,1))

        # Creating Dataloaders for our training and test data to allow us to iterate over minibatches 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        return trainloader, testloader

    trainloader,testloader = dataloaders(X_train_scaled,y_train,X_test_scaled,y_test,batchsize)


    def train_model(model,criterion,optimizer,trainloader,epochs,device):
        model = model.to(device)
        model.train()
        cost = []
        for epoch in range(epochs):

            running_loss = 0.0
        
            for i, data in enumerate(trainloader):

                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            cost.append(running_loss)
        return cost

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), momentum = MOMENTUM, weight_decay=WEIGHT_DECAY, lr=LR)
    cost_path = train_model(net,criterion,optimizer,trainloader,epochs,device)

    def test_model(model,test_loader):
        with torch.no_grad(): 
            model = model.to(device)
            model.eval()
            test_preds = []
            for data in test_loader:
                inputs = data[0].to(device)
                outputs = model.forward(inputs)
                test_preds.extend(outputs.cpu().squeeze().tolist())
        return test_preds

    testpreds = test_model(net,testloader)
    r2 = r2_score(y_test, testpreds)
    rmse = mean_squared_error(y_test, testpreds, squared=False)
    print(round(r2,2))
    print(round(rmse,2))
    

if __name__ == '__main__':
    main()