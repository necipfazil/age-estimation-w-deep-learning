# age_estimation.py - Necip Fazil Yildiran

# Age estimation with deep learning
# AdHoc solution for METU CENG483 HW2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

# network model
class Net(nn.Module):
    # constructor
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, 256).cuda() # input layer to hidden layer 1
        self.fc2 = nn.Linear(256, 256).cuda() # hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(256, 256).cuda() # hidden layer 2 to hidden layer 3
        self.fc4 = nn.Linear(256, 1).cuda()   # hidden layer 3 to output layer
    
    # forward
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# returns trainLossHistory, validLossHistory, validAccuracyHistory
# valid and valid_gt data is supplied only for outputting histories of
# .. loss and accuracy on validation data
def train(net, train, train_gt, valid, valid_gt, numOfEpochs, learningRate):
    # Loss func: MeanSquaredError
    criterion = nn.MSELoss()

    # Optimizer: RMSprop, LearningRate: 1e-3
    optimizer = optim.RMSprop(net.parameters(), lr = learningRate)

    # history lists
    trainLossHistory = []
    validLossHistory = []
    validAccuracyHistory = []

    print("Training is starting. It will take 4000 epochs and will dump information every 50 epochs.")
    for epoch in range(0, numOfEpochs):
        # reset optimizer's grad space to all zeros
        optimizer.zero_grad()
        
        # obtain hypothesis using the current state of network
        train_hypothesis = net(train)
        valid_hypothesis = net(valid)
    
        # compute loss from hypothesis and ground truth
        loss_train = criterion(train_hypothesis, train_gt)
        loss_valid = criterion(valid_hypothesis, valid_gt)

        # append the current loss to the loss history
        trainLossHistory.append(loss_train.item())
        validLossHistory.append(loss_valid.item())

        # evaluate accuracy on validation set and append it to history
        validAccuracy = evaluateAccuracy(valid_hypothesis, valid_gt)
        validAccuracyHistory.append(validAccuracy)

        # Write the epoch&loss every 50 epochs
        if(epoch % 50 == 0):
            print(
                "Epoch [%4d]: \tTrainLoss: %7.2f, ValidLoss: %7.2f, ValidAcc: %3.2f"
                %(epoch, loss_train, loss_valid, validAccuracy))

        # to avoid last optimization that is not recorded
        if(epoch + 1 == numOfEpochs):
            break

        # for gradient computation, apply backward()
        loss_train.backward()

        # perform optimization on network model
        optimizer.step()

    # output last training statistic
    print("Training is done with %d epochs. \n\tTrainLoss: %7.2f, ValidLoss: %7.2f, ValidAcc: %3.2f" %(numOfEpochs, trainLossHistory[-1], validLossHistory[-1], validAccuracyHistory[-1]))

    return trainLossHistory, validLossHistory, validAccuracyHistory

# adapted from evaluate.py
def evaluateAccuracy(hypothesis, groundtruth):
    accuracy = torch.sum(abs(hypothesis - groundtruth) < 10).item() / float(hypothesis.shape[0])
    return accuracy

# Draw the accuracy history plot and save it to the file 'acc_history.png'
def plotAccHist(validAccHist, params):
    numOfEpochs = len(validAccHist)

    horizontalSpace = numOfEpochs + numOfEpochs / 14
    plt.axis([0, horizontalSpace, 0, 1])

    xValues = np.arange(0, numOfEpochs, 1)
    plt.plot(xValues, validAccHist, label='Accuracy on Validation Data', c='b', linewidth = 1.0)

    plt.annotate('{:.2f}'.format(validAccHist[-1]), xy=(numOfEpochs, validAccHist[-1]), color='blue')

    (learningRate, numOfHiddenLayers, hiddenLayerSizes) = params
    hyperparameters_str = ' Number of Epochs = ' + str(numOfEpochs)
    hyperparameters_str += '\n Learning Rate = ' + learningRate
    hyperparameters_str += '\n Number of Hidden Layers = ' + numOfHiddenLayers
    hyperparameters_str += '\n Hidden Layer Sizes = ' + hiddenLayerSizes
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)
    plt.text(horizontalSpace / 2.25, 1 - 0.125, hyperparameters_str, horizontalalignment='left', verticalalignment='top', bbox = props)

    plt.legend()

    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title("Accuracy History on Validation Data")
    
    plt.savefig('acc_history.png')
    plt.close('all')

# Draw the loss history plot and save it to the file 'loss_history.png'
def plotLossHist(trainLossHist, validLossHist, params):
    numOfEpochs = len(trainLossHist)
    
    # max(lossHistory) could be used to capture all the loss history
    # .. however, for making comparison easier and outputting a more precise
    # .. plot, use a constant
    yMax = 1800
    horizontalSpace = numOfEpochs + numOfEpochs * 0.12
    plt.axis([-10, horizontalSpace, 0, yMax])

    xValues = np.arange(0, numOfEpochs, 1)
    plt.plot(xValues, trainLossHist, label='Loss History on Training Data', c='b', linewidth = 1.0)
    plt.plot(xValues, validLossHist, label='Loss History on Validation Data', c='r', linewidth = 1.0)

    # annotate last values
    plt.annotate('{:.2f}'.format(trainLossHist[-1]), xy=(numOfEpochs, trainLossHist[-1] - 5), color='blue')
    plt.annotate('{:.2f}'.format(validLossHist[-1]), xy=(numOfEpochs, validLossHist[-1] + 5), color='red')

    (learningRate, numOfHiddenLayers, hiddenLayerSizes) = params
    hyperparameters_str = ' Number of Epochs = ' + str(numOfEpochs)
    hyperparameters_str += '\n Learning Rate = ' + learningRate
    hyperparameters_str += '\n Number of Hidden Layers = ' + numOfHiddenLayers
    hyperparameters_str += '\n Hidden Layer Sizes = ' + hiddenLayerSizes

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)
    plt.text(horizontalSpace / 2.25, yMax * 0.82, hyperparameters_str, horizontalalignment='left', verticalalignment='top', bbox = props)

    plt.legend()

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title("Loss History on Training and Validation Data")
    
    plt.savefig('loss_history.png')
    plt.close('all')

# adapted from homework files
def feature_extraction(imagePath):
    fe = Img2Vec(cuda=True) # change this if you use Cuda version of the PyTorch.
    img = Image.open(imagePath)
    img = img.resize((224, 224))
    feats = fe.get_vec(img).reshape(1, -1)
    return feats

if __name__=='__main__':
    # notice that all tensor operations are done on GPU (CUDA)
    # this makes the computations much faster

    # assign a manual seed, so that, the results become comparable while testing
    # manual seed is obtained from a session by calling inital_seed
    # ! The implementation is finalized. Therefore, comment out manual seed
    # torch.manual_seed(1525968144)

    # load training data
    training = np.load('train.npy')
    training_gt = np.load('train_gt.npy')

    # numpy inputs are read as DoubleTensor
    # convert the input to required form
    training = torch.from_numpy(training).float().cuda()

    training_gt = torch.from_numpy(training_gt).float()
    training_gt = training_gt.view(training_gt.shape[0], 1).cuda()

    # load validation data
    validation = np.load('valid.npy')
    validation_gt = np.load('valid_gt.npy')

    # convert the input to required form
    validation = torch.from_numpy(validation).float().cuda()
    validation_gt = torch.from_numpy(validation_gt).float()
    validation_gt = validation_gt.view(validation_gt.shape[0], 1).cuda()

    # create the network
    net = Net().cuda()

    # train the network
    numOfEpochs = 4000
    learningRate = 1e-5
    trainLossHist, validLossHist, validAccHist = train(net, training, training_gt, validation, validation_gt,
        numOfEpochs, learningRate)

    # draw estimations on test data

    # load test data (test.npy)
    test = np.load("test.npy")

    # convert the input to the required form
    test = torch.from_numpy(test).float().cuda()

    # draw estimations
    hypothesisOnTest = net(test)

    # save the estimations to estimations_test.npy
    np.save("esimations_test.npy", hypothesisOnTest.detach().cpu().numpy())

    # Save model
    # ! If you want to save model, use the following commented lines of code
    #print("Saving model")
    #torch.save(net, 'ae_network.pt')

    # plot the histories
    # ! If you want to save the plots of histories returned from training,
    # .. use the following lines of code
    #params = ('1e-5', '3', '[256,256,256]') # learningRate, numOfHiddenLayers, hiddenLayerSizes
    #plotLossHist(trainLossHist, validLossHist, params)
    #plotAccHist(validAccHist, params)
