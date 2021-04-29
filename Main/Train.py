from tqdm import tqdm

import numpy as np
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.optim as optim

import Dataset.DataLoader
import Dataset.DataReader
import Model.DeepSNR
import Model.D_AEDNet

'''
    Initialization as follows:
'''
TFsName = 'CTCF'
DataSetName = 'ChIP-exo'
NeuralNetworkName = 'DeepSNR'
'''
    Initialization End    
'''

'''
    Main Process as follows: 
'''
FeatureMatrix, DenseLabel = Dataset.DataReader.DataReader(TFName=TFsName, DataSetName=DataSetName)
FeatureMatrix, DenseLabel = np.array(FeatureMatrix), np.array(DenseLabel)

CrossFold = sklearn.model_selection.KFold(n_splits=5)
CurrentFold = 1

LossFunction = nn.BCELoss()

MaxEpoch = 1

for TrainIndex, TestIndex in CrossFold.split(FeatureMatrix):
    if NeuralNetworkName == 'DeepSNR':
        NeuralNetwork = Model.DeepSNR.DeepSNR(SequenceLength=100, MotifLength=15)
    else:
        NeuralNetwork = Model.D_AEDNet.D_AEDNN(SequenceLength=100)
    optimizer = optim.Adam(NeuralNetwork.parameters())
    TrainFeatureMatrix = torch.tensor(FeatureMatrix[TrainIndex], dtype=torch.float32).unsqueeze(dim=1)
    TrainDenseLabels = torch.tensor(DenseLabel[TrainIndex])
    TrainLoader = Dataset.DataLoader.SampleLoader(FeatureMatrix=TrainFeatureMatrix, DenseLabel=TrainDenseLabels, BatchSize=32)

    for Epoch in range(MaxEpoch):
        NeuralNetwork.train()
        ProgressBar = tqdm(TrainLoader)
        for data in ProgressBar:
            ProgressBar.set_description("Epoch %d" % Epoch)
            optimizer.zero_grad()
            X, Y = data
            Prediction = NeuralNetwork(X)
            Loss = LossFunction(Prediction.squeeze(), Y.to(torch.float32))
            Loss.backward()
            optimizer.step()
    torch.save(NeuralNetwork.state_dict(), 'Weight/' + NeuralNetworkName + DataSetName + TFsName + '%dFold.pth' % CurrentFold)
    CurrentFold = CurrentFold + 1
print('Finished Training')
