import numpy as np
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.optim as optim

import Dataset.DataLoader
import Dataset.DataReader
import Model.DeepSNR

FeatureMatrix, DenseLabel = Dataset.DataReader.DataReader(TFName='CDX2', DataSetName='ChIP-seq')
FeatureMatrix, DenseLabel = np.array(FeatureMatrix), np.array(DenseLabel)

CrossFold = sklearn.model_selection.KFold(n_splits=5)

LossFunction = nn.BCELoss()

MaxEpoch = 1

for TrainIndex, TestIndex in CrossFold.split(FeatureMatrix):
    NeuralNetwork = Model.DeepSNR.DeepSNR(SequenceLength=100, MotifLength=15)
    optimizer = optim.Adam(NeuralNetwork.parameters())
    TrainFeatureMatrix = torch.tensor(FeatureMatrix[TrainIndex], dtype=torch.float32).unsqueeze(dim=1)
    TrainDenseLabels = torch.tensor(DenseLabel[TrainIndex])
    TestFeatureMatrix = torch.tensor(FeatureMatrix[TestIndex], dtype=torch.float32).unsqueeze(dim=1)
    TestDenseLabels = torch.tensor(DenseLabel[TestIndex])
    TrainLoader = Dataset.DataLoader.SampleLoader(FeatureMatrix=TrainFeatureMatrix, DenseLabel=TrainDenseLabels, BatchSize=32)
    TestLoader = Dataset.DataLoader.SampleLoader(FeatureMatrix=TestFeatureMatrix, DenseLabel=TestDenseLabels, BatchSize=8)
    for Epoch in range(MaxEpoch):
        NeuralNetwork.train()
        for step, data in enumerate(TrainLoader, start=0):
            optimizer.zero_grad()
            X, Y = data
            Prediction = NeuralNetwork(X)
            Loss = LossFunction(Prediction.squeeze(), Y.to(torch.float32))
            Loss.backward()
            optimizer.step()

print('Finished Training')