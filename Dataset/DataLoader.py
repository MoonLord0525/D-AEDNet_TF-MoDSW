from Dataset import MyDataSet
from torch.utils.data import DataLoader


def SampleLoader(FeatureMatrix, DenseLabel, BatchSize):
    Loader = DataLoader(
        dataset=MyDataSet.MyDataSet(FeatureMatrix, DenseLabel),
        batch_size=BatchSize,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    return Loader
