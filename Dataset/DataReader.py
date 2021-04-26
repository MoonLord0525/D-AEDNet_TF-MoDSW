import scipy.io
import os
import Utils.OneHot

'''
    TFName represents name of TFs
    DataSetName represents name of Datasets
'''


def DataReader(TFName, DataSetName):
    FeatureMatrix = []
    DenseLabels = []
    if DataSetName == 'ChIP-exo':
        '''
            AR and CTCF -> labels
            GR -> label
        '''
        DenseLabels = scipy.io.loadmat(os.path.dirname(__file__) + '/ChIP-exo/' + TFName + '/label.mat')['labels']
        with open(os.path.dirname(__file__) + '/ChIP-exo/' + TFName + '/sequence.txt', 'r') as SReader:
            for line in SReader.readlines():
                FeatureMatrix.append(list(map(str, line.rstrip('\n'))))
        FeatureMatrix = Utils.OneHot.OneHot(sequence=FeatureMatrix, number=len(FeatureMatrix), nucleotide=4,
                                            length=DenseLabels.shape[1])
    else:
        with open(os.path.dirname(__file__) + '/ChIP-seq/' + TFName + '/sequence.txt', 'r') as SReader, open(
                  os.path.dirname(__file__) + '/ChIP-seq/' + TFName + '/label.txt', 'r') as LReader:
            for SRLine, LRLine in zip(SReader.readlines(), LReader.readlines()):
                FeatureMatrix.append(SRLine.rstrip('\n'))
                DenseLabels.append(list(map(int, [label for label in LRLine.rstrip('\n')])))
        FeatureMatrix = Utils.OneHot.OneHot(sequence=FeatureMatrix, number=len(FeatureMatrix), nucleotide=4,
                                            length=len(DenseLabels[1]))
    return FeatureMatrix, DenseLabels
