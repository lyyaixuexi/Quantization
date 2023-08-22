import numpy as np

class LinearRegressionDataset:
    def __init__(self, N, d):
        self.X = np.random.randn(N, d)
        self.Y = np.random.randn(N)

    def __getitem__(self, idx):
        return (self.X[idx].astype('float32'), self.Y[idx])

    def __len__(self):
        return len(self.X)

N, d = (100, 6)
dataset = LinearRegressionDataset(N, d)

from ffcv.fields import NDArrayField, FloatField

writer = DatasetWriter(write_path, {
    'covariate': NDArrayField(shape=(d,), dtype=np.dtype('float32')),
    'label': FloatField(),

}, num_workers=16)