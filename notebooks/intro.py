from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import rasterio as rio
from rasterio.plot import show

ROOT_DIR = Path().cwd()

DATA_DIR = ROOT_DIR / 'data' / 'raw'


def img_loader(path: str):
    with rio.open(path) as src:
        array = src.read(out_dtype="int16")
        print(type(array))
    return torch.from_numpy(array)


data_transform = T.Compose([
    T.Resize(size=(250, 250))
])

data = datasets.DatasetFolder(
    root=ROOT_DIR / 'data' / 'raw',
    loader=img_loader,
    extensions=['.tif'],
    transform=data_transform
)

data_loader = DataLoader(data, shuffle=False, batch_size=1)

for i in range(1):
    x, y = next(iter(data_loader))
    print(x.shape)
    

show(x[0].numpy(), adjust=True)

