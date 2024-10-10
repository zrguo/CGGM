from torch.utils.data import DataLoader

from datasets.CMUDataset import CMUData
from datasets.IEMODataset import IEMOData
from datasets.FOODDataset import Food101
from datasets.BratsDataset import BraTSData


class opt:
    cvNo = 1
    A_type = "comparE"
    V_type = "denseface"
    L_type = "bert_large"
    norm_method = 'trn'
    in_mem = False


def getdataloader(dataset, batch_size, data_path):
    if dataset == 'mosi':
        data = {
            'train': CMUData(data_path, 'train'),
            'valid': CMUData(data_path, 'valid'),
            'test': CMUData(data_path, 'test'),
        }
        orig_dim = data['test'].get_dim()
        dataLoader = {
            ds: DataLoader(data[ds],
                           batch_size=batch_size,
                           num_workers=8)
            for ds in data.keys()
        }
    elif dataset == 'iemo':
        data = {
            'train': IEMOData(opt, data_path, set_name='trn'),
            'valid': IEMOData(opt, data_path, set_name='val'),
            'test': IEMOData(opt, data_path, set_name='tst'),
        }
        orig_dim = data['test'].get_dim()
        dataLoader = {
            ds: DataLoader(data[ds],
                           batch_size=batch_size,
                           drop_last=False,
                           collate_fn=data['test'].collate_fn)
            for ds in data.keys()
        }
    elif dataset == 'food':
        data = {
            'train': Food101(mode='train', dataset_root_dir=data_path),
            'valid': Food101(mode='test', dataset_root_dir=data_path),
            'test': Food101(mode='test', dataset_root_dir=data_path),
        }
        orig_dim = None
        dataLoader = {
            ds: DataLoader(data[ds],
                           batch_size=batch_size,
                           num_workers=8)
            for ds in data.keys()
        }
    elif dataset == 'brats':
        data = {
            'train': BraTSData(root=data_path, mode='train'),
            'valid': BraTSData(root=data_path, mode='valid'),
            'test': BraTSData(root=data_path, mode='test'),
        }
        orig_dim = None
        dataLoader = {
            ds: DataLoader(data[ds],
                           batch_size=batch_size,
                           num_workers=8)
            for ds in data.keys()
        }

    return dataLoader, orig_dim