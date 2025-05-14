import os
import torch
import torch.utils.data as Data

def get_data_loader(batch_size, train_data, test_data):
    train_tensor = torch.tensor(train_data.astype(float).tolist(),
                          dtype=torch.float32)
    test_tensor = torch.tensor(test_data.astype(float).tolist(),
                         dtype=torch.float32)

    train_loader = Data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
