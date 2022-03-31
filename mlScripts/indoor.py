import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
import time
import torch
from fastai.text.all import *
from sklearn.preprocessing import MinMaxScaler
import joblib


class Model2(nn.Module):
    def __init__(self, vocab_sz, embed_dim, n_hidden, n_layers, p):
        super(Model2, self).__init__()

        self.i_h = nn.Embedding(vocab_sz, embed_dim)
        self.rnn = nn.LSTM(embed_dim + 1, n_hidden, n_layers,
                           batch_first=True, dropout=p)
        self.drop = nn.Dropout(p)
        self.h_o = nn.Linear(n_hidden, 3)

    def forward(self, x):

        print(x.shape)
        raw, _ = self.rnn(torch.cat(
            (self.drop(self.i_h(x[:, :, 0].long())), x[:, :, 1].unsqueeze(2)), 2))
        return self.h_o(self.drop(raw))


def getCoordinates(user_id, rssi, model_path, scaler_path, trial=True):
    print(rssi)
    rssi = abs(rssi)
    if trial == True:
        rssi_list = []
        joblib.dump(rssi_list, f'{user_id}.pkl')
        return [None, None, None]
    else:
        rssi_list = joblib.load(f'{user_id}.pkl')
        rssi_list.append(rssi)
        print(rssi_list)
        if len(rssi_list) < 6:
            joblib.dump(rssi_list, f'{user_id}.pkl')
            return [None, None, None]

        elif len(rssi_list) >= 6:
            rssi_list = rssi_list[len(rssi_list) - 6:]
            joblib.dump(rssi_list, f'{user_id}.pkl')
            embed_dim = 128
            hidden_size = (embed_dim + 1 + 3) // 2
            n_layers = 2
            p = 0

            model = Model2(63115, embed_dim, hidden_size, n_layers, p)
            model.load_state_dict(torch.load(
                model_path, map_location='cpu'), strict=True)
            scalerr = joblib.load(scaler_path)
            inp1 = []
            inp2 = []
            for i in rssi_list:
                for j in range(0, 29):
                    inp1.append(i)
                    inp2.append(-100)
            # inp1 = list(scalerr.transform(np.array(inp1).reshape(-1,1)).flatten())
            fin = torch.zeros(1, 174, 2)
            fin[:, :, 0] = torch.Tensor(inp1)
            fin[:, :, 1] = torch.Tensor(inp2)
            model.eval()
            with torch.no_grad():
                preds = model(fin)[-1].detach().cpu().numpy()
            x_cords = preds[:, 1].mean()
            y_cords = preds[:, 2].mean()
            floor = round(preds[:, 0].mean())

            return [x_cords, y_cords, floor]
