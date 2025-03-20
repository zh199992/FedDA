import torch
def SF(pred, y):
    d = pred - y
    positive_mark = d > 0
    negative_mark = d < 0
    sf1 = torch.exp(d[positive_mark] / 10)
    sf2 = torch.exp(-d[negative_mark] / 13)
    sf = torch.cat([sf1, sf2], dim=0).sum()
    return sf




