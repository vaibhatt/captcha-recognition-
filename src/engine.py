from tqdm import tqdm
import torch
import config

def train_fn(model,dataloader,optimizer):
    model.train()
    fin_loss = 0
    tk = tqdm(dataloader,total = len(dataloader))
    for data in tk:
        for k,v in data.items():
            data[k] = v.to(config.DEVICE)
        optimizer.zero_grad()
        _,loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss/len(dataloader)

def eval_fn(model,dataloader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    tk0 = tqdm(dataloader,total = len(dataloader))
    for data in tk0:
        for k,v in data.items():
            data[k] = v.to(config.DEVICE)
        with torch.no_grad():
            batch_preds,loss = model(**data)
            fin_loss += loss.item()
            fin_preds.append(batch_preds)
    return fin_loss/len(dataloader),fin_preds