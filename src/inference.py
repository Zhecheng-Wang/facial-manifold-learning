import torch

def infer(model, x):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.inference_mode():
        x = torch.tensor(x).to(device, dtype=torch.float32)
        y = model(x)
        if isinstance(y, tuple):
            y = y[0]
        y = y.detach().cpu().numpy()
    return y
