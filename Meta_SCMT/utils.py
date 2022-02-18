import numpy as np
import torch

def h2index(h, dh):
    if isinstance(h, np.ndarray):
        h = (np.round(h/dh)).astype(int)
    else:
        h = int(round(h/dh))
    return h

def gen_decay_rate(total_steps, decay_steps):
    expected_lr_decay = 0.1
    decay_rate = np.exp(np.log(expected_lr_decay)/(total_steps/decay_steps))
    print('decay_rate: ', decay_rate)
    return decay_rate

class Model(torch.nn.Module):
    def __init__(self, in_size, out_size, layers = 2, nodes = 64):
        super(Model, self).__init__()
        module_list = [torch.nn.Linear(in_size, nodes), torch.nn.ReLU()]
        for _ in range(layers):
            module_list.append(torch.nn.Linear(nodes, nodes))
            module_list.append(torch.nn.ReLU())
        module_list.append(torch.nn.Linear(nodes, out_size))
        self.fc = torch.nn.Sequential(*module_list)
    def forward(self, x):
        return self.fc(x)

def train(model, X, Y, steps, lr):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    log_steps = int(steps // 10)
    model = model.to(device)
    X = torch.tensor(X, dtype = torch.float, device= device)
    Y = torch.tensor(Y, dtype = torch.float, device= device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    decay_rate = gen_decay_rate(steps, log_steps)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    mse = torch.nn.MSELoss(reduction = 'sum')
    for step in range(steps):
        Y_pred = model(X)
        loss = mse(Y_pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_steps == 0:
            if step != 0:
                my_lr_scheduler.step()
            relative_error = torch.mean(torch.abs(Y_pred - Y)/torch.abs(Y)) * 100
            relative_error = relative_error.cpu().detach().numpy()
            print("relative_error:" + str(relative_error) + "%.")
            if relative_error < 0.1:
                print("fitting error < 0.1%, accurate enough, stoped.")
                break
    if relative_error > 0.1:
        print("fitting error > 0.1%, increase total steps or number of layers in fullconnected network.")
    Y_pred = Y_pred.cpu().detach().numpy()
    return Y_pred