from cv2 import sampsonDistance
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import cv2

def lens_2D(total_size, dx, focal_lens, k):
    x = (np.arange(total_size) - (total_size - 1)/2) * dx
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    #phase = k * (r**2 - X**2 - Y**2)/(2 * focal_lens)
    phase = k * (focal_lens - np.sqrt(X**2 + Y**2 + focal_lens**2))
    return x , phase

def fourier_conv(signal: torch.Tensor, f_kernel: torch.Tensor) -> torch.Tensor:
    '''
        args:
        signal size: dim = 2
        signal, kernel: complex tensor, assume the images are square. the last 2 dim of signal is the height, and width of images.
    '''
    s_size = list(signal.size())
    k_size = list(f_kernel.size())
    padding = (k_size[-1] - s_size[-1])//2
    if (k_size[-1] - s_size[-1])%2== 0:
        signal = torch.nn.functional.pad(signal, (padding, padding, padding, padding))
    else:
        signal = torch.nn.functional.pad(signal, (padding, padding + 1, padding, padding + 1))

    f_signal = torch.fft.fftn(signal, dim = (-2, -1))

    f_output = f_signal * f_kernel
    f_output = torch.fft.ifftn(f_output, dim = (-2, -1))
    f_output = f_output[padding: padding + s_size[-1], padding:padding + s_size[-1]]
    
    return f_output

def resize_field2D(Ey, new_size):
    if new_size < Ey.shape[-1]:
        Eys_new = cv2.resize(Ey, (new_size, new_size), interpolation = cv2.INTER_NEAREST)
    else:
        Eys_new = cv2.resize(Ey, (new_size, new_size), interpolation = cv2.INTER_LINEAR)
    return Eys_new

def h2index(h, dh):
    if isinstance(h, np.ndarray):
        h = (np.round(h/dh)).astype(int)
    else:
        h = int(round(h/dh))
    return h

def gen_decay_rate(total_steps, decay_steps):
    expected_lr_decay = 0.1
    decay_rate = np.exp(np.log(expected_lr_decay)/(total_steps/decay_steps))
    print(f"decay_rate: {decay_rate:2f}")
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

class SimpleDataset(Dataset):
    def __init__(self,X, Y, transform = None):
        self.X = X 
        self.Y = Y
        self.transform = transform
        
    def __getitem__(self,index): 
        sample = {'X':self.X[index], 'Y': self.Y[index]}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.X.shape[0]

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        X, Y = sample['X'], sample['Y']
        return {'X': torch.tensor(X, dtype = torch.float),
                'Y': torch.tensor(Y, dtype = torch.float)}
        
def train(model, X, Y, epochs, lr, batch_size):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("using device: ", device)
    log_epochs = int(epochs // 10)
    model = model.to(device)
    dataset_train = SimpleDataset(X, Y, ToTensor())
    dataloader_train = DataLoader(dataset_train,
                            shuffle=True,
                            batch_size= batch_size)
    dataloader_test = DataLoader(dataset_train,
                            shuffle=False,
                            batch_size= batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    decay_rate = gen_decay_rate(epochs, log_epochs)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    mse = torch.nn.MSELoss(reduction = 'sum')
    for epoch in range(epochs):
        train_epoch(dataloader_train, model, mse, optimizer, device)
        if epoch % log_epochs == 0 or (epoch == epochs - 1):
            if epoch != 0:
                my_lr_scheduler.step()
            Y_pred = test(dataloader_test, model, device)
            Y_pred = np.array(Y_pred)
            #Y can be zeros. So we take sum() first. otherwise, you will divide zero.
            relative_error = np.mean(np.abs(Y_pred - Y).sum()/np.abs(Y).sum()) * 100
            print(f"total epoches:{epochs:5d} [curr:{epoch:5d} relative_error:{round(relative_error, 3):5f}%].")
            if relative_error < 0.1: 
                print("fitting error < 0.1%, accurate enough, stoped.")
                break
    if relative_error > 0.1:
        print("fitting error > 0.1%, increase total steps or number of layers in fullconnected network.(note: < 1% is good enough, but < 0.1% is the safest.)")
    return Y_pred

def train_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for _, samples in enumerate(dataloader):
        X, Y = samples['X'].to(device), samples['Y'].to(device)
        # Compute prediction error
        Y_pred = model(X)
        loss = loss_fn(Y_pred, Y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return None

def test(dataloader, model, device):
    model.eval()
    Y_pred = []
    with torch.no_grad():
        for samples in dataloader:
            X, _ = samples['X'].to(device), samples['Y'].to(device)
            pred = model(X)
            Y_pred = Y_pred + pred.cpu().tolist()
    return Y_pred