# Gaussian Mixtured VAE for imbalanced MNIST dataset
from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from tensorboardX import SummaryWriter

from tqdm import tqdm

batch_size = 512
epochs = 1000
seed = 1
torch.manual_seed(seed)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# https://github.com/wohlert/semi-supervised-pytorch/blob/master/examples/notebooks/datautils.py

from functools import reduce
from operator import __or__
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
import numpy as np

root = '~/data'
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambd=lambda x: x.view(-1))])

mnist_train = MNIST(root=root, train=True, download=True, transform=transform)
mnist_valid = MNIST(root=root, train=False, transform=transform)

from torch.utils.data.sampler import WeightedRandomSampler

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size)

from pixyz.distributions import Normal, Bernoulli, RelaxedCategorical, MixtureModel, Categorical
from pixyz.models import Model
from pixyz.losses import ELBO
from pixyz.utils import print_latex

x_dim = 784
y_dim = 10
z_dim = 64


# inference model q(z|x,y)
class Inference(Normal):
    def __init__(self):
        super().__init__(var=["z"], cond_var=["x","y"], name="q")

        self.fc1 = nn.Linear(x_dim+y_dim, 512)
        self.fc21 = nn.Linear(512, z_dim)
        self.fc22 = nn.Linear(512, z_dim)

    def forward(self, x, y):
        h = F.relu(self.fc1(torch.cat([x, y], 1)))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}

# classifier q(y|x)
class Classifier(RelaxedCategorical):
    def __init__(self):
        super(Classifier, self).__init__(var=["y"], cond_var=["x"], name="q")
        self.fc1 = nn.Linear(x_dim, 512)
        self.fc2 = nn.Linear(512, y_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.softmax(self.fc2(h), dim=1)
        return {"probs": h}

# prior model p(z|y)
class Prior(Normal):
    def __init__(self):
        super().__init__(var=["z"], cond_var=["y"], name="p_{prior}")

        self.fc1 = nn.Linear(y_dim, 64)
        self.fc21 = nn.Linear(64, z_dim)
        self.fc22 = nn.Linear(64, z_dim)
    
    def forward(self, y):
        h = F.relu(self.fc1(y))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}

# generative model p(x|z)    
class Generator(Bernoulli):
    def __init__(self):
        super().__init__(var=["x"], cond_var=["z"], name="p")

        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, x_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return {"probs": torch.sigmoid(self.fc2(h))}

# distributions for supervised learning
p = Generator().to(device)
_q = Inference().to(device)
f = Classifier().to(device)
q = _q * f
prior = Prior().to(device)
p_joint = p * prior

print(p_joint)

from pixyz.losses import KullbackLeibler

elbo = ELBO(p_joint, q)

loss_cls = -elbo.mean()

print(loss_cls)
print_latex(loss_cls)

model = Model(loss_cls,test_loss=loss_cls,
              distributions=[p, _q, f, prior], optimizer=optim.Adam, optimizer_params={"lr":1e-3})
print(model)
print_latex(model)


def train(epoch):
    train_loss = 0
    for x, _y in tqdm(train_loader):
        x = x.to(device)
        loss = model.train({"x": x})
        train_loss += loss
        
    train_loss = train_loss
    print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
    
    return train_loss

from sklearn.metrics.cluster import homogeneity_score

def test(epoch):
    test_loss = 0
    correct = 0
    total = 0    
    hs = []
    for x, y in val_loader:
        x = x.to(device)       
        loss = model.test({"x": x})
        test_loss += loss
        
        y = y.numpy()
        pred_y = f.sample_mean({"x": x}).argmax(1).detach().cpu().numpy()      
        hs.append(homogeneity_score(y, pred_y))
        
    test_loss = test_loss
    test_hs = sum(hs) / len(hs)
    print('Test loss: {:.4f}, Test accuracy: {:.4f}'.format(test_loss, test_hs))
    return test_loss, test_hs

def plot_reconstruction(x):
    with torch.no_grad():
        yz = q.sample({"x":x}, return_all=False)
        recon_batch = p.sample_mean({"z": yz["z"]}).view(-1,1,28,28)
        recon = torch.cat([x.view(-1,1,28,28), recon_batch]).cpu()
        return recon

from matplotlib import pyplot as plt
# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

# borrowed from https://github.com/dragen1860/pytorch-mnist-vae/blob/master/plot_utils.py
def plot_latent(x, y):
    with torch.no_grad():
        label = torch.argmax(y, dim = 1).detach().cpu().numpy()
        _y = f.sample_mean({"x":x})
        z = _q.sample_mean({"x":x, "y":_y}).detach().cpu().numpy()
        N = y_dim
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(z[:, 0], z[:, 1], c=label, marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
        plt.colorbar(ticks=range(N))
        plt.grid(True)
        fig.canvas.draw()
        image = fig.canvas.renderer._renderer
        image = np.array(image).transpose(2, 0, 1)
        image = np.expand_dims(image, 0)
        return image

import pixyz    

nb_name = 'gmvae_unsupervised'
writer = SummaryWriter(nb_name)

_x = []
_y = []
for i in range(10):
    _xx, _yy = iter(val_loader).next()
    _x.append(_xx)
    _y.append(_yy)

_x = torch.cat(_x, dim = 0)
_y = torch.cat(_y, dim = 0)

_x = _x.to(device)
_y = torch.eye(10)[_y].to(device)

for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    test_loss , test_hs = test(epoch)
    writer.add_scalar('train_loss', train_loss.item(), epoch)
    writer.add_scalar('test_loss', test_loss.item(), epoch)
    writer.add_scalar('test_homogenity', test_hs.item(), epoch)
    
    # reconstructed images
    recon = plot_reconstruction(_x[:32])
    latent = plot_latent(_x, _y)
    writer.add_images("Image_reconstruction", recon, epoch)
    writer.add_images("Image_latent", latent, epoch)

writer.close()