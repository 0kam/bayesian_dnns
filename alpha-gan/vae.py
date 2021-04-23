from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.nn.functional import binary_cross_entropy
from torch.distributions.kl import kl_divergence
from tensorboardX import SummaryWriter
from torch_optimizer import RAdam

from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import datetime

batch_size = 128
epochs = 10
seed = 1
torch.manual_seed(seed)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# MNIST
root = '~/data'
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambd=lambda x: x.view(-1))])
kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=root, train=True, transform=transform, download=True),
    shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=root, train=False, transform=transform),
    shuffle=False, **kwargs)

from pixyz.distributions import Normal, RelaxedBernoulli

x_dim = 784
z_dim = 4


# inference model q(z|x)
class Inference(Normal):
    """
    parameterizes q(z | x)
    infered z follows a Gaussian distribution with mean 'loc', variance 'scale'
    z ~ N(loc, scale)
    """
    def __init__(self):
        super(Inference, self).__init__(var=["z"], cond_var=["x"], name="q")

        self.fc1 = nn.Linear(x_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, z_dim)
        self.fc32 = nn.Linear(512, z_dim)

    def forward(self, x):
        """
        given the observation x,
        return the mean and variance of the Gaussian distritbution
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

    
# generative model p(x|z)    
class Generator(RelaxedBernoulli):
    """
    parameterizes the bernoulli(for MNIST) observation likelihood p(x | z)
    """
    def __init__(self):
        super(Generator, self).__init__(var=["x"], cond_var=["z"], name="p")

        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, x_dim)

    def forward(self, z):
        """
        given the latent variable z,
        return the probability of Bernoulli distribution
        """
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return {"probs": torch.sigmoid(self.fc3(h))}

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.p = Generator().to(device)
        self.q = Inference().to(device)
        self.prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
               var=["z"], features_shape=[1], name="p_{prior}").to(device)
    
    def forward(self, x):
        z = self.q.sample({"x":x}, reparam=True)["z"]
        z_hat = self.prior.sample(reparam=True, sample_shape=z.shape)["z"].squeeze()
        x_recon = self.p.sample({"z":z}, reparam=True)["x"]
        return z, z_hat, x_recon

def loss_cls(x, x_recon, q, prior):
    kl = kl_divergence(q.dist, prior.dist).sum()
    recon = binary_cross_entropy(x_recon, x, reduction="sum")
    return recon + kl

vae = VAE()

optimizer = RAdam(vae.parameters(), lr=1e-3)

def _train(epoch):
    vae.train()
    running_loss = 0
    for x, y in tqdm(train_loader):
        optimizer.zero_grad()
        x = x.to(torch.float).to(device)
        y = y.to(device)
        z, z_hat, x_recon = vae(x)
        loss = loss_cls(x, x_recon, vae.q, vae.prior)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    print("epoch:", epoch, "loss;", running_loss)
    return running_loss

def plot_reconstrunction(x):
    """
    reconstruct image given input observation x
    """
    with torch.no_grad():
        # infer and sampling z using inference model q `.sample()` method
        z = vae.q.sample({"x": x}, return_all=False)
        
        # reconstruct image from inferred latent variable z using Generator model p `.sample_mean()` method
        recon_batch = vae.p.sample_mean(z).view(-1, 1, 28, 28)
        
        # concatenate original image and reconstructed image for comparison
        comparison = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
        return comparison

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def plot_latent(x, y):
    with torch.no_grad():
        label = torch.argmax(y, dim = 1).detach().cpu().numpy()
        z = vae.q.sample_mean({"x":x}).detach().cpu().numpy()
        N = 10
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(z[:, 0], z[:, 1], c=label, marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
        plt.colorbar(ticks=range(N))
        plt.grid(True)
        fig.canvas.draw()
        image = fig.canvas.renderer._renderer
        image = np.array(image).transpose(2, 0, 1)
        image = np.expand_dims(image, 0)
        return image

dt_now = datetime.datetime.now()
exp_time = dt_now.strftime('%Y%m%d_%H:%M:%S')
nb_name = 'vae'
writer = SummaryWriter("runs/" + nb_name + exp_time)

_x = []
_y = []
it = iter(test_loader)
for i in range(10):
    _xx, _yy = it.next()
    _x.append(_xx)
    _y.append(_yy)

_x = torch.cat(_x, dim = 0)
_y = torch.cat(_y, dim = 0)

_x = _x.to(torch.float).to(device)
_y = torch.eye(10)[_y].to(device)

for epoch in range(50):
    loss = _train(epoch)
    writer.add_scalar('train_loss', loss, epoch)
    recon = plot_reconstrunction(_x[:32])
    latent = plot_latent(_x, _y)
    writer.add_images("Image_reconstruction", recon, epoch)
    writer.add_images("Image_latent", latent, epoch)


writer.close()