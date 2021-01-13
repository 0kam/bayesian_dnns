# Kingma's M2 model for imbalanced MNIST dataset
# based on https://github.com/masa-su/pixyz/blob/master/examples/m2.ipynb

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from tensorboardX import SummaryWriter

from tqdm import tqdm

batch_size = 512
epochs = 50
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


# sampler weights for each labels(0 to 9)
weight = [100,1,1,1,1,10,7,4,5,3]
# for each labels, sample weight*10 images as labelled train data
labels_per_class = [i * 10 for i in weight]
n_labels = 10

root = '~/data'
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambd=lambda x: x.view(-1))])

mnist_train = MNIST(root=root, train=True, download=True, transform=transform)
mnist_valid = MNIST(root=root, train=False, transform=transform)


def get_sampler(labels, n=None):
    # Only choose digits in n_labels
    (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
    # Ensure uniform distribution of labels
    np.random.shuffle(indices)
    indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n[i]] for i in range(n_labels)])
    indices = torch.from_numpy(indices)
    sampler = SubsetRandomSampler(indices)
    return sampler

def get_sampler_val(labels, n=None):
    # Only choose digits in n_labels
    (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
    # Ensure uniform distribution of labels
    np.random.shuffle(indices)
    indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])
    indices = torch.from_numpy(indices)
    sampler = SubsetRandomSampler(indices)
    return sampler

# Dataloaders for MNIST
kwargs = {'num_workers': 1, 'pin_memory': True}
labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                       sampler=get_sampler(mnist_train.targets.numpy(), labels_per_class),
                                       **kwargs)

from torch.utils.data.sampler import WeightedRandomSampler

y_train = mnist_train.targets.numpy()
class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
samples_weight = torch.from_numpy(np.array([weight[t] for t in y_train]))
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
unlabelled = torch.utils.data.DataLoader(mnist_train, sampler=sampler, batch_size=batch_size)
validation = torch.utils.data.DataLoader(mnist_valid, sampler=get_sampler_val(mnist_valid.targets.numpy()),
                                           batch_size=batch_size, **kwargs)

from pixyz.distributions import Normal, Bernoulli, RelaxedCategorical, Categorical
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

    
# generative model p(x|z,y)    
class Generator(Bernoulli):
    def __init__(self):
        super().__init__(var=["x"], cond_var=["z","y"], name="p")

        self.fc1 = nn.Linear(z_dim+y_dim, 512)
        self.fc2 = nn.Linear(512, x_dim)

    def forward(self, z, y):
        h = F.relu(self.fc1(torch.cat([z, y], 1)))
        return {"probs": torch.sigmoid(self.fc2(h))}


# classifier p(y|x)
class Classifier(RelaxedCategorical):
    def __init__(self):
        super(Classifier, self).__init__(var=["y"], cond_var=["x"], name="p")
        self.fc1 = nn.Linear(x_dim, 512)
        self.fc2 = nn.Linear(512, y_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.softmax(self.fc2(h), dim=1)
        return {"probs": h}


# prior model p(z)
prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
               var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

# distributions for supervised learning
p = Generator().to(device)
q = Inference().to(device)
f = Classifier().to(device)
p_joint = p * prior


# distributions for unsupervised learning
_q_u = q.replace_var(x="x_u", y="y_u")
p_u = p.replace_var(x="x_u", y="y_u")
f_u = f.replace_var(x="x_u", y="y_u")

q_u = _q_u * f_u
p_joint_u = p_u * prior

p_joint_u.to(device)
q_u.to(device)
f_u.to(device)

print(p_joint_u)
print_latex(p_joint_u)

elbo_u = ELBO(p_joint_u, q_u)
elbo = ELBO(p_joint, q)
nll = -f.log_prob() # or -LogProb(f)

rate = 1 * (len(unlabelled) + len(labelled)) / len(labelled)

loss_cls = -elbo_u.mean() -elbo.mean() + (rate * nll).mean() 
print(loss_cls)
print_latex(loss_cls)

model = Model(loss_cls,test_loss=nll.mean(),
              distributions=[p, q, f], optimizer=optim.Adam, optimizer_params={"lr":1e-3})
print(model)
print_latex(model)


def train(epoch):
    train_loss = 0
    for x_u, y_u in tqdm(unlabelled):
        x, y = iter(labelled).next()
        x = x.to(device)
        y = torch.eye(10)[y].to(device)
        x_u = x_u.to(device)
        loss = model.train({"x": x, "y": y, "x_u": x_u})
        train_loss += loss
        
    train_loss = train_loss * unlabelled.batch_size / len(unlabelled.dataset)
    print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
    
    return train_loss


def test(epoch):
    test_loss = 0
    correct = 0
    total = 0    
    for x, y in validation:
        x = x.to(device)
        y = torch.eye(10)[y].to(device)        
        loss = model.test({"x": x, "y": y})
        test_loss += loss
        
        pred_y = f.sample_mean({"x": x})
        total += y.size(0)
        correct += (pred_y.argmax(dim=1) == y.argmax(dim=1)).sum().item()      

    test_loss = test_loss * validation.batch_size / len(validation.dataset)
    test_accuracy = 100 * correct / total
    print('Test loss: {:.4f}, Test accuracy: {:.4f}'.format(test_loss, test_accuracy))
    return test_loss, test_accuracy

def plot_reconstruction(x, y):
    with torch.no_grad():
        z = q.sample({"x":x, "y":y}, return_all=False)
        z.update({"y":y})
        recon_batch = p.sample_mean(z).view(-1,1,28,28)
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
        z = q.sample_mean({"x":x, "y":y}).detach().cpu().numpy()
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

import pixyz    
import datetime

dt_now = datetime.datetime.now()
exp_time = dt_now.strftime('%Y%m%d_%H:%M:%S')
v = pixyz.__version__
nb_name = 'm2_imbalanced_mnist'
writer = SummaryWriter("runs/" + v + "." + nb_name + exp_time)

_x = []
_y = []
for i in range(10):
    _xx, _yy = iter(validation).next()
    _x.append(_xx)
    _y.append(_yy)

_x = torch.cat(_x, dim = 0)
_y = torch.cat(_y, dim = 0)

_x = _x.to(device)
_y = torch.eye(10)[_y].to(device)

for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    test_loss, test_accuracy = test(epoch)
    writer.add_scalar('train_loss', train_loss.item(), epoch)
    writer.add_scalar('test_loss', test_loss.item(), epoch)
    writer.add_scalar('test_accuracy', test_accuracy, epoch)    
    
    recon = plot_reconstruction(_x[:32], _y[:32])
    latent = plot_latent(_x, _y)
    writer.add_images("Image_reconstruction", recon, epoch)
    writer.add_images("Image_latent", latent, epoch)

writer.close()