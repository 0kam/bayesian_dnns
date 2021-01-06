from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from tensorboardX import SummaryWriter

from tqdm import tqdm

batch_size = 128
epochs = 10
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

labels_per_class = 10
n_labels = 10

root = '../data'
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambd=lambda x: x.view(-1))])

mnist_train = MNIST(root=root, train=True, download=True, transform=transform)
mnist_valid = MNIST(root=root, train=False, transform=transform)


def get_sampler(labels, n=None):
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
weight = [100,1,1,1,1,10,30,4,5,3]
y_train = mnist_train.targets.numpy()
class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
samples_weight = torch.from_numpy(np.array([weight[t] for t in y_train]))
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
unlabelled = torch.utils.data.DataLoader(mnist_train, sampler=sampler, batch_size=batch_size)

validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size,
                                         sampler=get_sampler(mnist_valid.targets.numpy()), **kwargs)

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

# classifier q(y|x)
class Classifier(RelaxedCategorical):
    def __init__(self):
        super(Classifier, self).__init__(var=["y"], cond_var=["x"], name="p")
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

        self.fc11 = nn.Linear(y_dim, z_dim)
        self.fc12 = nn.Linear(y_dim, z_dim)

    def forward(self, y):
        return {"loc": self.fc11(y), "scale": F.softplus(self.fc12(y))}
   
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
q = Inference().to(device)
f = Classifier().to(device)
prior = Prior().to(device)
p_joint = p * prior


# distributions for unsupervised learning
_q_u = q.replace_var(x="x_u", y="y_u")
p_u = p.replace_var(x="x_u")
f_u = f.replace_var(x="x_u", y="y_u")
prior_u = prior.replace_var(y="y_u")

q_u = _q_u * f_u
p_joint_u = p_u * prior_u

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

import pixyz    
import datetime

dt_now = datetime.datetime.now()
exp_time = dt_now.strftime('%Y%m%d_%H:%M:%S')
v = pixyz.__version__
nb_name = 'm2'
writer = SummaryWriter("runs/" + v + "." + nb_name + exp_time)

for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    test_loss, test_accuracy = test(epoch)
    writer.add_scalar('train_loss', train_loss.item(), epoch)
    writer.add_scalar('test_loss', test_loss.item(), epoch)
    writer.add_scalar('test_accuracy', test_accuracy, epoch)    
    
writer.close()