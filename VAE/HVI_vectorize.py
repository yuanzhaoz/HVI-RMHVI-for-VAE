import sys
from hmc_base_pytorch import *
from hmc_unconstrained_pytorch import *
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import torch.utils.data
from torch import optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

cuda = True
batch_size = 64
epochs = 10
seed = 1
log_interval = 10
z_dim = 20

# Data preparation
torch.manual_seed(seed)
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_data = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size, shuffle=True, **kwargs)

def binarization(data):
    s = np.random.uniform(size = data.shape)
    out = np.array(s<data).astype(float)
    return out

result = []
for batch_idx, (data, _) in enumerate(train_loader):
    data = data.view(-1, 784).numpy()
    bi_data = binarization(data)
    d = torch.from_numpy(bi_data)
    result.append(d)
    
result_test = []
for batch_idx, (data, _) in enumerate(test_loader):
    data = data.view(-1, 784).numpy()
    bi_data = binarization(data)
    d = torch.from_numpy(bi_data)
    result_test.append(d)
    
###############################################
def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    #return eps.mul(std).add_(mu)
    return eps*std+mu

def log_prior(z):
    dim = z.shape[1]
    mean = torch.zeros(dim).cuda()
    cov = torch.eye(dim).cuda()
    m = MultivariateNormal(mean, cov)
    m.requires_grad=True
    return m.log_prob(z)

def multivariate_normal_diagonal_logpdf(mean, cov_diag, x):
    n = x.shape[0] # number of samples
    k = x.shape[1] # dimension
    t1 = -0.5*(x - mean)*(1/cov_diag)*(x-mean)
    t1 = torch.sum(t1, dim=1)
    t2 = torch.ones(n).cuda()*0.5*k*torch.log(torch.tensor([2*math.pi]).cuda()) + 0.5*torch.sum(torch.log(cov_diag),dim=1)
    return t1 - t2


################################################
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 400)
        self.fc2 = nn.Linear(400, 784)
    # single hidden layer
    def forward(self, x):
        #x = x.view(-1, 784)
        h1 = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(h1))
    
class q_z0(nn.Module):
    def __init__(self):
        super(q_z0, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc31 = nn.Linear(300, z_dim)
        self.fc32 = nn.Linear(300, z_dim)
    def forward(self, x):
        x = x.view(-1, 784)
        h1 = F.softplus(self.fc1(x))
        h2 = F.softplus(self.fc2(h1))
        logvar = self.fc31(h2)
        mu = self.fc32(h2)
        return mu, logvar
    
class r_v(nn.Module):
    def __init__(self):
        super(r_v, self).__init__()
        self.fc1 = nn.Linear(z_dim + 784, 300)
        self.fc21 = nn.Linear(300, z_dim)
        self.fc22 = nn.Linear(300, z_dim)
    def forward(self, x):
        x = x.view(-1, 784 + z_dim)
        h1 = F.softplus(self.fc1(x))
        logvar = self.fc21(h1)
        mu = self.fc22(h1)
        return mu, logvar

class q_v(nn.Module):
    def __init__(self):
        super(q_v, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        # no need to output mu because the mean of momentum is default 0
        self.fc21 = nn.Linear(300, z_dim)
    def forward(self, x):
        x = x.view(-1, 784)
        h1 = F.softplus(self.fc1(x))
        logvar = self.fc21(h1)
        return logvar

################################################
decoder = decoder().to(device)
q_z0 = q_z0().to(device)
r_v = r_v().to(device)
q_v = q_v().to(device)
################################################
def ELBO(data, decoder, q_z0, r_v, q_v, T):#q_z0_mean, q_z0_logvar):
    
    #file1 = open("q_z0_mean"+str(epoch)+".txt","w")
    #file1 = open("q_z0_var"+str(epoch)+".txt","w")
    #file1 = open("z0"+str(epoch)+".txt","w")
    batch_size = data.view(-1, 784).shape[0]
    data = data.to(device)
    
    q_z0_mean, q_z0_logvar = q_z0(data)
    # sample z0
    z0 = reparameterize(q_z0_mean, q_z0_logvar)
    
    # compute q(z0|x)
    var_z0 = torch.exp(q_z0_logvar)
    log_q_z0 = multivariate_normal_diagonal_logpdf(q_z0_mean, var_z0, z0)
    
    #print("q z0 mean: "+str(q_z0_mean))
    #print("q z0 var: "+str(var_z0))
    #print("z0: "+str(z0))
    #print("q(z0|x): "+str(log_q_z0))
    #print("np q(z0|x): "+str(np_log_q_z0))
    
    # compute prior
    log_prior_z0 = log_prior(z0)
    
    
    # compute joint
    decoder_output = decoder(z0)
    log_likelihood = 0. - F.binary_cross_entropy(decoder_output, data.view(-1, 784).float(), size_average=False, reduce=False)
    log_likelihood = torch.sum(log_likelihood, dim = 1)
    log_joint = log_prior_z0 + log_likelihood
    
    
    # initial L
    #L = log_joint - log_q_z0
    L = -log_q_z0
    #print("initial L: "+str(L))
    #print("log likelihood 0: "+str(torch.mean(log_likelihood)))
    
    for i in range(T):
        # sample v1
        
        logvar_v1 = q_v(data)
        
        #logvar_v1 = torch.zeros([batch_size, z_dim], requires_grad = True).cuda()
        var_v1 = torch.exp(logvar_v1)
        mu_v1 = torch.zeros([logvar_v1.shape[0], logvar_v1.shape[1]],requires_grad = True).cuda()
        v1 = reparameterize(mu_v1, logvar_v1)
        #print("v1: "+str(v1.shape))
        
        # compute q(v1|x) 
        log_q_v1 = multivariate_normal_diagonal_logpdf(mu_v1, var_v1 ,v1)
        #print("log_q_v1: "+str(log_q_v1.shape))
        #print("var_v1: "+str(var_v1))
        
        mass_diag = var_v1
        #print("mass_diag: "+str(mass_diag))
        
        def energy_function(z, cache):
            z = z.view(batch_size, z_dim)
            z_cuda = z.cuda()
            all_log_prior = log_prior(z_cuda)
            #print("all prior: "+str(all_log_prior.shape))
            all_log_prior = torch.sum(all_log_prior)
            #print(all_log_prior.shape)
            decoder_output = decoder(z_cuda)

            all_log_likelihood = 0. - F.binary_cross_entropy(decoder_output, data.view(-1, 784).float(), size_average=False)
            #print("one_log_likelihood: "+str(one_log_likelihood.shape))
            all_log_joint = all_log_prior + all_log_likelihood
            return 0 - all_log_joint
        
        
        init = z0.view(batch_size*z_dim)
        mass_diag = mass_diag.view(mass_diag.shape[0]*mass_diag.shape[1])
        mass_matrix = torch.diag(mass_diag)
        mom = v1.view(v1.shape[0]*v1.shape[1])
        #print("init: "+str(init.shape))
        #print("mass: "+str(mass_matrix.shape))
        
        sampler = IsotropicHmcSampler(energy_function, energy_grad=None, prng=None, mom_resample_coeff=0., dtype=np.float64)
        pos_samples, mom_samples, ratio = sampler.get_samples(init, 0.0001, 5, 2, mass_matrix, mom = mom)
        
        zt = pos_samples[1].cuda()
        vt = mom_samples[1].cuda()
        
        zt = zt.view(batch_size, z_dim)
        vt = vt.view(batch_size, z_dim)

        # get joint probaility p(x, zt)
        log_prior_zt = log_prior(zt)
        decoder_output_t = decoder(zt)
        #print("log prior: " + str(log_prior_zt.shape))
        #print("decoder: " + str(decoder_output_t.shape))

        log_likelihood_t = 0. - F.binary_cross_entropy(decoder_output_t, data.view(-1, 784).float(), size_average=False, reduce=False)
        log_likelihood_t = torch.sum(log_likelihood_t, dim = 1)
        #print("log ll: " + str(log_likelihood_t.shape))
        
        log_joint_t = log_prior_zt + log_likelihood_t
        #print("log joint t: " + str(log_joint_t.shape))
        
        # get r(vt|x,zt)
        d = data.view(-1, 784)
        #print(d.shape)
        new_data = torch.cat((d.float(), zt), 1) # append data with zt
        #print("new data: "+str(one_new_data.shape))

        mu_vt, logvar_vt = r_v(new_data)
        var_vt = torch.exp(logvar_vt)
        #print("var_vt: "+str(var_vt.shape))

        log_r_vt = multivariate_normal_diagonal_logpdf(mu_vt, var_vt, vt)

        print("log prior t: "+str(torch.mean(log_prior_zt)))
        print("log likelihood t: "+str(torch.mean(log_likelihood_t)))
        print("joint t: "+str(torch.mean(log_joint_t)))
        print("reverse: "+str(torch.mean(log_r_vt)))
        #print(log_joint[j])
        print("forward: "+str(torch.mean(log_q_v1)))
        print("q(z0|x): "+str(torch.mean(log_q_z0)))
        print("========================================`")
        
        #file1 = open("q_z0_mean"+str(epoch)+".txt","w")
        #file2 = open("q_z0_var"+str(epoch)+".txt","w")
        #file3 = open("z0"+str(epoch)+".txt","w")
        
        # get L for each sample
        #one_log_alpha = one_log_joint_t + one_log_r_vt - log_joint[j] - log_q_v1[j]
        log_alpha = log_joint_t + log_r_vt - log_q_v1
        #one_log_alpha = log_joint[j] #+ one_log_r_vt - log_q_v1[j]

        L = L + log_alpha
        #print("L: "+str(L.shape))
    
    
    return torch.mean(L)

################################################
batch_size = 64
z_dim = 20

params1 = list(decoder.parameters())+list(q_z0.parameters())+list(r_v.parameters())+list(q_v.parameters())
optimizer1 = optim.Adam(params1, lr=0.0035)#, weight_decay=1e-4)
#optimizer1 = optim.Adam([q_z0_mean], lr=0.005)
#optimizer2 = optim.Adam([q_z0_logvar], lr=0.005)


for epoch in range(20):
    print("Epoch: "+str(epoch+1))
    file = open("result11_"+str(epoch)+".txt","w")
    file_test = open("result11_test_"+str(epoch)+".txt","w")
    for i in range(len(result)):
        print("++++++++++"+" Epoch: "+str(epoch+1)+" batch: " + str(i) + " ++++++++++")
        data = result[i].float()
        optimizer1.zero_grad()
        loss = 0. - ELBO(data, decoder, q_z0, r_v, q_v, 1)#_mean_cuda, q_z0_logvar_cuda)
        print("ELBO: "+str(0-loss.item()))
        loss.backward()
        #nn.utils.clip_grad_norm_(q_z0.parameters(), 1)
        #print(q_z0.fc1.weight.grad)
        #print(q_z0.fc31.weight.grad)
        #print(q_z0.fc32.weight.grad)
        optimizer1.step()
        file.write(str(0.-loss.item())+"\n")
    file.close()
    for i in range(len(result_test)):
        print("++++++++++ test batch: " + str(i) + " ++++++++++")
        data = result_test[i].float()
        loss = 0. - ELBO(data, decoder, q_z0, r_v, q_v, 1)
        print("ELBO: "+str(0-loss.item()))
        file_test.write(str(0.-loss.item())+"\n")
        
    file_test.close()
    
    sample = torch.randn(64, 20).to(device)
    sample = decoder(sample).cpu()
    save_image(sample.view(64, 1, 28, 28), 'sample11_' + str(epoch) + '.png')










