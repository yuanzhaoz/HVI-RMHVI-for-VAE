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

############################################################3
def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def log_prior(z):
    dim = z.shape[1]
    mean = torch.zeros(dim).cuda()
    cov = torch.eye(dim).cuda()
    m = MultivariateNormal(mean, cov)
    m.requires_grad=True
    return m.log_prob(z)

def multivariate_normal_logpdf(mean, cov, x):
    mean = mean.cuda()
    cov = cov.cuda()
    k = x.shape[0]
    t1 = -0.5*(x - mean).view(1, k)@torch.inverse(cov)@(x - mean).view(k, 1)
    t2 = 0.5*k*torch.log(2*torch.tensor([math.pi]).cuda()) + 0.5*torch.log(torch.det(cov))
    return t1 - t2

def multivariate_normal_diagonal_logpdf(mean, cov_diag, x):
    mean = mean.cuda()
    cov_diag = cov_diag.cuda()
    n = x.shape[0] # number of samples
    k = x.shape[1] # dimension
    t1 = -0.5*(x - mean)*(1/cov_diag)*(x-mean)
    t1 = torch.sum(t1, dim=1)
    #t2 = 0.5*k*torch.log(2*torch.tensor([math.pi]).cuda()) + 0.5*torch.log(torch.prod(cov_diag,1)).cuda()
    t2 = 0.5*k*torch.log(2*torch.tensor([math.pi]).cuda()) + 0.5*torch.sum(torch.log(cov_diag)).cuda()
    #print("t1: "+str(t1)+"t2: "+str(t2))
    return t1 - t2

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
        #self.fc2 = nn.Linear(300, 300)
        self.fc31 = nn.Linear(300, z_dim)
        self.fc32 = nn.Linear(300, z_dim)
    def forward(self, x):
        x = x.view(-1, 784)
        h1 = F.tanh(self.fc1(x))
        #h2 = F.softplus(self.fc2(h1))
        logvar = self.fc31(h1)
        mu = self.fc32(h1)
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

    
decoder = decoder().to(device)
q_z0 = q_z0().to(device)
r_v = r_v().to(device)
q_v = q_v().to(device)
#mass_diag = torch.randn(z_dim, requires_grad=True)
log_mass_diag = torch.randn(z_dim, requires_grad=True)
q_z0_mean = torch.randn(z_dim, requires_grad=True) 
q_z0_logvar = torch.randn(z_dim, requires_grad=True)


def lower_bound(decoder, q_z0, r_v, data, T):
    batch_size = data.view(-1, 784).shape[0]
    data = data.to(device)
    
    
    #mu_z0, logvar_z0 = q_z0(data)
    #var_z0 = torch.exp(logvar_z0)
    #print(mu_z0.shape)
    #print("logvar z0:"+str(logvar_z0))

    # sample z0
    logvar_z0 = torch.zeros([batch_size, z_dim], requires_grad = True).cuda()
    var_z0 = torch.exp(logvar_z0)
    mu_z0 = torch.zeros([logvar_z0.shape[0], logvar_z0.shape[1]],requires_grad = True).cuda()
    
    z0 = reparameterize(mu_z0, logvar_z0)
    #print("z0: " + str(z0.shape))
    #print(z0)
    
    
    

    # get joint probaility p(x, z0)
    log_prior_z0 = log_prior(z0)
    #print("log_prior_z0: " + str(log_prior_z0.shape))
    decoder_output = decoder(z0)
    
    log_likelihood = 0. - F.binary_cross_entropy(decoder_output, data.view(-1, 784).float(), size_average=False, reduce=False)
    #print("log_likelihood: " + str(log_likelihood.shape))
    log_likelihood = torch.sum(log_likelihood, dim = 1)
    #print("log_likelihood: " + str(log_likelihood.shape))
    log_joint = log_prior_z0 + log_likelihood
    #print("log_joint: " + str(log_joint.shape))

    # get log q_z0
    log_q_z0 = multivariate_normal_diagonal_logpdf(mu_z0, var_z0, z0)
    #print("log_q_z0: "+str(log_q_z0.shape))

    # initial L for 128 samples
    #L = log_joint - log_q_z0.view(batch_size)
    
    L = log_joint - log_q_z0
    #print("L: "+str(L.shape))
    #print("log_joint: "+str(log_joint))
    #print("log_q_z0: "+str(log_q_z0))
    #L = torch.sum(L)
    print("initial L "+str(L))
    #print(L.shape)

    #print("====================================")
    for i in range(T):
        # sample v1
        #logvar_v1 = q_v(data)
        logvar_v1 = torch.zeros([batch_size, z_dim], requires_grad = True).cuda()
        mu_v1 = torch.zeros([logvar_v1.shape[0], logvar_v1.shape[1]],requires_grad = True).cuda()
        v1 = reparameterize(mu_v1, logvar_v1)
        
        
        var_v1 = torch.exp(logvar_v1)
        
        
        mass_diag = 1/var_v1
        #print("mass_diag: "+str(mass_diag.shape))
        #mass_matrix = torch.diag(mass_diag)
                
        #print("mu_v1: "+str(mu_v1.shape))
        #print("var_v1: "+str(var_v1.shape))
        # get q_v1
        log_q_v1 = multivariate_normal_diagonal_logpdf(mu_v1, var_v1 ,v1)
        
        
        

        log_joint_t = torch.zeros(0).cuda() # list of all the joint
        log_r_vt = torch.zeros(0).cuda()
        alpha = torch.tensor([0.]).cuda() # lower bound for each batch (128 samples)
        for j in range(batch_size):
            def energy_function(z, cache):
                z.retain_grad()
                z = z.view(1, z.shape[0])
                z = z.cuda()
                one_log_prior = log_prior(z)
                decoder_output = decoder(z)
                
                one_log_likelihood = 0. - F.binary_cross_entropy(decoder_output, data.view(-1, 784)[j].float(), size_average=False, reduce=False)
                #print(one_log_likelihood.shape)
                one_log_likelihood = torch.sum(one_log_likelihood, dim = 1)
                one_log_joint = one_log_prior + one_log_likelihood
                return 0 - one_log_joint
            sampler = IsotropicHmcSampler(energy_function, energy_grad=None, prng=None,
                                          mom_resample_coeff=1., dtype=np.float64)
            init = torch.zeros(z_dim).cuda()
            
            mass_matrix = torch.diag(mass_diag[j])
            #mass_matrix.cuda()
            
            #print("mass: "+str(mass_matrix.is_cuda))
            #print("v1: "+str(v1.is_cuda))
            #print("init: "+str(init))
            pos_samples, mom_samples, ratio = sampler.get_samples(init, 0.05, 5, 2, mass_matrix, mom = v1[j].view(z_dim))
            #pos_samples, mom_samples, ratio = sampler.get_samples(init, 1e-10, 1, 2, mass_matrix, mom = v1[j].view(z_dim))
            #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")
            #print(pos_samples[1].shape)

            # get joint probaility p(x, zt)
            zt = pos_samples[1].cuda()
            vt = mom_samples[1].cuda()
            zt = zt.view(1, zt.shape[0])
            vt = vt.view(vt.shape[0])

            # get joint probaility p(x, zt)
            one_log_prior_zt = log_prior(zt)
            #print("one_log_prior_zt: " + str(one_log_prior_zt.shape))
            one_decoder_output_t = decoder(zt)
            #print("one_decoder_output_t: " + str(one_decoder_output_t.shape))
            
           
            one_log_likelihood_t = 0. - F.binary_cross_entropy(one_decoder_output_t, data.view(-1, 784)[j].float(), size_average=False, reduce=False)
            one_log_likelihood_t = torch.sum(one_log_likelihood_t, dim = 1)
            #print("one_log_likelihood_t: " + str(one_log_likelihood_t.shape))
            one_log_joint_t = one_log_prior_zt + one_log_likelihood_t
            #print("one_log_joint_t: " + str(one_log_joint_t.shape))
            log_joint_t = torch.cat((log_joint_t, one_log_joint_t), 0)

            # get r_vt
            d = data.view(-1, 784)[j].view(1, 784)
            one_new_data = torch.cat((d.float(), zt), 1) # append data with zt
            one_mu_vt, one_logvar_vt = r_v(one_new_data)
            #print("one_mu_vt 1: " + str(one_mu_vt.shape))
            one_var_vt = torch.exp(one_logvar_vt)
            
            one_mu_vt = one_mu_vt.view(one_mu_vt.shape[1])
            one_cov = torch.diag(one_var_vt.view(one_var_vt.shape[1]))
            
            #print("one_mu_vt: " + str(one_mu_vt.shape))
            #print("one_logvar_vt: " + str(one_logvar_vt.shape))
            #m = MultivariateNormal(one_mu_vt, one_cov)
            #one_log_r_vt = m.log_prob(vt).view(1)
            
            one_log_r_vt = multivariate_normal_logpdf(one_mu_vt, one_cov, vt).view(1)
            #one_log_r_vt = multivariate_normal_diagonal_logpdf(one_mu_vt, one_cov, vt)
            #print("one_log_r_vt "+str(one_log_r_vt))
            log_r_vt = torch.cat((log_r_vt, one_log_r_vt), 0)
            #print("log_r_vt "+str(log_r_vt.shape))
            

            # get L for each sample
            one_log_alpha = log_joint_t[j] + log_r_vt[j] - log_joint[j] - log_q_v1[j]
            
            print("~~~~~~~~~~~`")
            print(log_joint_t[j])
            print(log_r_vt[j])
            print(log_joint[j])
            print(log_q_v1[j])
              
            
            #print("one log alpha: "+str(one_log_alpha))
            #one_log_alpha = torch.log(one_alpha)
            L[j] = L[j] + one_log_alpha
            #print("L: "+str(L))
            #alpha = alpha + one_alpha
        #L = L + torch.log(alpha)
    #print("final L: "+str(L.shape))
    #print("~~~~~~~~~~~~~~~~~~~ new L " + str(L) + " ~~~~~~~~~~~~~~~~~~~")
    return torch.sum(L)/batch_size    
                    
                
    
# Train
params1 = list(decoder.parameters())+list(r_v.parameters())#+list(q_z0.parameters())
optimizer1 = optim.Adam(params1, lr=0.0005, weight_decay=5e-5)
#optimizer2 = optim.Adam([log_mass_diag], lr=0.0005, weight_decay=1e-4)
#optimizer2 = optim.Adam(q_z0.parameters(), lr=0.00001, weight_decay=1e-3)

for epoch in range(10):
    print("Epoch: "+str(epoch+1))
    file = open("result5_"+str(epoch)+".txt","w")
    file_test = open("result5_test_"+str(epoch)+".txt","w")
    for i in range(len(result)):
        print("++++++++++ batch: " + str(i) + " ++++++++++")

        data = result[i].float()
        optimizer1.zero_grad()
        #optimizer2.zero_grad()
        #optimizer3.zero_grad()
        L = lower_bound(decoder, q_z0, r_v, data, 1)
        loss = 0. - L
        loss.backward()
        
        #nn.utils.clip_grad_norm_(q_v.parameters(), 0.5)
        nn.utils.clip_grad_norm_(q_z0.parameters(), 1)
        nn.utils.clip_grad_norm_(decoder.parameters(), 1)
        nn.utils.clip_grad_norm_(r_v.parameters(), 1)
        
        print('weight grad after backward')
        #print(net.conv1.bias.grad)
        #print(q_z0.fc1.bias.grad)
        #print(q_z0.fc31.bias.grad)
        #print(q_z0.fc32.bias.grad)
        optimizer1.step()
        #optimizer2.step()
        #optimizer3.step()
        file.write(str(0.-L.item())+"\n") 
        print(L.item())
    file.close()
    for i in range(len(result_test)):
        print("++++++++++ test batch: " + str(i) + " ++++++++++")
        data = result_test[i].float()
        L = lower_bound(decoder, q_z0, r_v, data, 1)
        file_test.write(str(0.-L.item())+"\n")
        print(L.item())
    file_test.close()
    
    sample = torch.randn(64, 20).to(device)
    sample = decoder(sample).cpu()
    save_image(sample.view(64, 1, 28, 28), 'sample5_' + str(epoch) + '.png')
        
    
"""
    for batch_idx, (data, _) in enumerate(train_loader):
        print("++++++++++ " + str(batch_idx) + " ++++++++++")
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        L = lower_bound(decoder, q_z0_mean, q_z0_logvar, r_v, data, log_mass_diag, 1)
        loss = 0. - L
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        print(L.item())
        #file.write(str(L)+"\n") 
    print(L.item())
    
#file.close()
"""

