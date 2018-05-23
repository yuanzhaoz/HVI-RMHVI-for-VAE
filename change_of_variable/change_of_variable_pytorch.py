import autograd.numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from torch.autograd.gradcheck import zero_gradients

def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph = True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)


def reparameter_pdf(transform, x_pdf, y):
    '''
    transform: function x = g(y) that transforms random variable y into x
    x_pdf: pdf of x
    '''
    # get gradient g'(y)
    #y.grad.zero_()
    #yy = torch.tensor([y.item()], requires_grad=True)
    #y.retain_grad()
    yy = y.clone()
    yy.retain_grad()
    x = transform(yy)
    print("x "+str(x))
    num = x.shape[0]
    
    # one dimensional 
    if num == 1:
        x.backward(retain_graph=True)
        print("y grad "+str(yy.grad))
        grad = torch.abs(yy.grad).float()
        px = x_pdf(x)
        print("px "+str(px))
        result = px*grad
        yy.grad.data.zero_()
        print("result "+str(result))
        #print(px.dtype)
        return result
    # multivariate case
    else:
        #x_new = torch.zeros(1, x.shape[0], requires_grad=True)
        yy_new = torch.zeros(1, yy.shape[0], requires_grad=True)
        #x_new.data[0] = x.data
        yy_new.data[0] = yy.data
        x_new = transform(yy_new)
        j = compute_jacobian(yy_new, x_new)
        determinant = torch.det(j[0])
        print(determinant)
        # get p(x)
        px = x_pdf(x)
        print("px "+str(px))
        result = px*determinant
        yy_new.grad.data.zero_()
        print("result "+str(result))
        #print(px.dtype)
        return result
        
    
    """
    tmp = torch.ones(num)
    x.backward(tmp, retain_graph=True)
    print("y grad "+str(yy.grad))
    grad = torch.abs(yy.grad).float()
    ones = torch.ones(num)
    grad_ones = grad/grad[0].item()
    decide = torch.equal(ones, grad_ones)
    if decide == False:
        raise ValueError('Gradient elements are not the same')

    grad_new = torch.tensor([grad[0].item()])
    """
    """
    # get p_x(x)
    px = x_pdf(x)
    print("px "+str(px))
    result = px*grad_new
    yy.grad.data.zero_()
    print("result "+str(result))
    #print(px.dtype)
    return result
    """
