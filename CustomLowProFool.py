
# imports
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


# Clipping function
def super_clip(current):
  current[current<0] = 0
  current[current>1] = 1

  return current

def lowProFoolWithAccessibilityConstraint(x, model, ac, maxiters, alpha, omega):
    """
    Generates an adversarial examples x' from an original sample x

    :param x: input data (preprocessed user-resource metadata)
    :param model: neural network
    :param ac: accessiblity constraint
    :param maxiters: maximum number of iterations ran to generate the adversarial examples
    :param alpha: scaling factor used to control the growth of the perturbation
    :param omega: trade off factor between fooling the classifier and generating imperceptible adversarial example
    :return: original label prediction, final label prediction, adversarial examples x'
    """

    r = Variable(torch.FloatTensor(1e-7 * np.ones(x.numpy().shape)), requires_grad=True) 
    r = Variable(r, requires_grad=True)
    v = torch.FloatTensor(np.array(ac))

    with torch.no_grad():
      outputs = model((x+r).double())
    outputs = torch.sigmoid(outputs)

    actual_ouput_prob = outputs
    preds = outputs.tolist()[0][0]
    preds = 1.0 if preds > 0.5 else 0.0

    target_pred = np.abs(1 - preds)
    target_list = [1.] if target_pred == 1 else [0.]
    target = np.array(target_list)

    target = Variable(torch.tensor(target, requires_grad=False))
    target = target.double()
    target = target.unsqueeze(1)
    
    omega = torch.tensor([omega])
    criterion = torch.nn.BCELoss()
    
    l1 = lambda v, r: torch.sum(torch.abs(v * r)) #L1 norm
    
    best_norm_weighted = np.inf
    best_pert_x = x
    
    loop_i, loop_change_class = 0, 0

    while loop_i < maxiters:
        if r.grad is not None:
            r.grad.zero_()
        
        # Computing loss 
        loss_1 = criterion(outputs, target)

        loss_2 = l1(v, r)
        loss = (loss_1 + omega * loss_2)

        # Get the gradient
        loss.backward(retain_graph=True)
        grad_r = r.grad.data.cpu().numpy().copy()

        # Guide perturbation to the negative of the gradient if target is deny
        if target_pred == 0:
          ri = - grad_r
        else:
          ri = grad_r
    
        # limit huge step
        ri *= alpha

        # Adds new perturbation to total perturbation
        r = r.clone().detach().cpu().numpy() + ri
        
        # For later computation
        r_norm_weighted = np.sum(np.abs(r * ac))
        
        # Ready to feed the model
        r = Variable(torch.FloatTensor(r), requires_grad=True) 
        
        # Compute adversarial example
        xprime = x + r
        
        # Clip to stay in valid range
        xprime = super_clip(xprime)

        with torch.no_grad():
          outputs = model(xprime.double())
        outputs = torch.sigmoid(outputs)

        output_pred = outputs.tolist()[0][0]
        output_pred = 1.0 if output_pred > 0.5 else 0.0
        
        # Keep the best adverse at each iterations
        if output_pred != preds and r_norm_weighted < best_norm_weighted:
            best_norm_weighted = r_norm_weighted
            best_pert_x = xprime
            
        loop_i += 1 

    with torch.no_grad():
      outputs = model(best_pert_x.double())
    outputs = torch.sigmoid(outputs)
    output_pred = outputs.tolist()[0][0]
    output_pred = 1.0 if output_pred > 0.5 else 0.0

    #print('#### outputs:', outputs.tolist(), ' actual prob: ', actual_ouput_prob)

    return preds, output_pred, best_pert_x.clone().detach().cpu().numpy() 


def lowProFoolWithNoAccessibilityConstraint(x, model, maxiters, alpha, omega):
    """
    Generates an adversarial examples x' from an original sample x

    :param x: tabular sample
    :param model: neural network
    :param maxiters: maximum number of iterations ran to generate the adversarial examples
    :param alpha: scaling factor used to control the growth of the perturbation
    :param omega: trade off factor between fooling the classifier and generating imperceptible adversarial example
    :return: original label prediction, final label prediction, adversarial examples x'
    """

    r = Variable(torch.FloatTensor(1e-7 * np.ones(x.numpy().shape)), requires_grad=True) 
    r = Variable(r, requires_grad=True)

    with torch.no_grad():
      outputs = model((x+r).double())
    outputs = torch.sigmoid(outputs)

    actual_ouput_prob = outputs
    preds = outputs.tolist()[0][0]
    preds = 1.0 if preds > 0.5 else 0.0

    target_pred = np.abs(1 - preds)
    target_list = [1.] if target_pred == 1 else [0.]
    target = np.array(target_list)

    target = Variable(torch.tensor(target, requires_grad=False))
    target = target.double()
    target = target.unsqueeze(1)
    
    omega = torch.tensor([omega])
    criterion = torch.nn.BCELoss()
    
    l1 = lambda r: torch.sum(torch.abs(r)) #L1 norm


    best_norm_weighted = np.inf
    best_pert_x = x
    
    loop_i, loop_change_class = 0, 0

    while loop_i < maxiters:
            
        #zero_gradients(r)
        if r.grad is not None:
            r.grad.zero_()
        
        # Computing loss 
        loss_1 = criterion(outputs, target)
        loss_2 = l1(r)
        loss = (loss_1 + omega * loss_2)

        # Get the gradient
        loss.backward(retain_graph=True)
        grad_r = r.grad.data.cpu().numpy().copy()

        # Guide perturbation to the negative of the gradient if target is deny
        if target_pred == 0:
          ri = - grad_r
        else:
          ri = grad_r
    
        # limit huge step
        ri *= alpha

        # Adds new perturbation to total perturbation
        r = r.clone().detach().cpu().numpy() + ri
        
        # Ready to feed the model
        r = Variable(torch.FloatTensor(r), requires_grad=True) 
        
        # Compute adversarial example
        xprime = x + r
        
        # Clip to stay in legitimate bounds
        xprime = super_clip(xprime)

        with torch.no_grad():
          outputs = model(xprime.double())
        outputs = torch.sigmoid(outputs)

        output_pred = outputs.tolist()[0][0]
        output_pred = 1.0 if output_pred > 0.5 else 0.0
        
        # Keep the best adverse at each iterations
        if output_pred != preds:
            best_pert_x = xprime
            
        loop_i += 1 

    with torch.no_grad():
      outputs = model(best_pert_x.double())
    outputs = torch.sigmoid(outputs)
    output_pred = outputs.tolist()[0][0]
    output_pred = 1.0 if output_pred > 0.5 else 0.0

    #print('#### outputs:', outputs.tolist(), ' actual prob: ', actual_ouput_prob)

    return preds, output_pred, best_pert_x.clone().detach().cpu().numpy() 
