import torch
from torch.autograd import grad


def sgdm(loss, xv, T=200, lr=0.1, gamma=0.9):
    """ SGD plus momentum. Instead of calling .backward in pytorch, we explicitly call 
    torch.autograd.grad to make it closer to the notation in the paper, and simplifly
    writing the reversed system.
    """
    x, v = xv[0], xv[1]

    for i in range(T):
        v = gamma*v - (1-gamma)*grad(loss(x), x)[0]
        x = x + lr*v

    xv = (x, v)

    def mdgs(loss=loss, xv=xv, T=T, lr=lr, gamma=gamma):
        """ Reversed dynamics. We define it as a local function to be able to save the arguments T, lr, ...
            from the primal function without having to define a new class to save the state etc
        """
        x, v = xv[0], xv[1]
        for i in range(T):
            x = x - lr*v
            v = (v + (1-gamma)*grad(loss(x), x)[0])/gamma

        return (x, v)

    return xv, mdgs

def fgsmm(loss, xv, T=200, lr=0.1, gamma=0.9):
    """ FGSM plus momentum. Instead of calling .backward in pytorch, we explicitly call 
    torch.autograd.grad to make it closer to the notation in the paper, and simplifly
    writing the reversed system.
    """
    x, v = xv[0], xv[1]

    for i in range(T):
        v = gamma*v - (1-gamma)*torch.sign(grad(loss(x), x)[0])
        x = x + lr*v

    xv = (x, v)

    def mmsgf(loss=loss, xv=xv, T=T, lr=lr, gamma=gamma):
        """ Reversed dynamics. We define it as a local function to be able to save the arguments T, lr, ...
            from the primal function without having to define a new class to save the state etc
        """
        x, v = xv[0], xv[1]
        for i in range(T):
            x = x + lr*v
            #v = (v + (1-gamma)*torch.sign(grad(loss(x), x)[0]))/gamma
            v = gamma*v - (1-gamma)*torch.sign(grad(loss(x), x)[0])

        return (x, v)

    return xv, mmsgf
