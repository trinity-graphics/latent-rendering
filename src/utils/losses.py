import drjit as dr
import lpips
import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0,stagnation_eps=1e-4, opt=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.stagnation_eps = stagnation_eps
        self.min_validation_loss = float('inf')
        self.opt_min = opt
    
    def set_opt(self, opt):
        self.opt_min = opt

    def update(self, validation_loss, opt):
        diverging = False
        stagnating = False
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.opt_min = opt
        if validation_loss > (self.min_validation_loss + self.min_delta):
            diverging = True
        # if dr.abs(validation_loss - self.min_validation_loss) > self.stagnation_eps:
        #     stagnating = True
        
        if diverging or stagnating:
            self.counter += 1
        
        kill = self.counter > self.patience
        
        return kill

def dr_huber(true, test, delta=1.0):
    a = test - true
    abs_a = dr.abs(a)
    quadratic = 0.5 * dr.square(a)
    linear = delta * (abs_a - 0.5 * delta)
    loss = dr.select(abs_a <= delta, quadratic, linear)
    return dr.mean(loss)

def dr_mse(true, test):
    a = test - true
    loss = dr.square(a)
    return dr.mean(loss)

def huber(true, test, delta=1.0):
    a = test - true
    abs_a = torch.abs(a)
    quadratic = 0.5 * torch.square(a)
    linear = delta * (abs_a - 0.5 * delta)
    loss = torch.where(abs_a <= delta, quadratic, linear)
    return loss.mean()

def mse(true, test):
    a = test - true
    loss = torch.square(a)
    return loss.mean()

def gaussian_regularizer(test, threshold = 1e-4):
    x_threshold = dr.sqrt(-2 * dr.log(threshold * dr.sqrt(2 * dr.pi)))
    abs_test = dr.abs(test)
    zero = dr.zeros_like(test)
    quadratic = dr.square(abs_test - x_threshold)
    loss = dr.select(abs_test <= x_threshold, zero, quadratic)
    return loss

def power_regularizer(test, p=6, scale_factor=1):
    abs_test = dr.abs(test) / scale_factor
    loss = dr.power(abs_test, p)
    return loss

def hard_regularization(train_images, test_min=-3, test_max=3, lambda_range=1000.0, power=6):
    below = F.relu(test_min - train_images)
    above = F.relu(train_images - test_max)

    penalty = (below ** power + above ** power).mean()

    return lambda_range * penalty


kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
def kldiv(test: torch.tensor):
    test = torch.nn.functional.log_softmax(test, dim=1)
    target = torch.randn_like(test, dtype=test.dtype, device=test.device)
    return kl_loss(test, target)

def ssimLoss(true, test):
    r"""
    Converts an sigmoid latent into latent range

    Args:
        test (`mi.TensorXf`):
            The generated latent image.
        true (`mi.TensorXf`):
            The reference latent image.

    Returns:
        loss:
            The SSIM loss between the two latents.
    """
    # Generate latent distribution
    
    true_sigmoid = torch.sigmoid(true)
    test_sigmoid = torch.sigmoid(test)
    ssimLoss = 1.0 - fused_ssim(true_sigmoid, test_sigmoid)
    return ssimLoss

lpips_fn = lpips.LPIPS(net='alex').cuda()
def lpipsLoss(true, test):
    true = torch.as_tensor(true, device='cuda')
    test = torch.as_tensor(test, device='cuda')
    true = true * 2 - 1
    test = test * 2 - 1
    loss = lpips_fn.forward(test, true)
    return loss.item()

# def kldiv(test):
#     mu = dr.mean(test)
#     dr.eval(mu)
#     sigma = dr.sqrt(dr.mean(dr.power((test - mu), 2)))
#     kl = dr.log(1 / sigma) + (dr.power(sigma, 2) + dr.power(mu, 2) - 1) / 2
#     return kl