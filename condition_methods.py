from abc import ABC, abstractmethod
import torch

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    

@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0) 

    def conditioning(self, x_i, x_t_minus_one, x_0_hat, measurement, **kwargs):
        """
        The conditioning function as shown in line 7
        
        Args:
            x_i: torch.Tensor, x_i
            x_t, torch.Tensor, x_t_minus_1 prime
            x_0_hat: torch.Tensor, predicted x_0
            measurement: torch.Tensor, y, the corrputed image
        """
        # norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        ############    Start TODO  ###########
        ###### Implentment the conditional sampling in line 7 #######
        ###### A(x_0_hat) is already provided to you as A ########
        ###### Also torch.autograd.grad() is provided to you to calculate the gredient of the 
        ###### norm term with respect to x_i, you can check https://pytorch.org/docs/stable/generated/torch.autograd.grad.html#torch.autograd.grad
        ###### for its detailed usage. You only need to specify the outputs and inputs here.
        # A = self.operator.forward(x_0_hat, **kwargs)
        A = self.operator.forward(x_0_hat,**kwargs)
         
        difference = measurement - A
        norm = torch.sqrt(torch.sum(difference ** 2))
        diff_output = norm  # outputs of the differentiated function
        diff_input = x_i   # Inputs w.r.t. which the gradient will be returned
        
        ## TODO: Don't delete this line, you will use this
        norm_grad = torch.autograd.grad(outputs=diff_output, inputs=diff_input)[0]
        new_x_t_minus_one = x_t_minus_one - norm_grad
        ############    END TODO  ###########
        return new_x_t_minus_one