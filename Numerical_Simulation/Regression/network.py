import math
import torch
import torch.nn as nn
import torch.nn.functional as F


pi = math.pi
sqpi = math.sqrt(pi)
sq2pi = math.sqrt(2.0 * pi)
sq2 = math.sqrt(2.0)
sq8 = 2.0 * sq2
torch.set_default_dtype(torch.float64)


###########################################
############ GENERAL FUNCTIONS ############
###########################################

activations = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "erf": lambda x: torch.erf(x / sq2),
    "erfxmx3": lambda x: 2 / sqpi * (x / sq2 - x ** 3 / (3 * sq8)),
    "erf+": lambda x: 0.5 * (1 + torch.erf(x / sq2)),
    "x3": lambda x: x ** 3 / 3.0,
    "xmx3": lambda x: x - x ** 3 / 3.0,
    "relu": F.relu,
    "linear": lambda x: x,
    "quadratic": lambda x: x**2,
    "sign":lambda x: torch.sign(x),
    "sepecial_sign": lambda x: torch.erf(g*x)
}


def get_act(act_type,g=50):
    if act_type=="sepecial_sign":
        f = lambda x: torch.erf(g*x)
        return f
    else:
        return activations[act_type]


###########################################
############ COMMITTEE CLASS ##############
###########################################


class Committee(nn.Module):
    def __init__(
        self,
        N = 100,
        K = 10,
        random_v = False,
        mean_field_v = False,
        soft = False,
        nonlinearity = "erf",
        g= 50 ,
        use_bias_W = False,
        use_bias_v = False,
        dtype=torch.float64,
        device=torch.device("cpu"),
        seed=0
    ):

        super(Committee, self).__init__()
        if seed!=None:
            torch.manual_seed(seed) # It allows us to fix the randomness
        self.N = N
        self.K = K
        self.soft = soft
        self.g = get_act(nonlinearity, g)
        self.use_bias_W = use_bias_W
        self.use_bias_v = use_bias_v 

        self.sqN = math.sqrt(N)
        self.sqK = math.sqrt(K)
                
        # init first layer
        self.W = nn.Parameter(torch.randn((N, K), dtype=dtype, device=device))
        self.bias_W = None
        if use_bias_W:
            self.bias_W = nn.Parameter(1e-3 * torch.randn(K, dtype=dtype, device=device))

        # init second layer
        scale_factor = 1.0 / self.sqK if not mean_field_v else 1.0 / K
        init_func_v = torch.randn if random_v else torch.ones
        self.v = init_func_v(K, dtype=dtype, device=device) * scale_factor
        if not soft:
            self.v = nn.Parameter(self.v)
        self.bias_v = None
        if use_bias_v:
            self.bias_v = nn.Parameter(0.01 * torch.tensor(0.0, dtype=dtype, device=device))


    def forward(self, x):
        h = x @ self.W / self.sqN
        if self.use_bias_W:
            h += self.bias_W
        x = self.g(h)
        x = x @ self.v
        if self.use_bias_v:
            x += self.bias_v
        return h, x
                    
                    
    def set_trainable(self, params):
        # reset all grads
        with torch.no_grad():
            for param in self.parameters():
                param.requires_grad = False

        # set trainable weights
        self.W.requires_grad_()
        to_train = [{"params": self.W, "lr": params["lr"]}]

        if params["train_bias"][0]:
            self.bias_W.requires_grad_()
            to_train += [{"params": self.bias_W, "lr": params["lr"]}]

        if not params["soft"] and params["train_bias"][1]:
            self.v.requires_grad_()
            to_train += [{"params": self.v, "lr": params["lr"]}]
        
        if params["train_bias"][1]:
            self.bias_v.requires_grad_()
            to_train += [{"params": self.bias_v, "lr": params["lr"]}]

        return to_train