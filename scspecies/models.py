import numpy as np
import pandas as pd
import muon as mu
import anndata as ad

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal

import time, os, pickle
from typing import Optional, Sequence, Union, Tuple, Dict, List
from datetime import timedelta

from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import chi2
from scipy.sparse import csr_matrix


def create_structure(
        structure: List[int],
        layer_order: list
    ) -> nn.Sequential:
    
    """
    Builds neural network architectures based on a list of layer sizes and operation order.

    Parameters
    ----------
    structure : list of int
        No. of neurons in each layer, including input and output dimensions.  
        For example, [input_dim, hidden1_dim, ..., output_dim]. Must have at least two entries.
    layer_order : list
        Sequence of layer specifications. Each element must be either

        1: 'linear', 
            Affine linear transformation

        2: 'batch_norm', 
            Batch normalization

        3: 'layer_norm', 
            Layer normalization

        4: ('act', activation: nn.Module or 'PReLU', [min_clip, max_clip] optional),
            Activation function. Unbounded activation functions should be clipped for numerical stability, example: ('act', torch.nn.ReLU(), [0, 6])

        5: ('dropout', dropout rate - float in [0, 1]), 
            Dropout layer, example: ('dropout', 0.1)

    Returns
    -------
    nn.Sequential
        A sequential container of PyTorch layers in the specified order for each pair in `structure`.
    """
    layer_operations = [l if type(l) == str else l[0] for l in layer_order]

    if 'dropout' in layer_operations: 
        dr_ind = layer_operations.index('dropout')
        dropout = layer_order[dr_ind][1]

    act_ind = layer_operations.index('act')
    act = layer_order[act_ind][1]

    if len(layer_order[act_ind]) == 3:
        clip_act = layer_order[act_ind][-1]

    else: 
        clip_act = False

    layers = []
    for neurons_in, neurons_out in zip(structure, structure[1:]):
        for operation in layer_operations:
            if operation == 'linear':
                layers.append(nn.Linear(neurons_in, neurons_out))
            elif operation == 'act':
                if act == 'PReLU': act = nn.PReLU(num_parameters=neurons_out)
                else: act = act

                if clip_act != False:
                    layers.append(make_act_bounded(act, min=clip_act[0], max=clip_act[1]))
                else:
                    layers.append(act)                      
            elif operation == 'dropout':
                layers.append(nn.Dropout(dropout))
            elif operation == 'layer_norm':
                layers.append(nn.LayerNorm(neurons_out))
            elif operation == 'batch_norm':
                layers.append(nn.BatchNorm1d(neurons_out))                    
    return nn.Sequential(*layers)

class make_act_bounded(nn.Module):
    """
    Wrapper module that applies an activation and clips its output.

    Parameters
    ----------
    act : nn.Module
        Activation function to apply.
    min : float
        Lower bound for clipping.
    max : float
        Upper bound for clipping.

    """

    def __init__(
            self,
            act: nn.Module,
            min: float,
            max: float
        ):
      
        super().__init__()

        self.act = act         
        self.min = min   
        self.max = max    

    def forward(
            self,
            x: torch.Tensor
        ) -> torch.Tensor:

        x = self.act(x)
        return torch.clamp(x, min=self.min, max=self.max)


class Encoder_outer(nn.Module):
    """
    Outer encoder module that concatenates data and labels, then applies a feed-forward network.
    Will be reinitialized by scSpecies after pre-training on a context dataset.

    Parameters
    ----------
    param_dict : dict
        Dictionary with keys:
        - 'data_dim' (int): Dimensionality of input data.
        - 'label_dim' (int): Dimensionality of input labels.
        - 'dims_enc_outer' (list of int): Hidden layer sizes after concatenation.
        - 'layer_order' (list): See `create_structure` for format.

    Attributes
    ----------
    model : nn.Sequential
        The feed-forward network created by `create_structure`.

    """

    def __init__(
            self, 
            param_dict: dict
        ):
   
        super(Encoder_outer, self).__init__()

        layer_order=param_dict['layer_order'].copy()

        structure = [param_dict['data_dim']+param_dict['label_dim']] + param_dict['dims_enc_outer']
        self.model = create_structure(structure=structure, 
                                      layer_order=layer_order,
                                      )

    def forward(
            self,
            data: torch.Tensor,
            label_inp: torch.Tensor
        ) -> torch.Tensor:

        """
        Forward pass through the outer encoder layers.

        Parameters
        ----------
        data : torch.Tensor
            Input data tensor of shape (batch_size, data_dim).
        label_inp : torch.Tensor
            Input label tensor of shape (batch_size, label_dim).

        Returns
        -------
        torch.Tensor
            Encoded representation of shape (batch_size, dims_enc_outer[-1]).
        """

        x = torch.cat((data, label_inp), dim=-1)
        x = self.model(x)
        return x 

    
class Encoder_inner(nn.Module):
    """
    Inner encoder module producing Gaussian latent parameters and sampling latent variables.
    Will be shared between context and target scVI self.

    Parameters
    ----------
    device : str
        Device identifier for sampling ('cpu', 'mps' or 'cuda').
    param_dict : dict
        Dictionary with keys:
        - 'dims_enc_outer' (list of int): Output dims of outer encoder.
        - 'dims_enc_inner' (list of int): Hidden layer sizes for inner encoder.
        - 'lat_dim' (int): Dimensionality of the latent space.
        - 'layer_order' (list): See `create_structure`.

    Attributes
    ----------
    model : nn.Sequential
        Feed-forward network for intermediate representation.
    mu : nn.Linear
        Linear layer mapping to latent mean.
    log_sig : nn.Linear
        Linear layer mapping to log-standard deviation.
    sampling_dist : Normal
        Standard normal distribution for sampling latent representations.

    """


    def __init__(
            self, 
            device: str,
            param_dict: dict
        ):
        
        super(Encoder_inner, self).__init__()

        structure = [param_dict['dims_enc_outer'][-1]] + param_dict['dims_enc_inner']

        layer_order=param_dict['layer_order'].copy()

        self.model = create_structure(structure=structure, 
                                      layer_order=layer_order,
                                      )

        self.mu = nn.Linear(structure[-1], param_dict['lat_dim'])
        self.log_sig = nn.Linear(structure[-1], param_dict['lat_dim'])

        self.sampling_dist = Normal(
            torch.zeros(torch.Size([param_dict['lat_dim']]), device=torch.device(device)), 
            torch.ones(torch.Size([param_dict['lat_dim']]), device=torch.device(device)))

    def encode(
            self,
            inter: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Compute latent mean and log-std from intermediate representation.

        Parameters
        ----------
        inter : torch.Tensor
            Intermediate features of shape (batch_size, dims_enc_inner[-1]).

        Returns
        -------
        mu : torch.Tensor
            Latent means of shape (batch_size, lat_dim).
        log_sig : torch.Tensor
            Latent log-standard deviations of shape (batch_size, lat_dim).
        """

        x = self.model(inter)
        mu = self.mu(x)
        log_sig = self.log_sig(x)
        return mu, log_sig


    def forward(
            self,
            inter: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Sample latent variable and compute KL-Divergence.

        Parameters
        ----------
        inter : torch.Tensor
            Intermediate features from outer encoder.

        Returns
        -------
        z : torch.Tensor
            Sampled latent tensor of shape (batch_size, lat_dim).
        kl_div : torch.Tensor
            Scalar KL-Divergence across the batch.
        """

        mu, log_sig = self.encode(inter)
        eps = self.sampling_dist.sample(torch.Size([log_sig.size(dim=0)])) 
        kl_div = torch.mean(0.5 * torch.sum(mu.square() + torch.exp(2.0 * log_sig) - 1.0 - 2.0 * log_sig, dim=1))

        z = mu + log_sig.exp() * eps
        return z, kl_div

class Library_encoder(nn.Module):
    """
    Encoder for library size factor, modeling a 1D log-normal distribution.

    Parameters
    ----------
    device : str
        Device identifier for sampling.
    param_dict : dict
        Dictionary with keys:
        - 'data_dim', 'label_dim' (int): Input dims.
        - 'dims_l_enc' (list of int): Hidden layer sizes.
        - 'lib_mu_add' (float): Offset added to the mean.
        - 'layer_order' (list): For `create_structure`.

    Attributes
    ----------
    model : nn.Sequential
        Feed-forward network for concatenated input.
    mu : nn.Linear
        Layer mapping to log-mean of library.
    log_sig : nn.Linear
        Layer mapping to log-std of library.
    sampling_dist : Normal
        Standard normal for sampling.
    mu_add : float
        Added to the decoded mean.

    """

    def __init__(
            self, 
            device,
            param_dict,
            ):

        super(Library_encoder, self).__init__()
        structure = [param_dict['data_dim']+param_dict['label_dim']] + param_dict['dims_l_enc']

        self.model = create_structure(structure=structure, 
                                      layer_order=param_dict['layer_order'],
                                      )        

        self.mu_add = param_dict['lib_mu_add']

        self.mu = nn.Linear(structure[-1], 1)
        self.log_sig = nn.Linear(structure[-1], 1)

        self.sampling_dist = Normal(
            torch.zeros(torch.Size([1]), device=torch.device(device)), 
            torch.ones(torch.Size([1]), device=torch.device(device))) 

    def encode(
            self,
            data: torch.Tensor,
            label_inp: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        """
        Compute library log-mean and log-std from inputs.

        Parameters
        ----------
        data : torch.Tensor
            Data tensor of shape (batch_size, data_dim).
        label_inp : torch.Tensor
            Label tensor of shape (batch_size, label_dim).

        Returns
        -------
        mu : torch.Tensor
            Adjusted log-mean of shape (batch_size, 1).
        log_sig : torch.Tensor
            Log-std of shape (batch_size, 1).
        """

        x = torch.cat((data, label_inp), dim=-1)
        x = self.model(x)
        mu = self.mu(x)
        log_sig = self.log_sig(x)
        return mu + self.mu_add, log_sig 

    def forward(
            self,
            data: torch.Tensor,
            label_inp: torch.Tensor,
            prior_mu: Optional[torch.Tensor] = None,
            prior_sig: Optional[torch.Tensor] = None
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        """
        Sample library factor and compute optional KL-Divergence to prior.

        Parameters
        ----------
        data : torch.Tensor
            Input data.
        label_inp : torch.Tensor
            Input labels.
        prior_mu : torch.Tensor, optional
            Precomputed prior mean parameter for KL-Divergence.
        prior_sig : torch.Tensor, optional
            Precomputed prior std parameter for KL-Divergence.

        Returns
        -------
        l : torch.Tensor
            Sampled library factor of shape (batch_size, 1).
        kl_div : torch.Tensor, optional
            KL-Divergence if prior_mu and prior_sig are not None.
        """

        mu, log_sig = self.encode(data, label_inp)
        eps = self.sampling_dist.sample(torch.Size([log_sig.size(dim=0)]))
        l = torch.exp(mu + log_sig.exp() * eps) 
        
        if prior_mu != None or prior_sig != None:
            kl_div = torch.mean(prior_sig.log() - log_sig.squeeze() + (1 / torch.clamp((2.0 * prior_sig.square()), min=1e-7)) * ((mu.squeeze() - prior_mu) ** 2 + torch.exp(2.0 * log_sig.squeeze()) - prior_sig.square()))
            return l, kl_div
        else:
            return l

class Decoder(nn.Module):
    """
    Decoder mapping latent and label inputs back to data distribution parameters.

    Parameters
    ----------
    param_dict : dict
        Dictionary with keys:
        - 'lat_dim', 'label_dim', 'data_dim' (int): Latent, label, and data dims.
        - 'dims_dec' (list of int): Hidden layer sizes.
        - 'layer_order' (list): For `create_structure`.
        - 'data_distr' (str): 'nb' or 'zinb'.
        - 'dispersion' (str): One of 'dataset', 'batch', 'cell'.
        - 'dispersion' and 'data_distr' control parameter layers.
        - 'homologous_genes' (list of int): Indices for homologous genes.

    Attributes
    ----------
    model : nn.Sequential
        Feed-forward network for decoder.
    rho_pre : nn.Linear
        Linear layer for relative expression logits.
    log_alpha : Parameter or nn.Linear
        Dispersion parameter(s) depending on `dispersion`.
    pi_nlogit : nn.Linear, optional
        Zero-inflation logits if `data_distr == 'zinb'`.

    """

    def __init__(
            self,
            param_dict: dict
        ):

        super(Decoder, self).__init__()

        structure = [param_dict['lat_dim']+param_dict['label_dim']] + param_dict['dims_dec']

        self.data_distr = param_dict['data_distr']
        self.dispersion = param_dict['dispersion']
        self.homologous_genes = np.array(param_dict['homologous_genes'])
        self.non_hom_genes = np.setdiff1d(np.arange(param_dict['data_dim']), self.homologous_genes)
        self.gene_ind = np.argsort(np.concatenate((self.homologous_genes, self.non_hom_genes)))
        self.data_dim = param_dict['data_dim']

        if self.data_distr not in ['zinb', 'nb']:
            raise ValueError(f"data_distr must be a list containing these strings: {'zinb', 'nb'}")        

        if self.dispersion not in ['dataset', 'batch', 'cell']:
            raise ValueError(f"dispersion must be a list containing these strings: {'dataset', 'batch', 'cell'}")     

        self.model = create_structure(structure=structure, 
                                      layer_order=param_dict['layer_order'],
                                      )   
          
        self.rho_pre = nn.Linear(structure[-1], self.data_dim)
        
        if self.dispersion == "dataset":
            self.log_alpha = torch.nn.parameter.Parameter(data=torch.randn(self.data_dim)*0.1, requires_grad=True)
        elif self.dispersion == "batch":
            self.log_alpha = torch.nn.parameter.Parameter(data=torch.randn((param_dict['label_dim'], self.data_dim))*0.1, requires_grad=True)    
        elif self.dispersion == "cell":
            self.log_alpha = nn.Linear(structure[-1], self.data_dim)

        if self.data_distr == 'zinb':
            self.pi_nlogit = nn.Linear(structure[-1], self.data_dim)    

    def calc_nlog_likelihood(
            self,
            dec_outp: List[torch.Tensor],
            library: torch.Tensor,
            x: torch.Tensor,
            eps: float = 1e-7
        ) -> torch.Tensor: 

        """
        Compute negative log-likelihood under NB or ZINB self.

        Parameters
        ----------
        dec_outp : list of torch.Tensor
            [alpha, rho] or [alpha, rho, pi_nlogit] depending on distribution.
        library : torch.Tensor
            Library size factor.
        x : torch.Tensor
            Observed count data.
        eps : float
            Numerical stability constant.

        Returns
        -------
        torch.Tensor
            Negative log-likelihood per sample.
        """

        if self.data_distr == 'nb':
            alpha, rho = dec_outp 
            alpha = torch.clamp(alpha, min=eps)
            rho = torch.clamp(rho, min=1e-8, max=1-eps)
            mu = rho * library
            p = torch.clamp(mu / (mu + alpha), min=eps, max=1-eps)            
            log_likelihood = x * torch.log(p) + alpha * torch.log(1.0 - p) - torch.lgamma(alpha) - torch.lgamma(1.0 + x) + torch.lgamma(x + alpha)   

        elif self.data_distr == 'zinb':
            alpha, rho, pi_nlogit = dec_outp  
            alpha = torch.clamp(alpha, min=eps)
            rho = torch.clamp(rho, min=1e-8, max=1-eps)            
            mu = rho * library
            log_alpha_mu = torch.log(alpha + mu)

            log_likelihood = torch.where(x < eps,
                F.softplus(pi_nlogit + alpha * (torch.log(alpha) - log_alpha_mu)) - F.softplus(pi_nlogit),
                - F.softplus(pi_nlogit) + pi_nlogit 
                + alpha * (torch.log(alpha) - log_alpha_mu) + x * (torch.log(mu) - log_alpha_mu) 
                + torch.lgamma(x + alpha) - torch.lgamma(alpha) - torch.lgamma(1.0 + x))
   
        return - torch.sum(log_likelihood, dim=-1) 

    def decode(
            self,
            z: torch.Tensor,
            label_inp: torch.Tensor
        ) -> List[torch.Tensor]:

        """
        Decode latent and label inputs to distribution parameters.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor of shape (batch_size, lat_dim).
        label_inp : torch.Tensor
            Label tensor of shape (batch_size, label_dim).

        Returns
        -------
        outputs : list of torch.Tensor
            [alpha, rho] or [alpha, rho, pi_nlogit].
        """

        x = torch.cat((z, label_inp), dim=-1)
        x = self.model(x)

        if self.dispersion == "dataset":
            alpha = self.log_alpha.exp()
        elif self.dispersion == "batch":
            alpha = self.log_alpha[torch.argmax(label_inp, dim=-1)].exp()
        elif self.dispersion == "cell":
            alpha = self.log_alpha(x).exp()

        rho_pre = self.rho_pre(x)
        rho_pre_hom = F.softmax(rho_pre[:, self.homologous_genes], dim=-1) * len(self.homologous_genes)/self.data_dim
        rho_pre_nonhom = F.softmax(rho_pre[:, self.non_hom_genes], dim=-1) * len(self.non_hom_genes)/self.data_dim
        rho = torch.cat((rho_pre_hom, rho_pre_nonhom), dim=-1)[:, self.gene_ind]

        outputs = [alpha, rho]

        if self.data_distr == 'zinb':
            pi_nlogit = self.pi_nlogit(x)
            outputs.append(pi_nlogit)
        return outputs  
    
    def decode_homologous(
            self,
            z: torch.Tensor,
            label_inp: torch.Tensor
        ) -> torch.Tensor:

        """
        Decodes the latent variables and label input into gene expression for homologous genes.
        This method is specifically used to asess and compare the log2-fold change between species.

        Parameters
        ----------
        z (Tensor): The latent space representation.
        label_inp (Tensor): The label input tensor.

        Returns
        -------
        Tensor: The decoded gene expression probabilities for homologous genes.
        """

        if self.data_distr == 'zinb':
            pi_nlogit = self.decode(z, label_inp)[-1]
            pi_hom = torch.sigmoid(pi_nlogit[:, self.homologous_genes])

            x = torch.cat((z, label_inp), dim=-1)
            x = self.model(x)
            rho_pre = self.rho_pre(x)
            rho_hom = F.softmax(rho_pre[:, self.homologous_genes], dim=-1) * pi_hom

        if self.data_distr == 'nb':
            x = torch.cat((z, label_inp), dim=-1)
            x = self.model(x)
            rho_pre = self.rho_pre(x)
            rho_hom = F.softmax(rho_pre[:, self.homologous_genes], dim=-1)

        return rho_hom  

    def forward(
            self,
            z: torch.Tensor,
            label_inp: torch.Tensor,
            library: torch.Tensor,
            x: torch.Tensor
        ) -> torch.Tensor:

        """
        Compute mean negative log-likelihood loss.

        Parameters
        ----------
        z : torch.Tensor
            Latent representations.
        label_inp : torch.Tensor
            Labels.
        library : torch.Tensor
            Library size factors.
        x : torch.Tensor
            Observed data.

        Returns
        -------
        torch.Tensor
            Mean negative log-likelihood over batch.
        """

        outputs = self.decode(z, label_inp)
        n_log_likeli = self.calc_nlog_likelihood(outputs, library, x).mean()
        return n_log_likeli     


class scSpecies():
    """
    The scSpecies cross-species architecture alignment framework built on scVI.

    This class implements end-to-end preprocessing, variational encoding, decoding, and alignment
    for a “context” dataset (e.g., mouse) and a “target” dataset (e.g., human). It supports:
    
    - Training scVI models on context and target (latent or intermediate alignment).
    - Library size encoding and negative-binomial / zero-inflated NB likelihoods.
    - Establishing a direct correspondece between traget can context cell via a likelihood-based similarity measure
    - Latent-space nearest-neighbor label transfer based on the similarity measure.
    - Log-fold-change computation of homologous genes.

    Parameters
    ----------
    device : str
        PyTorch device identifier ('cpu', 'mps' or 'cuda').
    mdata : mu.MuData
        Multi-modal container holding context and target AnnData objects, set up by the `create_mdata` class.
    directory : str
        Base path for saving model parameters, data, and figures.
    random_seed : int, default=369963
        Seed for NumPy and PyTorch RNGs.
    context_key : str, default='mouse'
        Key in `mdata.mod` for the context dataset.
    target_key : str, default='human'
        Key in `mdata.mod` for the target dataset.
    context_optimizer, target_optimizer : torch.optim.Optimizer classes
        Optimizer constructors for context and target models.
    context_hidden_dims_enc_outer, target_hidden_dims_enc_outer : list[int]
        Hidden layer sizes for the outer encoders.
    hidden_dims_enc_inner : list[int]
        Hidden layer sizes for the inner encoder.
    context_hidden_dims_l_enc, target_hidden_dims_l_enc : list[int]
        Hidden layer sizes for the library encoder.
    context_hidden_dims_dec, target_hidden_dims_dec : list[int]
        Hidden layer sizes for the decoder.
    context_layer_order, target_layer_order : list
        Layer specification lists for `create_structure`.
    b_s : int, default=128
        Batch size for training and inference.
    context_data_distr, target_data_distr : {'nb', 'zinb'}
        Observation models for counts.
    lat_dim : int, default=10
        Dimensionality of the latent space.
    context_dispersion, target_dispersion : {'dataset', 'batch', 'cell'}
        Dispersion parameterization strategy.
    alignment : {'inter', 'latent'}
        Alignment mode between context and target. Either at the outer encoder output space or at the latent space.
    k_neigh : int, default=25
        Number of neighbors candidates for alignment from the data-level NNS.
    top_percent : float, default=20
        Percentile cutoff for selecting top-agreement neighbors.
    context_beta_*, target_beta_*, eta_* : floats and ints
        Schedules for KL and alignment weight ramps.
    use_lib_enc : bool, default=True
        Whether to include a library-size encoder.
    """
 
    def __init__(
            self, 
            device: str,
            mdata: mu.MuData, 
            directory: str,  
            random_seed: int = 369963, 

            context_key: str = 'mouse', 
            target_key: str = 'human',      

            context_optimizer: torch.optim.Optimizer = torch.optim.Adam,
            target_optimizer: torch.optim.Optimizer = torch.optim.Adam,   

            context_hidden_dims_enc_outer: List[int] = [300],
            target_hidden_dims_enc_outer: List[int] = [300],
            hidden_dims_enc_inner: List[int] = [200],
            context_hidden_dims_l_enc: List[int] = [200],
            target_hidden_dims_l_enc: List[int] = [200],
            context_hidden_dims_dec: List[int] = [200, 300],
            target_hidden_dims_dec: List[int] = [200, 300],
            lat_dim: int = 10,

            context_layer_order: list = ['linear', 'layer_norm', ('act', nn.ReLU()), ('dropout', 0.1)],
            target_layer_order: list = ['linear', 'layer_norm', ('act', nn.ReLU()), ('dropout', 0.1)],#
            use_lib_enc: bool = True,  

            b_s: int = 128,    

            context_data_distr: str = 'zinb',
            target_data_distr: str = 'zinb',

            context_dispersion: str = 'batch',
            target_dispersion: str = 'batch',

            alignment: int = 'inter',
            k_neigh: int = 25,
            top_percent: float = 20,

            context_beta_start: float = 0.1,                
            context_beta_max: float  = 1,
            context_beta_epochs_raise: int = 10, 
            target_beta_start: float = 0.1,                
            target_beta_max: float  = 1,
            target_beta_epochs_raise: int = 10, 
            eta_start: float = 10,
            eta_max: float = 25,
            eta_epochs_raise: int = 10,    
        ):
        
        self.context_likeli_hist_dict = []
        self.target_likeli_hist_dict = []
        self.mdata = mdata

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.rng = np.random.default_rng(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed) 
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        _, hom_ind_context, hom_ind_target = np.intersect1d(mdata.mod[context_key].var_names, mdata.mod[target_key].var['var_names_transl'], return_indices=True)    

        self.config_dict = {
            'random_seed': random_seed, 
            'device': device, 
            'inter_dim': hidden_dims_enc_inner[0],
            'lat_dim': lat_dim,
            'b_s': b_s,    
            'alignment': alignment,   
            'use_lib_enc': use_lib_enc            
        }

        self.context_config = {
            'context_key': context_key,
            'optimizer': context_optimizer,
            'homologous_genes': list(hom_ind_context),
            'data_dim': self.mdata.mod[context_key].n_vars,
            'label_dim': np.shape(self.mdata.mod[context_key].obsm['batch_label_enc'])[1],  
            'lib_mu_add': round(np.mean(self.mdata.mod[context_key].obs['library_log_mean']),5),    
            'dims_enc_outer': context_hidden_dims_enc_outer,
            'dims_enc_inner': hidden_dims_enc_inner,
            'dims_l_enc': context_hidden_dims_l_enc,
            'lat_dim': lat_dim,
            'dims_dec': context_hidden_dims_dec,
            'layer_order': context_layer_order,
            'data_distr': context_data_distr,
            'dispersion': context_dispersion,
            'beta_start': context_beta_start, 
            'beta_max': context_beta_max,   
            'beta_epochs_raise': context_beta_epochs_raise,  
            'beta': context_beta_start, 
        }     

        self.target_config = {
            'target_key': target_key, 
            'optimizer': target_optimizer,
            'homologous_genes': list(hom_ind_target),
            'data_dim': self.mdata.mod[target_key].n_vars,
            'label_dim': np.shape(self.mdata.mod[target_key].obsm['batch_label_enc'])[1], 
            'lib_mu_add': round(np.mean(self.mdata.mod[target_key].obs['library_log_mean']),5), 
            'dims_enc_outer': target_hidden_dims_enc_outer,   
            'dims_enc_inner': hidden_dims_enc_inner,
            'dims_l_enc': target_hidden_dims_l_enc,
            'lat_dim': lat_dim,
            'dims_dec': target_hidden_dims_dec,
            'layer_order': target_layer_order,          
            'data_distr': target_data_distr,
            'dispersion': target_dispersion,            
            'beta_start': target_beta_start,                                      
            'beta_max': target_beta_max,     
            'beta_epochs_raise': target_beta_epochs_raise,              
            'beta': target_beta_start,          
            'k_neigh': k_neigh,
            'top_percent': top_percent,            
            'eta_start': eta_start,     
            'eta_max': eta_max,
            'eta_epochs_raise': eta_epochs_raise,             
            'eta': eta_start,  
        }     

        if self.context_config['dims_enc_outer'][-1] != self.target_config['dims_enc_outer'][-1]:
            raise ValueError("Context and target dims_enc_outer must have the same output dimensions.")       

        self.create_directory(directory)
        self.initialize()    

    def get_batch(
            self,
            array: Union[torch.Tensor, Sequence],
            step: int,
            *,
            perm: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = None
        ) -> Union[torch.Tensor, Sequence]:

        """
        Slice out a minibatch and move to device.

        Parameters
        ----------
        array : Tensor or sequence
            Data to batch (e.g., features, labels, indices).
        step : int
            Batch index.
        perm : sequence of int, optional
            Permutation for shuffling; if None, uses contiguous slices.
        batch_size : int, optional
            Number of samples per batch; defaults to `self.config_dict['b_s']`.

        Returns
        -------
        Tensor or sequence
            The selected batch, on the configured device if a Tensor.
        """

        bs = batch_size if batch_size is not None else self.config_dict['b_s']

        start = step * bs
        end   = start + bs
        idx = perm[start:end] if perm is not None else slice(start, end)

        batch = array[idx]

        device = self.config_dict.get('device')
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        
        return batch

    def initialize(
            self, 
            initialize: str = 'both'
        ):
        
        """
        Instantiate or reinstantiate context and/or target encoder and decoder modules.

        Parameters
        ----------
        initialize : {'context', 'context_decoder', 'target', 'both'}, default='both'
            Which sub-model(s) to initialize.
        """


        if initialize in ['context', 'context_decoder', 'both']:
            print('Initializing context scVI model.')
            self.context_decoder = Decoder(param_dict=self.context_config).to(self.config_dict['device']) 
            model_params = list(self.context_decoder.parameters())

            if initialize != 'context_decoder':
                self.context_encoder_inner = Encoder_inner(device=self.config_dict['device'],  param_dict=self.context_config).to(self.config_dict['device'])
                self.context_encoder_outer = Encoder_outer(param_dict=self.context_config).to(self.config_dict['device'])     
                model_params += list(self.context_encoder_outer.parameters()) + list(self.context_encoder_inner.parameters())

                if self.config_dict['use_lib_enc']:
                    self.context_lib_encoder = Library_encoder(device=self.config_dict['device'], param_dict=self.context_config).to(self.config_dict['device'])    
                    model_params +=  list(self.context_lib_encoder.parameters())
                    self.context_lib_encoder.__name__ = 'context_lib_encoder'                    
                    self.context_lib_encoder.eval()

            self.context_optimizer = self.context_config['optimizer'](model_params)

            self.context_encoder_inner.eval()
            self.context_encoder_outer.eval()
            self.context_decoder.eval()

            self.context_encoder_inner.__name__ = 'context_encoder_inner'
            self.context_encoder_outer.__name__ = 'context_encoder_outer'
            self.context_decoder.__name__ = 'context_decoder'        
            self.context_optimizer.__name__ = 'context_optimizer'   

        if initialize in ['target', 'both']:
            print('Initializing target scVI model.')
            self.target_encoder_outer = Encoder_outer(param_dict=self.target_config).to(self.config_dict['device'])
            self.target_decoder = Decoder(param_dict=self.target_config).to(self.config_dict['device'])  
            model_params = list(self.target_encoder_outer.parameters()) + list(self.target_decoder.parameters())

            if self.config_dict['use_lib_enc']:
                self.target_lib_encoder = Library_encoder(device=self.config_dict['device'], param_dict=self.target_config).to(self.config_dict['device'])       
                self.target_lib_encoder.eval()                    
                self.target_lib_encoder.__name__ = 'target_lib_encoder'      
                model_params += list(self.target_lib_encoder.parameters())

            if self.config_dict['alignment'] == 'latent':
                self.target_encoder_inner = Encoder_inner(device=self.config_dict['device'],  param_dict=self.context_config).to(self.config_dict['device'])
                model_params += list(self.target_encoder_inner.parameters())

            elif self.config_dict['alignment'] == 'inter':
                self.target_encoder_inner = self.context_encoder_inner

            self.target_optimizer = self.target_config['optimizer'](model_params)

            self.target_encoder_inner.eval()
            self.target_encoder_outer.eval()
            self.target_decoder.eval()

            self.target_encoder_inner.__name__ = 'target_encoder_inner' 
            self.target_encoder_outer.__name__ = 'target_encoder_outer'
            self.target_decoder.__name__ = 'target_decoder'        
            self.target_optimizer.__name__ = 'target_optimizer'   


    def create_directory(
            self, 
            directory: str
        ):
        
        """
        Create project subdirectories for parameters, data, and figures.

        Parameters
        ----------
        directory : str
            Base output directory.
        """

        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory '{directory}'.")

        self.prm_dir = os.path.join(directory, 'params')
        self.dat_dir = os.path.join(directory, 'data')
        self.fig_dir = os.path.join(directory, 'figures')

        for path in (self.prm_dir, self.dat_dir, self.fig_dir):
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Created directory '{path}'.")

    def pkl(self, model_name: str, save_key: str): return os.path.join(self.prm_dir, f"{model_name}_{save_key}.pkl")
    def pth(self, model_name: str, save_key: str): return os.path.join(self.prm_dir, f"{model_name}_{save_key}.pth")
    def opt(self, model_name: str, save_key: str): return os.path.join(self.prm_dir, f"{model_name}_{save_key}.opt")
    def hmu(self, model_name: str, save_key: str): return os.path.join(self.dat_dir, f"{model_name}_{save_key}.h5mu")

    def save(
            self, 
            models: str = 'both', 
            save_key: str = ''
        ):
        
        """
        Serialize model configuration, optimizers, and context and/or target scVI weights to disk.

        Parameters
        ----------
        models : {'context', 'target', 'both'}
            Which sub-models to save.
        save_key : str
            Suffix for filenames.
        """

        path = os.path.join(self.prm_dir, f"config_dict.pkl")
        with open(path, 'wb') as f:
            pickle.dump(self.config_dict, f)
        print(f"Saved {path}")

        model_list = []
        if models in ('context', 'both'):
            model_list += [self.context_encoder_inner, self.context_encoder_outer, self.context_decoder]
            if self.config_dict['use_lib_enc']:            
                model_list += [self.context_lib_encoder]
                
            path = self.pkl('context_config', save_key)
            with open(path, 'wb') as f:
                pickle.dump(self.context_config, f)
            print(f"Saved {path}")

            path = self.opt('context_optimizer', save_key)
            torch.save(self.context_optimizer.state_dict(), path)
            print(f"Saved {path}")

        if models in ('target', 'both'):
            model_list += [self.target_encoder_inner, self.target_encoder_outer, self.target_decoder]
            if self.config_dict['use_lib_enc']:
                model_list += [self.target_lib_encoder]
                
            path = self.pkl('target_config', save_key)
            with open(path, 'wb') as f:
                pickle.dump(self.target_config, f)
            print(f"Saved {path}")

            path = self.opt('target_optimizer', save_key)
            torch.save(self.target_optimizer.state_dict(), path)
            print(f"Saved {path}")

        for model in model_list:
            path = self.pth(model.__name__, save_key)
            torch.save(model.state_dict(), path)
            print(f'Saved {path}.')  

    def save_mdata(
            self, 
            save_key: str
        ): 
        
        """
        Write the assembled MuData object to `.h5mu`.

        Parameters
        ----------
        save_key : str
            Suffix for the data filename.
        """

        path = self.hmu(self.dat_dir, save_key) 
        self.mdata.write(path) 
        print(f'Saved {path}')

    def load(
            self, 
            models: str = 'both', 
            save_key:str = ''
        ): 
        
        """
        Load previously saved configs, optimizers, and weights.

        Parameters
        ----------
        models : {'context', 'target', 'both'}
        save_key : str
        """

        path = os.path.join(self.prm_dir, f"config_dict.pkl")
        with open(path, 'wb') as f:
            pickle.dump(self.config_dict, f)
        print(f"Loaded {path}")

        if models in ('context','both'):
            path = self.pkl('context_config', save_key)
            with open(path, 'rb') as f:
                self.context_config = pickle.load(f)
            print(f"Loaded {path}")

        if models in ('target','both'):
            path = self.pkl('target_config', save_key)
            with open(path, 'rb') as f:
                self.target_config = pickle.load(f)
            print(f"Loaded {path}")


        if models in ('context','both'):
            path = self.opt('context_optimizer', save_key)
            state = torch.load(path, map_location=torch.device(self.config_dict['device']))
            self.context_optimizer.load_state_dict(state)
            print(f"Loaded {path}")

        if models in ('target', 'both'):
            path = self.opt('target_optimizer', save_key)
            state = torch.load(path, map_location=torch.device(self.config_dict['device']))
            self.target_optimizer.load_state_dict(state)
            print(f"Loaded {path}")            


        model_list = []
        if models in ('context', 'both'):
            model_list += [self.context_encoder_outer, self.context_decoder, self.context_encoder_inner]
            if self.config_dict['use_lib_enc']:
                model_list += [self.context_lib_encoder]

        if models in ('target', 'both'):
            model_list += [self.target_encoder_outer, self.target_decoder, self.target_encoder_inner]
            if self.config_dict['use_lib_enc']:
                model_list += [self.target_lib_encoder]

        for model in model_list:
            path = self.pth(model.__name__, save_key)
            model.load_state_dict(torch.load(path, map_location=torch.device(self.config_dict['device'])))    
            print(f"Loaded {path}")        

            if self.config_dict['alignment'] == 'inter':
                self.target_encoder_inner = self.context_encoder_inner   

    @staticmethod
    def most_frequent(arr: np.ndarray) -> np.ndarray:
        """
        Return the modal value of a 1D array.
        Helper for the `label_transfer` function.

        Parameters
        ----------
        arr : array-like

        Returns
        -------
        element
            The value occurring most often.
        """

        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]

    def transfer_labels_cell(
            self,     
            target_ind: int,
            context_obs_transfer: Union[List[str], str],
        ) -> pd.DataFrame:
        
        """
        Calculate similarity scores for a specific target cell specified by its index in `self.mdata[target_key].X`
        and all context cells. Transfers labels specifies in context_obs_transfer. Returns a dataframe 
        of context cells sorted by similarity scores. 

        Parameters
        ----------
        target_ind : int
            Target cell indices.
        context_obs_transfer : str or List of str
            Observation key from context dataset to return as columns in the outpt (e.g., 'cell_type').

        Returns
        -------
        DataFrame
            Context labels, source indices, and similarity scores with the specified target cell.
        """

        if isinstance(context_obs_transfer, str):
            context_obs_transfer = [context_obs_transfer]

        context_inds = np.arange(self.mdata[self.context_config['context_key']].n_obs)[np.newaxis, :]
        similarities = self.similarity_metric(np.full(np.arange(1).shape, target_ind, dtype=int), context_inds)

        df_neigbor = self.mdata.mod[self.context_config['context_key']][np.argsort(similarities)].obs.copy()[context_obs_transfer]
        df_neigbor['index'] = np.squeeze(np.argsort(similarities))
        df_neigbor['similarity_score'] = np.squeeze(similarities[:, np.argsort(similarities)])
        return df_neigbor

    def similarity_metric(
            self,
            target_ind: np.ndarray,
            context_ind: np.ndarray,
            b_s: Optional[int] = None,
            b_sc: Optional[int] = None,
            display = True,
        ) -> np.ndarray:
        
        """
        Compute negative log-likelihood based similarity scores for target and context cells specified by their indices.

        Parameters
        ----------
        target_ind : array of integers 
            Traget cell indices in `self.mdata[target_key].X` shape (n_target, 1)
        context_ind : array of integers 
            Context cell neighbors in `self.mdata[context_key].X` shape (n_target, k). 
            Calculates the similarity of k candidates for a specific entry in the first axis.
        b_s : int, optional
            Batch size for target.
        b_sc : int, optional
            Chunk size for context neighbors.
        display : bool    
            If True, prints progress.

        Returns
        -------
        similarities : ndarray 
            Contains the similarity scores between the context cells and their k candidates, shape (n_target, k).
        """

        if b_s == None: 
            b_s = self.config_dict['b_s']

        if b_sc == None:
            b_sc = int(128*25/b_s)    

        k_neigh = np.shape(context_ind)[1]
        steps = int(np.ceil(np.shape(target_ind)[0]/b_s)) # +1e-10
        steps_c_ind = int(np.ceil(k_neigh/b_sc))

        similarities = []

        with torch.no_grad():
            tic = time.time()
            for step in range(steps):   

                if display == True and time.time() - tic > 0.5:
                    tic = time.time()
                    print('\rCalculate similarity metric. Step {}/{}.'.format(str(step), str(steps)), end='', flush=True) 

                target_ind_batch = self.get_batch(target_ind, step, batch_size=b_s)
                target_adata_batch = self.mdata.mod[self.target_config['target_key']][target_ind_batch]
                target_x_batch = torch.from_numpy(target_adata_batch.X.toarray()).to(self.config_dict['device'])  
                target_s_batch = torch.from_numpy(target_adata_batch.obsm['batch_label_enc']).to(self.config_dict['device'])  
                target_l_batch = torch.from_numpy(target_adata_batch.obsm['l_mu']).exp().to(self.config_dict['device'])  
                sim_batch_c = []

                for step_c in range(steps_c_ind):
                    context_ind_batch = self.get_batch(context_ind, step, batch_size=b_s)[:, step_c*b_sc:(step_c+1)*b_sc]
                    
                    s_interl = torch.repeat_interleave(target_s_batch, repeats=np.shape(context_ind_batch)[-1], dim=0)
                    l_interl = torch.repeat_interleave(target_l_batch, repeats=np.shape(context_ind_batch)[-1], dim=0)
                    x_interl = torch.repeat_interleave(target_x_batch, repeats=np.shape(context_ind_batch)[-1], dim=0)                
                    context_ind_batch_sq = np.squeeze(np.reshape(context_ind_batch, (-1, target_x_batch.size(0)*np.shape(context_ind_batch)[-1])))
                    context_z_batch = torch.from_numpy(self.mdata.mod[self.context_config['context_key']].obsm['z_mu'][context_ind_batch_sq]).to(self.config_dict['device']) 

                    outp_neighbors = self.target_decoder.decode(context_z_batch, s_interl)
                    outp = self.target_decoder.calc_nlog_likelihood(outp_neighbors, l_interl, x_interl).reshape(target_x_batch.size(0), np.shape(context_ind_batch)[-1]).cpu().numpy() 
                    sim_batch_c.append(outp)

                sim_batch_c = np.concatenate(sim_batch_c, axis=-1)
                target_z_batch = torch.from_numpy(target_adata_batch.obsm['z_mu']).to(self.config_dict['device'])  
                outp_target = self.target_decoder.decode(target_z_batch, target_s_batch)
                sim_batch_c -= self.target_decoder.calc_nlog_likelihood(outp_target, target_l_batch, target_x_batch).unsqueeze(-1).cpu().numpy() 

                similarities.append(sim_batch_c)

            similarities = np.concatenate(similarities)
            return similarities


    def ret_pred_df(
            self,
            pred_key: str,
            target_label_key: str,
            context_label_key: str    
        ) -> Tuple[pd.DataFrame, float]:

        """
        Compute a normalized confusion matrix (%) and balanced accuracy for label transfer.

        This evaluates how well the predicted context-derived labels match the true labels
        on the target dataset.

        Parameters
        ----------
        pred_key : str
            Key in `self.mdata.mod[target_key].obs` under which predicted labels are stored.
        target_label_key : str
            Key in `self.mdata.mod[target_key].obs` for the ground-truth labels.
        context_label_key : str
            Key in `self.mdata.mod[context_key].obs` for the reference context labels.

        Returns
        -------
        df : pd.DataFrame
            Confusion matrix (in percent) with
            - index: sorted labels of `target_label_key`,
            - columns: sorted labels of `context_label_key`,
            - values: percentage of cells with true label = row and predicted label = column.
        bas : float
            Balanced accuracy score computed only over the subset of cells whose true labels
            also appear in the context set.
        """

        predicted_labels = self.mdata.mod[self.target_config['target_key']].obs[pred_key].to_numpy()
        target_labels = self.mdata.mod[self.target_config['target_key']].obs[target_label_key].to_numpy()
        context_labels = self.mdata.mod[self.context_config['context_key']].obs[context_label_key].to_numpy()
        unique_target_labels = np.unique(target_labels)
        unique_context_labels = np.unique(context_labels)

        joint_labels = np.intersect1d(context_labels, target_labels)
        joint_ind = np.where(np.array([cell_label in joint_labels for cell_label in target_labels]))[0]
        bas = balanced_accuracy_score(target_labels[joint_ind], predicted_labels[joint_ind])

        df = pd.DataFrame(0, index=unique_target_labels, columns=unique_context_labels, dtype=int)
        for true_lbl, pred_lbl in zip(target_labels, predicted_labels):
            df.loc[true_lbl, pred_lbl] += 1

        df = df.div(df.sum(axis=1), axis=0) * 100
        return df, bas

    def similarity_metric_on_latent_space(
            self,
            precompute_neighbors: bool = True
        ) -> Tuple[np.ndarray, np.ndarray]:
        
        """
        Compute similarity scores for the whole context and target dataset pairs.
        Either for a precomputed set of neighbors based on the results of a latent spce neighborhood search to speed up computation
        or the whole dataset. (Should only be done for small datasets.)

        Parameters
        ----------
        precompute_neighbors : bool
            If True precomutes a set of 250 euclidean neighbors on the aligned latent space.

        Returns
        -------
        similarities : ndarray of shape (target.n_obs, 250) or (target.n_obs, context.n_obs)
        context_ind : ndarray of shape (target.n_obs, 250) or (target.n_obs, context.n_obs)
        """


        if precompute_neighbors != True: 
            n_obs_context = self.mdata.mod[self.context_config['context_key']].n_obs
            n_obs_target = self.mdata.mod[self.target_config['target_key']].n_obs

            n_evals = n_obs_context * n_obs_target
            prompt = (
                f"Warning, not precomputing neighbors on the latent space will lead to "
                f"a total of {n_evals} decoder evaluations. Proceed? [y/N] "
            )

            resp = input(prompt)
            if resp.strip().lower() not in ('y', 'yes'):
                print("Set precompute_neighbors to true.")
                precompute_neighbors == True

            else: 
                context_ind = np.concat([np.arange(n_obs_target)[:, np.newaxis] for i in range(n_obs_context)], axis=-1)    

        if precompute_neighbors == True:
            print('Pre-computing latent space NNS with 250 neighbors using the euclidean distance.')

            neigh = NearestNeighbors(n_neighbors=250, metric='euclidean')
            neigh.fit(self.mdata.mod[self.context_config['context_key']].obsm['z_mu'])
            _, context_ind = neigh.kneighbors(self.mdata.mod[self.target_config['target_key']].obsm['z_mu'])

        target_ind = np.arange(self.mdata.mod[self.target_config['target_key']].n_obs)    
        similarities = self.similarity_metric(target_ind, context_ind, b_s=None, b_sc=None)
        return similarities, context_ind

    def transfer_labels_data(    
            self,
            context_obs_transfer: Union[List[str], str],
            top_neigh: int = 25,
            write_sim: bool = False
            ):
        
        """
        Assign context-derived labels via similarity scores to each target cell by majority vote among its top candidates.

        For each observation key in `context_obs_transfer`, finds the `top_neigh` most similar context
        cells (based on decoder likelihood in latent space), takes the most frequent label among
        those neighbors, and writes it into `self.mdata.mod[target_key].obs['pred_sim_<obs_key>']`.
        When target cell annotation is unknown, the inferred values of the last entry in `context_obs_transfer` will serve as a replacement for target cell annotation in downstream analyses.

        Parameters
        ----------
        context_obs_transfer : List of str or str
            One or more keys in `self.mdata.mod[context_key].obs` whose values to transfer. 
        top_neigh : int, default=25
            Number of nearest neighbors to consider for the majority vote.
        write_sim : bool, default=False
            If True, also stores raw similarity scores and neighbor indices in
            `self.mdata.mod[target_key].obsm['similarities']` and
            `['similarities_ind']`.
        """
        target_key = self.target_config['target_key']

        similarities, context_ind = self.similarity_metric_on_latent_space()
                    
        if isinstance(context_obs_transfer, str):
            context_obs_transfer = [context_obs_transfer]

        if write_sim == True:
            self.mdata.mod[target_key].obsm['similarities'] = pred_labels 
            self.mdata.mod[target_key].obsm['similarities_ind'] = context_ind 

        for obs_key in context_obs_transfer:
            context_labels = self.mdata.mod[self.context_config['context_key']].obs[obs_key].to_numpy()
            target_n_obs = self.mdata.mod[target_key].n_obs
            pred_labels = np.stack([self.most_frequent(context_labels[context_ind[i][np.argsort(similarities[i])]][:top_neigh]) for i in range(target_n_obs)])
            self.mdata.mod[target_key].obs['pred_sim_'+obs_key] = pred_labels 

        if self.mdata.mod[target_key].uns['metadata']['cell_key'] == 'unknown':
            self.mdata.mod[target_key].uns['metadata']['cell_key_transferred'] = context_obs_transfer[-1]
            print(f'Set {context_obs_transfer[-1]} as target cell key for downstream analyses.')

    @staticmethod
    def average_slices(    
            array: np.ndarray,
            slice_sizes: Sequence[int]
        ) -> np.ndarray:

        """
        Compute the mean of consecutive subarrays of a flat 2D array.
        Helper for `compute_logfold_change`.

        Parameters
        ----------
        array : ndarray, shape (sum(slice_sizes), n_features)
            The concatenated data.
        slice_sizes : sequence of int
            Positive integers that sum to `array.shape[0]`.

        Returns
        -------
        stacked_means : ndarray, shape (len(slice_sizes), n_features)
            The mean of each slice.
        """

        averages = []
        start = 0
        for size in slice_sizes:
            end = start + size
            slice_avg = np.mean(array[start:end], axis=0)
            averages.append(slice_avg)
            start = end
        return np.stack(averages)

    @staticmethod
    def filter_outliers(    
            data: np.ndarray,
            confidence_level: float = 0.9
        ) -> Tuple[np.ndarray, np.ndarray]:
        
        """
        Identify inlier and outlier rows based on the Mahalanobis distance.

        Computes the Mahalanobis distance of each row in `data` from the multivariate mean,
        uses a chi-squared cutoff at the given `confidence_level`, and returns boolean masks.
        Helper for `compute_logfold_change`.
        
        Parameters
        ----------
        data : ndarray, shape (n_samples, n_features)
            Input points in feature space.
        confidence_level : float, default=0.9
            Threshold percentile for declaring a point an inlier.

        Returns
        -------
        inlier_mask : ndarray of bool, shape (n_samples,)
            True for rows whose Mahalanobis distance is below the threshold.
        outlier_mask : ndarray of bool, shape (n_samples,)
            True for rows whose distance exceeds the threshold.
        """

        mean = np.mean(data, axis=0)
        data_centered = data - mean
        cov_matrix = np.dot(data_centered.T, data_centered) / (data_centered.shape[0] - 1)
        cov_inv = np.linalg.inv(cov_matrix)

        m_dist = np.sqrt(np.sum(np.dot(data_centered, cov_inv) * data_centered, axis=1))

        df = mean.shape[0]  
        threshold = np.sqrt(chi2.ppf(confidence_level, df))

        filtered_data_ind = m_dist < threshold
        outlier_ind = m_dist >= threshold
        return filtered_data_ind, outlier_ind


    def generate_homologous_samples(
            self,
            samples: int = 5000,
            target_cell_key = None,
            b_s: int = 128,
            confidence_level: float = 0.9
        ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        """
        Decode homologous normalized expression profiles for context and target species by Monte Carlo sampling.

        Parameters
        ----------
        target_cell_key : str or None
            Column name in `.obs` specifying inferred cell type labels for the target dataset;
        samples : int, default=5000
            Total number of decoded samples to return per cell type.
        b_s : int, default=128
            Batch size for decoding iterations.
        confidence_level : float, default=0.9
            Quantile threshold used in `filter_outliers` to remove extreme latent embeddings.

        Returns
        -------
        target_rho_dict : dict of str to ndarray of shape (samples, genes)
            Decoded normalized expression (`rho`) for shared cell types in the target species.
        context_rho_dict : dict of str to ndarray of shape (samples, genes)
            Decoded normalized expression (`rho`) for shared cell types in the context species.
        """

        self.context_decoder.eval()   
        self.target_decoder.eval()    
        self.context_encoder_inner.eval()   
        self.target_encoder_inner.eval() 
        self.context_encoder_outer.eval()   
        self.target_encoder_outer.eval() 
        self.context_lib_encoder.eval()   
        self.target_lib_encoder.eval()         

        context_key = self.context_config['context_key']
        target_key = self.target_config['target_key']
        context_cell_key = self.mdata.mod[context_key].uns['metadata']['cell_key']
        if target_cell_key == None:
            target_cell_key = self.mdata.mod[target_key].uns['metadata']['cell_key']
        context_batch_key = self.mdata.mod[context_key].uns['metadata']['batch_key']
        target_batch_key = self.mdata.mod[target_key].uns['metadata']['batch_key']   

        if target_cell_key == 'unknown':
            if 'cell_key_transferred' in self.mdata.mod[target_key].uns['metadata'].keys():
                target_cell_key = self.mdata.mod[target_key].uns['metadata']['cell_key_transferred']
                print(f'Use inferred context labels in {target_cell_key} for similarity calculation.')
            else:    
                raise ValueError(f"Target cell labels must be known or transferred from the context dataset via `label_transfer`.") 

        context_cell_labels = self.mdata.mod[context_key].obs[context_cell_key].to_numpy()
        context_cell_types = np.unique(context_cell_labels)

        target_cell_labels = self.mdata.mod[target_key].obs[target_cell_key].to_numpy()
        target_cell_types = np.unique(target_cell_labels)
        target_cell_index = {c : np.where(target_cell_labels == c)[0] for c in target_cell_types}
        
        context_batch_labels = self.mdata.mod[context_key].obs[context_batch_key].to_numpy().reshape(-1, 1)
        target_batch_labels = self.mdata.mod[target_key].obs[target_batch_key].to_numpy().reshape(-1, 1)

        context_enc = OneHotEncoder()
        context_enc.fit(context_batch_labels)

        target_enc = OneHotEncoder()
        target_enc.fit(target_batch_labels)

        context_batches = {c : self.mdata.mod[context_key][self.mdata.mod[context_key].obs[context_cell_key] == c].obs[context_batch_key].value_counts() > 3 for c in context_cell_types}
        context_batches = {c : context_batches[c][context_batches[c]].index.to_numpy() for c in context_cell_types}
        context_batches = {c : context_enc.transform(context_batches[c].reshape(-1, 1)).toarray().astype(np.float32)  for c in context_cell_types}
        context_batches['unknown'] = context_enc.transform(np.unique(context_batch_labels).reshape(-1, 1)).toarray().astype(np.float32)

        joint_cell_types = np.intersect1d(context_cell_types, target_cell_types, return_indices=True)[0]
        target_batches = {c : self.mdata.mod[target_key][self.mdata.mod[target_key].obs[target_cell_key] == c].obs[target_batch_key].value_counts() > 3 for c in target_cell_types}
        target_batches = {c : target_batches[c][target_batches[c]].index.to_numpy() for c in target_cell_types}
        target_batches = {c : target_enc.transform(target_batches[c].reshape(-1, 1)).toarray().astype(np.float32)  for c in target_cell_types}
        target_batches['unknown'] = target_enc.transform(np.unique(target_batch_labels).reshape(-1, 1)).toarray().astype(np.float32)

        context_rho_dict = {}
        target_rho_dict = {}

        for cell_type in joint_cell_types:
            adata_target = self.mdata.mod[target_key][target_cell_index[cell_type]]

            filtered_data_ind, _ = self.filter_outliers(adata_target.obsm['z_mu'], confidence_level=confidence_level)
            adata_target = adata_target[filtered_data_ind]      

            steps = np.ceil(adata_target.n_obs/b_s).astype(int)    
            iterations = int(np.ceil(samples/adata_target.n_obs))

            with torch.no_grad():
                context_rho_dict[cell_type] = []    
                target_rho_dict[cell_type] = []  

                for iter in range(iterations):
                    for step in range(steps):   
                        batch_adata = adata_target[step*b_s:(step+1)*b_s]
                        context_cell_type = batch_adata.obs[context_cell_key].to_numpy()
                        target_cell_type = batch_adata.obs[target_cell_key].to_numpy() 

                        context_labels = np.concatenate([context_batches[c] for c in context_cell_type])
                        target_labels = np.concatenate([target_batches[c] for c in target_cell_type])
                    
                        context_labels = torch.from_numpy(context_labels).to(self.config_dict['device'])
                        target_labels = torch.from_numpy(target_labels).to(self.config_dict['device'])            

                        context_ind_batch = np.array([np.shape(context_batches[c])[0] for c in context_cell_type])
                        target_ind_batch = np.array([np.shape(target_batches[c])[0] for c in target_cell_type])

                        shape = np.shape(batch_adata.obsm['z_sig'])
                        z = np.float32(batch_adata.obsm['z_mu'] + batch_adata.obsm['z_sig'] * np.random.rand(shape[0], shape[1])) 

                        context_z = np.concatenate([np.tile(z[j], (i, 1)) for j, i in enumerate(context_ind_batch)])
                        target_z = np.concatenate([np.tile(z[j], (i, 1)) for j, i in enumerate(target_ind_batch)])

                        context_z = torch.from_numpy(context_z).to(self.config_dict['device'])
                        target_z = torch.from_numpy(target_z).to(self.config_dict['device'])

                        context_rho = self.context_decoder.decode_homologous(context_z, context_labels).cpu().numpy()
                        context_rho = self.average_slices(context_rho, context_ind_batch)

                        target_rho = self.target_decoder.decode_homologous(target_z, target_labels).cpu().numpy()
                        target_rho = self.average_slices(target_rho, target_ind_batch)

                        context_rho_dict[cell_type].append(context_rho)
                        target_rho_dict[cell_type].append(target_rho)

            target_rho_dict[cell_type] = np.concatenate(target_rho_dict[cell_type])[:samples]
            context_rho_dict[cell_type] = np.concatenate(context_rho_dict[cell_type])[:samples]  

        return target_rho_dict, context_rho_dict


    def compute_logfold_change(    
            self,
            eval_cell_types: Optional[Sequence[str]] = None,
            eps: float = 1e-6,
            lfc_delta: float = 1,
            samples: int = 50000,
            target_cell_key = None,            
            b_s: int = 128,
            confidence_level: float = 0.9
        ) -> Dict[str, pd.DataFrame]:
        """
        Monte Carlo estimation of per-gene Log2-fold-changes and associated probabilities.

        For each specified cell type (or the intersection of context/target types), samples
        from the scVI posterior, computes the ratio of target vs. context expression for each
        homologous gene, and aggregates:
        - Median Log2-fold-change (on normalized decoder space),
        - Probability(abs(Log2Fc) > lfc_delta),
        - Mean gene expression on normalized decoder space and NB parameter space.
        
        Parameters
        ----------
        eval_cell_types : sequence of str, optional
            Cell types to include; defaults to the intersection of context and target types.
        eps : float, default=1e-6
            Small constant added before log to prevent small gene expression patterns from returning large LFC values.
        lfc_delta : float, default=1
            Threshold for computing the probability of large fold-changes.
        target_cell_key : str or None
            Column name in `.obs` specifying inferred cell type labels for the target dataset;            
        samples : int, default=50000
            Total number of Monte Carlo draws per cell.
        b_s : int, default=128
            Batch size for sampling iterations.
        confidence_level : float, default=0.9
            Outlier filtering threshold for latent space.

        Returns
        -------
        lfc_dict : dict of str to pd.Dataframe
            Dictionary with cell-wise data frames containing the keys:
            - 'rho_median_context' : Median context normalized gene expression, 
            - 'mu_median_context' : Median context expected value gene expression,  
            - 'rho_median_target' : Median target normalized gene expression,  
            - 'mu_median_target' : Median target expected value gene expression,  
            - 'lfc' : Median Log2 fold-change of the relative expression parameter rho,
            - 'p' : Probability of Log2 fold-change values greater than lfc_delta,
            - 'lfc_rand' : Median Log2 fold-change of the relative expression parameter rho on permuted data,
            - 'p_rand' : Probability of Log2 fold-change values greater than lfc_delta  on permuted data. 
        """

        context_key = self.context_config['context_key']
        target_key = self.target_config['target_key']
        context_cell_key = self.mdata.mod[context_key].uns['metadata']['cell_key']
        if target_cell_key == None:
            target_cell_key = self.mdata.mod[target_key].uns['metadata']['cell_key']
        context_batch_key = self.mdata.mod[context_key].uns['metadata']['batch_key']
        target_batch_key = self.mdata.mod[target_key].uns['metadata']['batch_key']    

        if target_cell_key == 'unknown':
            if 'cell_key_transferred' in self.mdata.mod[target_key].uns['metadata'].keys():
                target_cell_key = self.mdata.mod[target_key].uns['metadata']['cell_key_transferred']
                print(f'Use inferred context labels in {target_cell_key} for differential gene expression analysis.')
            else:    
                raise ValueError(f"Target cell labels must be known or transferred from the context dataset via `label_transfer`.") 

        self.context_decoder.eval()   
        self.target_decoder.eval()    
        self.context_encoder_inner.eval()   
        self.target_encoder_inner.eval() 
        self.context_encoder_outer.eval()   
        self.target_encoder_outer.eval() 
        self.context_lib_encoder.eval()   
        self.target_lib_encoder.eval()         
                            
        target_ind = np.array(self.target_config['homologous_genes'])
        target_gene_names = self.mdata.mod[target_key].var_names.to_numpy()[target_ind]

        context_cell_labels = self.mdata.mod[context_key].obs[context_cell_key].to_numpy()
        context_cell_types = np.unique(context_cell_labels)
        context_cell_index = {c : np.where(context_cell_labels == c)[0] for c in context_cell_types}

        target_cell_labels = self.mdata.mod[target_key].obs[target_cell_key].to_numpy()
        target_cell_types = np.unique(target_cell_labels)
        target_cell_index = {c : np.where(target_cell_labels == c)[0] for c in target_cell_types}
        
        context_batch_labels = self.mdata.mod[context_key].obs[context_batch_key].to_numpy().reshape(-1, 1)
        target_batch_labels = self.mdata.mod[target_key].obs[target_batch_key].to_numpy().reshape(-1, 1)

        context_enc = OneHotEncoder()
        context_enc.fit(context_batch_labels)

        target_enc = OneHotEncoder()
        target_enc.fit(target_batch_labels)

        context_batches = {c : self.mdata.mod[context_key][self.mdata.mod[context_key].obs[context_cell_key] == c].obs[context_batch_key].value_counts() > 3 for c in context_cell_types}
        context_batches = {c : context_batches[c][context_batches[c]].index.to_numpy() for c in context_cell_types}
        context_batches = {c : context_enc.transform(context_batches[c].reshape(-1, 1)).toarray().astype(np.float32)  for c in context_cell_types}
        context_batches['unknown'] = context_enc.transform(np.unique(context_batch_labels).reshape(-1, 1)).toarray().astype(np.float32)

        if eval_cell_types==None: 
            eval_cell_types = np.intersect1d(context_cell_types, target_cell_types)

        target_batches = {c : self.mdata.mod[target_key][self.mdata.mod[target_key].obs[target_cell_key] == c].obs[target_batch_key].value_counts() > 3 for c in target_cell_types}
        target_batches = {c : target_batches[c][target_batches[c]].index.to_numpy() for c in target_cell_types}
        target_batches = {c : target_enc.transform(target_batches[c].reshape(-1, 1)).toarray().astype(np.float32)  for c in target_cell_types}
        target_batches['unknown'] = target_enc.transform(np.unique(target_batch_labels).reshape(-1, 1)).toarray().astype(np.float32)

        random_perm = np.random.permutation(len(target_gene_names))

        lfc_dict = {}
        for cell_type in eval_cell_types:
            adata_context = self.mdata.mod[context_key][context_cell_index[cell_type]]
            adata_target = self.mdata.mod[target_key][target_cell_index[cell_type]]
            adata_context.obs_names_make_unique() 
            adata_target.obs_names_make_unique() 

            filtered_data_ind, _ = self.filter_outliers(adata_context.obsm['z_mu'], confidence_level=confidence_level)
            adata_context = adata_context[filtered_data_ind].copy()

            filtered_data_ind, _ = self.filter_outliers(adata_target.obsm['z_mu'], confidence_level=confidence_level)
            adata_target = adata_target[filtered_data_ind].copy()    

            latent_target = adata_target.obsm['z_mu']
            latent_context = adata_context.obsm['z_mu']
            nn = NearestNeighbors(n_neighbors=25, metric='cosine', algorithm='auto')
            nn.fit(latent_context)
            distances, indices = nn.kneighbors(latent_target)
            adata_target.obsm['cell_context_ind'] = indices

            steps = np.ceil(adata_target.n_obs/b_s).astype(int)    
            sampling_size = max(int(samples / adata_target.n_obs), 1)

            with torch.no_grad():
                lfc_list = []    
                lfc_list_random = []                  
                rho_mouse = []     
                mu_mouse = []        
                rho_human = []     
                mu_human = []                    

                for step in range(steps):   
                    batch_adata = adata_target[step*b_s:(step+1)*b_s]
                    context_cell_type = batch_adata.obs[target_cell_key].to_numpy()
                    target_cell_type = batch_adata.obs[target_cell_key].to_numpy() 

                    context_labels = np.concatenate([context_batches[c] for c in context_cell_type])
                    target_labels = np.concatenate([target_batches[c] for c in target_cell_type])
                    context_labels = torch.from_numpy(context_labels).to(self.config_dict['device'])
                    target_labels = torch.from_numpy(target_labels).to(self.config_dict['device'])            

                    context_ind_batch = np.array([np.shape(context_batches[c])[0] for c in context_cell_type])
                    target_ind_batch = np.array([np.shape(target_batches[c])[0] for c in target_cell_type])

                    shape = np.shape(batch_adata.obsm['z_sig'])
                    
                    for k in range(sampling_size):
                        z = np.float32(batch_adata.obsm['z_mu'] + batch_adata.obsm['z_sig'] * np.random.rand(shape[0], shape[1])) 
                        target_l = np.exp(np.float32(batch_adata.obsm['l_mu'] + batch_adata.obsm['l_sig'] * np.random.rand(shape[0], 1)))
                        neigh_ind = batch_adata.obsm['cell_context_ind']
                        
                        context_l = np.exp(np.float32(adata_context.obsm['l_mu'][neigh_ind] + adata_context.obsm['l_sig'][neigh_ind] * np.random.rand(shape[0], 25, 1)))
                        context_l = context_l.mean(axis=1)

                        context_z = np.concatenate([np.tile(z[j], (i, 1)) for j, i in enumerate(context_ind_batch)])
                        target_z = np.concatenate([np.tile(z[j], (i, 1)) for j, i in enumerate(target_ind_batch)])

                        context_z = torch.from_numpy(context_z).to(self.config_dict['device'])
                        target_z = torch.from_numpy(target_z).to(self.config_dict['device'])

                        context_rho = self.context_decoder.decode_homologous(context_z, context_labels).cpu().numpy()
                        context_rho = self.average_slices(context_rho, context_ind_batch)

                        target_rho = self.target_decoder.decode_homologous(target_z, target_labels).cpu().numpy()
                        target_rho = self.average_slices(target_rho, target_ind_batch)
   
                        context_mu = context_rho * context_l
                        target_mu = target_rho * target_l

                        rho_mouse.append(context_rho)
                        mu_mouse.append(context_mu)
                        rho_human.append(target_rho)
                        mu_human.append(target_mu)

                        lfc_list.append(np.log2(target_rho+eps) - np.log2(context_rho+eps))
                        lfc_list_random.append(np.log2(target_rho+eps) - np.log2(context_rho[:, random_perm]+eps))

            lfc_dict[cell_type] = pd.DataFrame(0, index=target_gene_names, columns=[
                'rho_median_context', 'mu_median_context', 'rho_median_target', 'mu_median_target', 'lfc', 'p', 'lfc_rand', 'p_rand'])

            rho_mouse = np.concatenate(rho_mouse)
            mu_mouse = np.concatenate(mu_mouse)
            rho_human = np.concatenate(rho_human)
            mu_human = np.concatenate(mu_human)

            lfc_dict[cell_type]['rho_median_context'] = np.median(rho_mouse, axis=0)
            lfc_dict[cell_type]['mu_median_context'] = np.median(mu_mouse, axis=0)
            lfc_dict[cell_type]['rho_median_target'] = np.median(rho_human, axis=0)
            lfc_dict[cell_type]['mu_median_target'] = np.median(mu_human, axis=0)        

            lfc_list = np.concatenate(lfc_list)
            lfc_dict[cell_type]['lfc'] = np.median(lfc_list, axis=0)
            lfc_dict[cell_type]['p'] = np.sum(np.where(np.abs(lfc_list)>lfc_delta, 1, 0), axis=0) / np.shape(lfc_list)[0]

            lfc_list_random = np.concatenate(lfc_list_random)
            lfc_dict[cell_type]['lfc_rand'] = np.median(lfc_list_random, axis=0)
            lfc_dict[cell_type]['p_rand']  = np.sum(np.where(np.abs(lfc_list_random)>lfc_delta, 1, 0), axis=0) / np.shape(lfc_list_random)[0]

        lfc_dict['lfc_delta'] = lfc_delta
        return lfc_dict


    @staticmethod
    def mode_histogram(
            x: np.array, 
        ) -> np.float32:      
        """
        Return the mid-point of the histogram bin with the highest count.
        Helper for .self.similarity_cell_types
        
        Parameters
        ----------   
        x : np.array,
            Array of values for which to calculate the modal value.
        
        Returns
        -------
        mode: np.float32 
            modal value of the empirical distribution
        """
        
        counts, edges = np.histogram(x, bins='fd')
        j = np.argmax(counts)                 
        return (edges[j] + edges[j+1]) / 2.0

    def return_similarity_df(
            self,
            max_sample_targ=2000, 
            max_sample_cont=50,
            scale: str='none',
        ) -> pd.DataFrame:

        """
        Compute and return similarity scores between target and context cell types
        by sampling from latent cell type ditributions and calculating likelihood differences.
        Computes the modal value of the resulting distribution as similarity score.

        Parameters
        ----------
        max_sample_targ : int, default=2000
            Number of samples from the target cell types.
        max_sample_cont : int, default=50
            Number of samples from the context cell types per target cell.
        scale : {'min_max', 'max', 'none'}, default='max'
            Scaling strategy across rows: min-max normalization or max-based inversion.

        Returns
        -------
        df : DataFrame
            Similarity scores with
            - index: target cell types,
            - columns: context cell types.
        """

        context_key = self.context_config['context_key']
        target_key = self.target_config['target_key']

        context_cell_key = self.mdata.mod[context_key].uns['metadata']['cell_key']
        target_cell_key = self.mdata.mod[target_key].uns['metadata']['cell_key']

        if target_cell_key == 'unknown':
            if 'cell_key_transferred' in self.mdata.mod[target_key].uns['metadata'].keys():
                target_cell_key = self.mdata.mod[target_key].uns['metadata']['cell_key_transferred']
                print(f'Use inferred context labels in {target_cell_key} for similarity calculation.')
            else:    
                raise ValueError(f"Target cell labels must be known or transferred from the context dataset via `label_transfer`.") 

        cells_context = self.mdata.mod[context_key].obs[context_cell_key]
        cells_target = self.mdata.mod[target_key].obs[target_cell_key]

        cell_types_context = np.unique(cells_context)
        cell_types_target = np.unique(cells_target)

        df = pd.DataFrame(0, index=cell_types_target, columns=cell_types_context, dtype=float)

        for i in range(len(cell_types_target)):
            for j in range(len(cell_types_context)):
                print('\r{}/{} Similarity calculation for the {}-{} pair'.format(
                    str(i * len(cell_types_context) + j + 1), 
                    str(len(cell_types_target) * len(cell_types_context)),
                    cell_types_target[i],
                    cell_types_context[j]
                ), end=' '*25, flush=True)

                target_ind = np.where(cells_target == cell_types_target[i])[0]
                target_ind = np.random.choice(target_ind, size=min(max_sample_targ, np.shape(target_ind)[0]), replace=False)  

                context_ind = np.where(cells_context == cell_types_context[j])[0]
                context_ind = np.stack([np.random.choice(context_ind, size=min(max_sample_cont, np.shape(context_ind)[0]), replace=False) for k in range(np.shape(target_ind)[0])])

                sim_metric = self.similarity_metric(target_ind, context_ind, b_s=250, b_sc=250, display=False)
                sim_metric_mode = self.mode_histogram(sim_metric.flatten())

                df.loc[cell_types_target[i], cell_types_context[j]] = sim_metric_mode

        if scale == 'min_max':
            df = (df - np.array(df.min(1))[:,np.newaxis]) / (np.array(df.max(1))[:,np.newaxis] - np.array(df.min(1))[:,np.newaxis])
        if scale == 'max':    
            df = np.array(df.max(1))[:,np.newaxis] / df
        if scale == 'none':    
            df = - df

        return  df


    @staticmethod
    def update_param(    
            parameter: float,
            min_value: float,
            max_value: float,
            steps: int
        ) -> float:

        """
        Linearly increment `parameter` toward `max_value` over `steps`.

        Parameters
        ----------
        parameter : float
            Current parameter value.
        min_value : float
            Starting value.
        max_value : float
            Final cap.
        steps : int
            Number of increments until max.

        Returns
        -------
        float
            Updated (and capped) parameter.
        """

        if steps == 0 or min_value == max_value:
            return parameter

        parameter += (max_value - min_value) / steps
        return min(parameter, max_value)


    def train_context(
            self,
            epochs: int = 40,
            raise_beta: bool = True,
            save_model: bool = True,
            train_decoder_only: bool = False,
            save_key: str = ''
        ):

        """
        Pretrain the context scVI model on the context dataset.

        Parameters
        ----------
        epochs : int, default=40
            Number of training epochs.
        raise_beta : bool, default=True
            If True, increase KL weight over initial epochs.
        save_model : bool, default=True
            If True, save model parameters after training.
        train_decoder_only : bool, default=False
            If True, freeze encoders and train only the decoder.
        save_key : str, default=''
            Filename suffix when saving.
        """

        b_s = self.config_dict['b_s']
        n_obs = self.mdata.mod[self.context_config['context_key']].n_obs

        steps_per_epoch = int(n_obs/b_s)

        if self.config_dict['use_lib_enc']:
            progBar = Progress_Bar(epochs, steps_per_epoch, ['nELBO', 'nlog_likeli', 'KL-Div z', 'KL-Div l'])
        else:     
            progBar = Progress_Bar(epochs, steps_per_epoch, ['nELBO', 'nlog_likeli', 'KL-Div z'])

        print(f'Pretraining on the context dataset for {epochs} epochs (= {epochs*steps_per_epoch} iterations).')

        x = torch.from_numpy(self.mdata.mod[self.context_config['context_key']].X.toarray())
        s = torch.from_numpy(self.mdata.mod[self.context_config['context_key']].obsm['batch_label_enc'])
        if self.config_dict['use_lib_enc']:
            lib_mu = torch.from_numpy(self.mdata.mod[self.context_config['context_key']].obs['library_log_mean'].to_numpy())
            lib_sig = torch.from_numpy(self.mdata.mod[self.context_config['context_key']].obs['library_log_std'].to_numpy())

        if not train_decoder_only: 
            self.context_encoder_outer.train()
            if self.config_dict['use_lib_enc']:
                self.context_lib_encoder.train()        
            self.context_encoder_inner.train()
        self.context_decoder.train()
     
        for epoch in range(epochs):
            perm = self.rng.permutation(n_obs)  

            for step in range(steps_per_epoch):         
                self.context_optimizer.zero_grad(set_to_none=True)

                x_batch = self.get_batch(x, step, perm=perm, batch_size=b_s)
                s_batch = self.get_batch(s, step, perm=perm, batch_size=b_s)
                if self.config_dict['use_lib_enc']:
                    lib_mu_batch = self.get_batch(lib_mu, step, perm=perm, batch_size=b_s)
                    lib_sig_batch = self.get_batch(lib_sig, step, perm=perm, batch_size=b_s)

                z_batch, z_kl_div = self.context_encoder_inner(self.context_encoder_outer(x_batch, s_batch)) 
                if self.config_dict['use_lib_enc']:
                    l_batch, l_kl_div = self.context_lib_encoder(x_batch, s_batch, lib_mu_batch, lib_sig_batch)     
                else:    
                    l_batch = x_batch.sum(-1).unsqueeze(-1)
                
                nlog_likeli = self.context_decoder(z_batch, s_batch, l_batch, x_batch)
                nelbo = nlog_likeli + self.context_config['beta'] * z_kl_div 
                if self.config_dict['use_lib_enc']:
                    nelbo = nelbo + self.context_config['beta'] * l_kl_div

                nelbo.backward()
                self.context_optimizer.step() 
                self.context_likeli_hist_dict.append(nlog_likeli.item())
                
                if self.config_dict['use_lib_enc']:
                    progBar.update({'nELBO': nelbo.item(), 'nlog_likeli': nlog_likeli.item(), 'KL-Div z': (self.context_config['beta'] * z_kl_div).item(), 'KL-Div l': (self.context_config['beta'] * l_kl_div).item()})
                else: 
                    progBar.update({'nELBO': nelbo.item(), 'nlog_likeli': nlog_likeli.item(), 'KL-Div z': (self.context_config['beta'] * z_kl_div).item()})

            if raise_beta: 
                self.context_config['beta'] = self.update_param(self.context_config['beta'], self.context_config['beta_start'], self.context_config['beta_max'], self.context_config['beta_epochs_raise'])    


        if not train_decoder_only: 
            self.context_encoder_outer.eval()
            if self.config_dict['use_lib_enc']:
                self.context_lib_encoder.eval()        
            self.context_encoder_inner.eval()
        self.context_decoder.eval()

        if save_model == True:    
            self.save('context',save_key=save_key)  

    def train_target(
            self,
            epochs: int = 40,
            save_model: bool = True,
            raise_beta: bool = True,
            raise_eta: bool = True,
            save_key: str = '',
        ):

        """
        Train the target-side scVI model, optionally aligning to context.

        Parameters
        ----------
        epochs : int, default=40
            Number of training epochs.
        save_model : bool, default=True
            Save parameters after training.
        raise_beta : bool, default=True
            If True, increase KL weight over initial epochs.
        raise_eta : bool, default=True
            If True, increase alignment weight over initial epochs.
        save_key : str, default=''
            Suffix for saved files.
        """
        context_cell_key=self.mdata.mod[self.context_config['context_key']].uns['metadata']['cell_key']

        n_obs = self.mdata.mod[self.target_config['target_key']].n_obs
        k_neigh = self.target_config['k_neigh']
        top_percent = self.target_config['top_percent']

        steps_per_epoch = int(n_obs/self.config_dict['b_s'])

        if self.config_dict['use_lib_enc']:
            progBar = Progress_Bar(epochs, steps_per_epoch, ['nELBO', 'nlog_likeli', 'KL-Div z', 'KL-Div l', 'Align-Term'])
        else:     
            progBar = Progress_Bar(epochs, steps_per_epoch, ['nELBO', 'nlog_likeli', 'KL-Div z', 'Align-Term'])
        print(f'Training on the target dataset for {epochs} epochs (= {epochs*steps_per_epoch} iterations).')

        x = torch.from_numpy(self.mdata.mod[self.target_config['target_key']].X.toarray())

        self.target_encoder_outer.train()
        self.target_encoder_inner.eval()
        if self.config_dict['use_lib_enc']:
            self.target_lib_encoder.train()        
        self.target_decoder.train()
        self.target_encoder_inner.train()

        for epoch in range(epochs):
            perm = self.rng.permutation(n_obs)     

            for step in range(steps_per_epoch): 
                self.target_optimizer.zero_grad(set_to_none=True)

                batch_adata = self.mdata.mod[self.target_config['target_key']][perm[step*self.config_dict['b_s']:(step+1)*self.config_dict['b_s']]]

                x_batch = self.get_batch(x, step, perm=perm)
                s_batch = torch.from_numpy(batch_adata.obsm['batch_label_enc']).to(self.config_dict['device'])      
                if self.config_dict['use_lib_enc']:   
                    lib_mu_batch = torch.from_numpy(batch_adata.obs['library_log_mean'].to_numpy()).to(self.config_dict['device'])          
                    lib_sig_batch = torch.from_numpy(batch_adata.obs['library_log_std'].to_numpy()).to(self.config_dict['device'])  

                inter = self.target_encoder_outer(x_batch, s_batch)

                z_batch, z_kl_div = self.target_encoder_inner(inter)      
                if self.config_dict['use_lib_enc']:           
                    l_batch, l_kl_div = self.target_lib_encoder(x_batch, s_batch, lib_mu_batch, lib_sig_batch)   
                else:    
                    l_batch = x_batch.sum(-1).unsqueeze(-1)
                        
                nlog_likeli = self.target_decoder(z_batch, s_batch, l_batch, x_batch)               
   
                ind_top = np.where(batch_adata.obs['top_percent_'+context_cell_key].to_numpy()<top_percent/100)[0]  
                if np.shape(ind_top)[0] < 1: ind_top = np.reshape(np.random.randint(self.config_dict['b_s']), (1,))

                ind_neigh = batch_adata.obsm['ind_neigh_nns'][ind_top, :k_neigh]
                neigh_mu = torch.from_numpy(self.mdata.mod[self.context_config['context_key']].obsm['z_mu'][ind_neigh]).to(self.config_dict['device'])  
                neigh_sig = torch.from_numpy(self.mdata.mod[self.context_config['context_key']].obsm['z_sig'][ind_neigh]).to(self.config_dict['device'])    
                neigh_z = neigh_mu + neigh_sig * self.target_encoder_inner.sampling_dist.sample(torch.Size([neigh_sig.size(dim=0), neigh_sig.size(dim=1)]))
                  
                s_interl = torch.repeat_interleave(s_batch[ind_top], repeats=k_neigh, dim=0)
                l_interl = torch.repeat_interleave(l_batch[ind_top], repeats=k_neigh, dim=0)
                x_interl = torch.repeat_interleave(x_batch[ind_top], repeats=k_neigh, dim=0)

                outp = self.target_decoder.decode(neigh_z.view(-1, neigh_z.size(-1)), s_interl)

                nlog_likeli_neighbors = self.target_decoder.calc_nlog_likelihood(outp, l_interl, x_interl).reshape(np.shape(ind_top)[0], k_neigh)
                best_pin_for_x = torch.argmin(nlog_likeli_neighbors, dim=1).cpu().numpy()

                if self.config_dict['alignment'] == 'inter':
                    align_target = torch.from_numpy(self.mdata.mod[self.context_config['context_key']].obsm['inter'][batch_adata.obsm['ind_neigh_nns'][ind_top, best_pin_for_x]]).to(self.config_dict['device'])
                    sqerror_align = torch.sum((inter[ind_top] - align_target)**2, dim=-1).mean()

                elif self.config_dict['alignment'] == 'latent':
                    sqerror_align = torch.sum((z_batch[ind_top] - neigh_z[np.arange(len(ind_top)), best_pin_for_x])**2, dim=-1).mean()

                nelbo = self.target_config['beta'] * z_kl_div + nlog_likeli + self.target_config['eta'] * sqerror_align
                if self.config_dict['use_lib_enc']:
                    nelbo = nelbo + self.target_config['beta'] * l_kl_div

                nelbo.backward()
                self.target_optimizer.step() 
                self.target_likeli_hist_dict.append(nlog_likeli.item())
                
                if self.config_dict['use_lib_enc']:
                    progBar.update({'nELBO': nelbo.item(), 'nlog_likeli': nlog_likeli.item(), 'KL-Div z': (self.target_config['beta'] * z_kl_div).item(), 'KL-Div l': (self.target_config['beta'] * l_kl_div).item(), 'Align-Term': (self.target_config['eta'] * sqerror_align).item()})
                else: 
                    progBar.update({'nELBO': nelbo.item(), 'nlog_likeli': nlog_likeli.item(), 'KL-Div z': (self.target_config['beta'] * z_kl_div).item(), 'Align-Term': (self.target_config['eta'] * sqerror_align).item()})

            if raise_beta:       
                self.target_config['beta'] = self.update_param(self.target_config['beta'], self.target_config['beta_start'], self.target_config['beta_max'], self.target_config['beta_epochs_raise'])    
            if raise_eta:
                self.target_config['eta'] = self.update_param(self.target_config['eta'], self.target_config['eta_start'], self.target_config['eta_max'], self.target_config['eta_epochs_raise'])    


        self.target_encoder_outer.eval()
        if self.config_dict['use_lib_enc']:
            self.target_lib_encoder.eval()        
        self.target_decoder.eval()
        self.target_encoder_inner.eval()

        if save_model == True:            
            self.save('target',save_key=save_key)        


    def encode(    
            self,
            x: torch.Tensor,
            s: torch.Tensor,
            encoder_outer: Optional[nn.Module] = None,
            encoder_inner: Optional[nn.Module] = None,
            lib_encoder: Optional[nn.Module] = None
        ) -> Union[
            Tuple[np.ndarray,np.ndarray,np.ndarray],
            Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]
        ]:

        """
        Encode data into biological and/or library latent variables.

        Parameters
        ----------
        x : Tensor, shape (n_cells, n_genes)
            Raw or log-transformed count matrix.
        s : Tensor, shape (n_cells, n_batches)
            One-hot encoded batch labels.
        encoder_outer : nn.Module, optional
            Outer encoder; if None, skips z/inter outputs.
        encoder_inner : nn.Module, optional
            Inner encoder; if None, skips z/inter outputs.
        lib_encoder : nn.Module, optional
            Library encoder; if None, skips l_mu/l_sig outputs.

        Returns
        -------
        Depending on provided encoders:
        (z_mu, z_sig, inter) if `lib_encoder` is None.
        (l_mu, l_sig) if only `lib_encoder` is provided.
        (z_mu, z_sig, inter, l_mu, l_sig) if all provided.
        """

        steps = int(np.ceil(x.size(0)/self.config_dict['b_s']+1e-10))        

        if encoder_outer != None and encoder_inner != None:
            encoder_outer.eval()
            encoder_inner.eval()
            z_mu_list, z_sig_list, inter_list = [], [], []
        if lib_encoder != None:
            lib_encoder.eval()
            l_mu_list, l_sig_list = [], []    
        
        with torch.no_grad():
            tic = time.time()
            for step in range(steps):   
                if time.time() - tic > 0.5:
                    tic = time.time()
                    print('\rCalculate latent variables. Step {}/{} '.format(str(step), str(steps)), end='', flush=True) 

                x_batch = self.get_batch(x, step)
                s_batch = self.get_batch(s, step)

                if encoder_outer != None and encoder_inner != None:
                    inter = encoder_outer(x_batch, s_batch) 
                    z_mu, z_log_sig = encoder_inner.encode(inter)             

                    z_mu_list.append(z_mu.cpu().numpy())
                    z_sig_list.append(z_log_sig.exp().cpu().numpy())                    
                    inter_list.append(inter.cpu().numpy())

                if lib_encoder != None:
                    if self.config_dict['use_lib_enc']:
                        l_mu, l_log_sig = lib_encoder.encode(x_batch, s_batch)  
                    else:         
                        l_mu, l_log_sig = x_batch.sum(-1).log(), torch.zeros_like(x_batch.sum(-1))
          
                    l_mu_list.append(l_mu.cpu().numpy())
                    l_sig_list.append(l_log_sig.exp().cpu().numpy())                        

            if encoder_outer != None and encoder_inner != None and lib_encoder == None:
                return np.concatenate(z_mu_list), np.concatenate(z_sig_list), np.concatenate(inter_list)

            elif encoder_outer == None and encoder_inner == None and lib_encoder != None:
                return np.concatenate(l_mu_list), np.concatenate(l_mu_list)

            elif encoder_outer != None and encoder_inner != None and lib_encoder != None:
                return np.concatenate(z_mu_list), np.concatenate(z_sig_list), np.concatenate(inter_list), np.concatenate(l_mu_list), np.concatenate(l_sig_list)          

    def get_representation(
            self,
            eval_model: str,
            save_intermediate: bool = False,
            save_libsize: bool = False
        ): 

        """
        Compute and store biological latent and/or library latent representations for a dataset.

        Parameters
        ----------
        eval_model : {'context','target'}
            Which dataset to encode.
        save_intermediate : bool, default=False
            If True, store the outer encoder output in `.obsm['inter']`.
        save_libsize : bool, default=False
            If True, store library mean/log-std in `.obsm['l_mu']`/`['l_sig']`.
        """

        if eval_model == 'target':
            dataset_key = self.target_config['target_key']
            encoder_outer = self.target_encoder_outer
            encoder_inner = self.target_encoder_inner
            lib_encoder = self.target_lib_encoder

        elif eval_model == 'context':
            dataset_key = self.context_config['context_key']
            encoder_outer = self.context_encoder_outer
            encoder_inner = self.context_encoder_inner
            lib_encoder = self.context_lib_encoder

        x = torch.from_numpy(self.mdata.mod[dataset_key].X.toarray())
        s = torch.from_numpy(self.mdata.mod[dataset_key].obsm['batch_label_enc'])

        if save_libsize == False:
            if self.config_dict['alignment'] == 'inter':
                z_mu, z_sig, inter = self.encode(x, s, encoder_outer=encoder_outer, encoder_inner=encoder_inner)
            elif self.config_dict['alignment'] == 'latent':
                z_mu, z_sig, inter  = self.encode(x, s, encoder_outer=encoder_outer, encoder_inner=encoder_inner)

        elif save_libsize == True:
            if self.config_dict['alignment'] == 'inter':
                z_mu, z_sig, inter, l_mu, l_sig = self.encode(x, s, encoder_outer=encoder_outer, encoder_inner=encoder_inner, lib_encoder=lib_encoder)
            elif self.config_dict['alignment'] == 'latent':
                z_mu, z_sig, inter, l_mu, l_sig  = self.encode(x, s, encoder_outer=encoder_outer, encoder_inner=encoder_inner, lib_encoder=lib_encoder)  

        self.mdata.mod[dataset_key].obsm['z_mu'] = z_mu
        self.mdata.mod[dataset_key].obsm['z_sig'] = z_sig
           
        if save_intermediate:
            self.mdata.mod[dataset_key].obsm['inter'] = inter

        if save_libsize:
            self.mdata.mod[dataset_key].obsm['l_mu'] = l_mu
            self.mdata.mod[dataset_key].obsm['l_sig'] = l_sig


def color_str(value, mode):
    text = str(value)
    if mode == "context":
        return f"\033[38;5;208m{text}\033[0m"
    elif mode == "target":
        return f"\033[38;5;135m{text}\033[0m"


class Progress_Bar():
    """
    A console progress bar that tracks multiple metrics over training iterations of scSpecies.

    Parameters
    ----------
    epochs : int
        Total number of epochs.
    steps_per_epoch : int
        Number of steps (batches) in each epoch.
    metrics : list of str
        Names of metrics to track (e.g., ['nELBO', 'nlog_likeli']).
    avg_over_n_steps : int, optional
        Number of recent steps over which to average metric values for display.
    sleep_print : float, optional
        Interval in seconds between console updates.
    """

    def __init__(
            self, 
            epochs: int, 
            steps_per_epoch: int, 
            metrics: List[str], 
            avg_over_n_steps: int = 100, 
            sleep_print: float = 0.5
            ):

        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch 
        self.total_steps = self.epochs * steps_per_epoch
        self.remaining_steps = self.epochs * steps_per_epoch
        self.avg_over_n_steps = avg_over_n_steps
        self.tic = time.time() 
        self.sleep_print = sleep_print
        self.iteration = 0
        self.metrics = metrics
        self.time_metrics = ['Progress', 'ETA', 'Epoch', 'Iteration', 'ms/Iteration']
        
        self.dict = {
                    'Progress' : "0.000%",
                    'ETA' : 0.0,
                    'Epoch' : int(1),
                    'Iteration' : int(0),
                    'ms/Iteration' : 0.0,
                    'time': [time.time()]
                    }

        self.dict.update({metric: [] for metric in metrics})
        self.dict.update({metric+' last ep': [] for metric in metrics})
        self.dict.update({metric+' impr': 0.0 for metric in metrics})
        
    @staticmethod
    def format_number(number, min_length):
        decimal_count = len(str(number).split('.')[0])  
        decimal_places = max(min_length - decimal_count, 0) 

        formatted_number = "{:.{}f}".format(number, decimal_places)
        return formatted_number
    
    def ret_sign(self, number, min_length):
        if number > 0.0:
            sign_str = '\033[92m{}\033[00m'.format("+" + self.format_number(np.abs(number), min_length))
        elif number < 0.0:
            sign_str = '\033[91m{}\033[00m'.format("-" + self.format_number(np.abs(number), min_length))
        else:
            sign_str = '---'
        return  sign_str      

    def update(self, values):   
        self.remaining_steps -= 1   
        for key, value in values.items():
            self.dict[key].append(value) 
            
        if self.dict['Iteration'] == 1:
            for key, value in values.items():
                self.dict[key+' last ep'].append(value) 
             
        self.dict['Iteration'] += 1
        
        epoch = int(np.ceil(self.dict['Iteration'] / self.steps_per_epoch))

        if self.dict['Epoch'] < epoch:
            for key in self.metrics:
                self.dict[key+' last ep'].append(np.mean(self.dict[key][-self.steps_per_epoch:]))
                self.dict[key+' impr'] = self.dict[key+' last ep'][-2] - self.dict[key+' last ep'][-1]
            self.dict['Epoch'] = epoch
     
        self.dict['time'].append(time.time())
  
        avg_steps = np.min((self.dict['Iteration'], self.avg_over_n_steps))
        avg_time = (self.dict['time'][-1] - self.dict['time'][-avg_steps-1]) / avg_steps 
        
        self.dict['ETA'] = timedelta(seconds=int(self.remaining_steps * avg_time))         
        self.dict['ms/Iteration'] = self.format_number(avg_time*1000.0, 4)
        self.dict['Progress'] = self.format_number(100.0 * self.dict['Iteration'] / self.total_steps, 3)+'%'

        if time.time() - self.tic > self.sleep_print:
            metric_string =  [f'\033[95m{key}\033[00m: {self.dict[key]}' for key in self.time_metrics]       
            metric_string += [f'\033[33m{key}\033[00m: {self.format_number(np.mean(self.dict[key][-avg_steps:]), 5)} ({self.ret_sign(self.dict[key+" impr"], 4)})' for key in self.metrics]               
            metric_string =  "\033[96m - \033[00m".join(metric_string)
            print(f"\r{metric_string}.           ", end='', flush=True)   
            self.tic = time.time()   



def neighbors_workaround(
        adata: ad.AnnData,
        use_rep: Optional[str] = None,
        n_neighbors: int = 15,
        metric: str = 'euclidean'
    ) -> ad.AnnData:

    """
    Compute the k-nearest-neighbors graph manually and store it in `adata`.
    Replacement for sc.pp.neighbors on M1/M2 chips to avoid kernel crashes.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object.
    use_rep : str
        Key in `adata.obsm` to use for neighbor search (e.g. 'X_pca'),
        or None to use `adata.X`.
    n_neighbors : int
        Number of nearest neighbors to use.
    metric : str or None
        Distance metric to use (default 'euclidean').

    Returns
    -------
    AnnData
        The same `adata`, with:
          - obsp['distances']       : sparse matrix of neighbor distances
          - obsp['connectivities']  : sparse binary connectivity matrix
          - uns['neighbors']        : dict of params & key names
    """

    if use_rep is None:
        X = adata.X
        rep_key = 'X'
    else:
        X = adata.obsm[use_rep]
        rep_key = use_rep

    if hasattr(X, "toarray"):
        X = X.toarray()

    n_obs = X.shape[0]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(X)
    distances, indices = nbrs.kneighbors(X)

    rows = np.repeat(np.arange(n_obs), n_neighbors)
    cols = indices.flatten()
    data_dist = distances.flatten()
    dist_matrix = csr_matrix((data_dist, (rows, cols)), shape=(n_obs, n_obs))

    data_conn = np.ones_like(data_dist)
    conn_matrix = csr_matrix((data_conn, (rows, cols)), shape=(n_obs, n_obs))

    adata.obsp['distances'] = dist_matrix
    adata.obsp['connectivities'] = conn_matrix
    adata.uns['neighbors'] = {
        'params': {
            'n_neighbors': n_neighbors,
            'method': 'umap',
            'use_rep': rep_key,
            'metric': metric
        },
        'distances_key': 'distances',
        'connectivities_key': 'connectivities'
    }

    return adata            