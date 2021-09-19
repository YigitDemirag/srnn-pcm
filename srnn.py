""" E-prop implementation with LIF neurons. Modified to support PCM crossbar array.

e-prop paper: Bellec, G. et al. A solution to the learning dilemma for recurrent networks of spiking neurons. Nat Commun 11, 3625 (2020).
Author: Yigit Demirag, Melika Payvand, 2020 @ NCS, INI of ETH Zurich and UZH
"""
import torch
from torch import nn
import torch.nn.functional as F
from xbar import XBar
import numpy as np
EPS = 5e-4

class SRNN(nn.Module):
    ''' Implements a SRNN layer.
    '''
    def __init__(self, inp_dim, out_dim, n_rec, thr, tau_rec, tau_out,
                 lr_inp, lr_rec, lr_out, w_init_gain, n_t, n_b, gamma, 
                 dt, reg, f0, xbar, xbar_n, perf, xbar_res, xbar_scale, 
                 prob_scale, grad_thr, method, device):

        super(SRNN, self).__init__()
        self.inp_dim     = inp_dim
        self.n_rec       = n_rec
        self.out_dim     = out_dim
        self.thr         = thr
        self.alpha       = np.exp(-dt / tau_rec)
        self.kappa       = np.exp(-dt / tau_out)
        self.lr_inp      = lr_inp
        self.lr_rec      = lr_rec
        self.lr_out      = lr_out
        self.n_t         = n_t
        self.n_b         = n_b
        self.gamma       = gamma
        self.w_init_gain = w_init_gain
        self.b_o         = 0.0
        self.reg         = reg
        self.f0          = f0
        self.dt          = dt
        self.xbar        = xbar
        self.xbar_res    = xbar_res
        self.xbar_scale  = xbar_scale
        self.prob_scale  = prob_scale
        self.grad_thr    = grad_thr
        self.xbar_n      = xbar_n
        self.perf        = perf
        self.T0          = 38.6
        self.method      = method
        self.device      = device

        # Parameters
        if self.xbar: # Crossbar array
          self.inp_xbar = XBar(G0=self.w_init_gain, N=self.xbar_n, size=(self.n_rec, self.inp_dim), res=self.xbar_res, scale=self.xbar_scale, prob_scale=self.prob_scale, device=self.device)
          self.rec_xbar = XBar(G0=self.w_init_gain, N=self.xbar_n, size=(self.n_rec, self.n_rec), res=self.xbar_res, scale=self.xbar_scale, prob_scale=self.prob_scale, device=self.device)
          self.out_xbar = XBar(G0=self.w_init_gain, N=self.xbar_n, size=(self.out_dim, self.n_rec), res=self.xbar_res, scale=self.xbar_scale, prob_scale=self.prob_scale, device=self.device)
          
          # Feedback path
          #self.w_fb = torch.normal(self.w_init_gain, self.w_init_gain*0.1, (self.out_dim, self.n_rec)).to(self.device)
          self.w_fb = nn.Parameter(torch.Tensor(out_dim, n_rec))      
          torch.nn.init.kaiming_normal_(self.w_fb)
          self.w_fb.data = self.w_init_gain * self.w_fb.data

        else: # No crossbar array
          self.w_inp = nn.Parameter(torch.Tensor(self.n_rec, self.inp_dim))
          self.w_rec = nn.Parameter(torch.Tensor(self.n_rec, self.n_rec))
          self.w_out = nn.Parameter(torch.Tensor(self.out_dim, self.n_rec))
          self.w_fb = nn.Parameter(torch.Tensor(self.out_dim, self.n_rec))
          self.initialize_weights(self.w_init_gain) 

        # Accumulator (Mixed-precision training)
        self.inp_acc = torch.zeros((self.n_rec, self.inp_dim), device=self.device)
        self.rec_acc = torch.zeros((self.n_rec, self.n_rec), device=self.device)
        self.out_acc = torch.zeros((self.out_dim, self.n_rec), device=self.device)

    def initialize_weights(self, gain):
        ''' Initialize input, recurrent and output weights if no crossbar array is used.

        Args:
            gain: scaling factor for the weights
        '''
        torch.nn.init.kaiming_normal_(self.w_inp)
        self.w_inp.data = gain * self.w_inp.data 
        torch.nn.init.kaiming_normal_(self.w_rec)
        self.w_rec.data = gain * self.w_rec.data 
        torch.nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain * self.w_out.data
        torch.nn.init.kaiming_normal_(self.w_fb)
        self.w_fb.data = gain * self.w_fb.data

    def init_CPU_states(self, n_b, n_t, n_rec, out_dim):
        ''' Initialize network states

        Args:
            n_b: batch size
            n_t: number of time steps
            n_rec: number of recurrent units
            out_dim: output dimension
        
        '''
        # Hidden state
        self.v = torch.zeros(n_t, n_b, n_rec).to(self.device)
        # Visible state
        self.z = torch.zeros(n_t, n_b, n_rec).to(self.device)
        self.vo = torch.zeros(n_t, n_b, out_dim).to(self.device)
        # Weight gradients
        self.w_inp_grad = torch.zeros(self.n_rec, self.inp_dim).to(self.device)
        self.w_rec_grad = torch.zeros(self.n_rec, self.n_rec).to(self.device)
        self.w_out_grad = torch.zeros(self.out_dim, self.n_rec).to(self.device)

    def reset_diagonal(self, tp):
        ''' Reset diagonal elements of recurrent weights to zero (or the lowest conductance)
        '''
        if self.xbar:
            reset_mask = torch.eye(self.n_rec, self.n_rec, device=self.device).repeat(2, self.xbar_n, 1, 1).bool()
            self.rec_xbar.reset(tp, mask=reset_mask, G0=0.1)
        else:
            self.w_rec *= (1 - torch.eye(self.n_rec, self.n_rec, device=self.device))

    def forward(self, x, tp):
        ''' Forward pass of the network

        Args:
            x  : input spike pattern
            tp : time point
        
        Returns:
            self.vo : Output units membrane voltages
        '''
        self.n_b, self.n_t = x.shape[0], x.shape[2]

        self.init_CPU_states(self.n_b, self.n_t, self.n_rec, self.out_dim)
        self.reset_diagonal(tp)

        for t in range(0, self.n_t - 1):
            curr_time = torch.tensor(tp + t * self.dt + EPS, dtype=torch.double) # current time
            if self.xbar:
                # Differential read of XBar
                inp_weight = torch.diff(-torch.sum(self.inp_xbar.read(t=curr_time, T0=self.T0, perf=self.perf),dim=1), dim=0).squeeze(0).t()
                rec_weight = torch.diff(-torch.sum(self.rec_xbar.read(t=curr_time, T0=self.T0, perf=self.perf),dim=1), dim=0).squeeze(0).t()
                out_weight = torch.diff(-torch.sum(self.out_xbar.read(t=curr_time, T0=self.T0, perf=self.perf),dim=1), dim=0).squeeze(0).t()
            else:
                inp_weight = self.w_inp.t()
                rec_weight = self.w_rec.t()
                out_weight = self.w_out.t()

            self.v[t + 1]  = (self.alpha * self.v[t] + torch.mm(self.z[t], rec_weight) + torch.mm(x[:, :, t], inp_weight)) - self.z[t] * self.thr
            self.z[t + 1]  = (self.v[t + 1] > self.thr).float()
            self.vo[t + 1] = self.kappa * self.vo[t] + torch.mm(self.z[t + 1], out_weight) + self.b_o
        return self.vo

    def calc_traces(self, x):
        ''' Calculate the eligibility traces inside the network

        Args:
            x  : input spike pattern
        
        '''
        h = self.gamma * torch.max(torch.zeros_like(self.v), 1 - torch.abs((self.v - self.thr) / self.thr)) # n_t, n_b, n_rec
        alpha_conv = torch.tensor([self.alpha ** (self.n_t - i - 1) for i in range(self.n_t)]).float().view(1, 1, -1).to(self.device) # 1, 1, n_t

        self.trace_in = F.conv1d(x, alpha_conv.expand(self.inp_dim, -1, -1),
                                 padding=self.n_t, groups=self.inp_dim)[:, :, 1:self.n_t + 1].unsqueeze(1).expand(-1, self.n_rec, -1, -1) #n_b, n_rec, inp_dim, n_t
        self.trace_in = torch.einsum('tbr,brit->brit', h, self.trace_in)  # n_b, n_r, inp_dim, n_t

        self.trace_rec = F.conv1d(self.z.permute(1, 2, 0), alpha_conv.expand(self.n_rec, -1, -1),
                                  padding=self.n_t, groups=self.n_rec)[:, :, :self.n_t].unsqueeze(1).expand(-1, self.n_rec, -1, -1)

        self.trace_rec = torch.einsum('tbr,brit->brit', h, self.trace_rec)

        self.fr = torch.sum(self.z, dim=(0,1))/ (self.n_t * self.dt) # Firing rate per neuron

        self.reg_term = (self.fr - self.f0)

        kappa_conv = torch.tensor([self.kappa ** (self.n_t - i - 1) for i in range(self.n_t)]).float().view(1, 1, -1).to(self.device)

        self.trace_out = F.conv1d(self.z.permute(1, 2, 0), kappa_conv.expand(self.n_rec, -1, -1),
                                  padding=self.n_t, groups=self.n_rec)[:, :, 1:self.n_t + 1]

        self.trace_in = F.conv1d(self.trace_in.reshape(self.n_b, self.inp_dim * self.n_rec, self.n_t),
                                 kappa_conv.expand(self.inp_dim * self.n_rec, -1, -1),
                                 padding=self.n_t, groups=self.inp_dim * self.n_rec)[:, :, 1:self.n_t + 1].reshape(self.n_b, self.n_rec, self.inp_dim, self.n_t)
        self.trace_rec = F.conv1d(self.trace_rec.reshape(self.n_b, self.n_rec * self.n_rec, self.n_t),
                                  kappa_conv.expand(self.n_rec * self.n_rec, -1, -1),
                                  padding=self.n_t, groups=self.n_rec * self.n_rec)[:, :, 1:self.n_t + 1].reshape(self.n_b, self.n_rec, self.n_rec, self.n_t)


    def acc_gradient(self, err, targeted_fr=True):
        ''' Accumulate the gradient in the network

        Args:
            err         : error between the target signal and the output unit membrane potential
            targeted_fr : True if weight regularization is used
        '''
        L_loss = torch.einsum('tbo,or->brt', err, self.w_fb)

        if not targeted_fr:
            L_reg  = torch.mean(self.trace_rec, dim=2)
        else:
            L_reg = self.reg_term.expand(self.n_t, self.n_rec).unsqueeze(0).permute(0, 2, 1) #torch.mean(self.reg_term) for MEAN fr

        L = L_loss + self.reg * L_reg

        # Weight gradient updates
        self.w_inp_grad += torch.sum(torch.einsum('bxt,bxyt->xyt', L, self.trace_in), dim=2).clamp(-100, 100)
        self.w_rec_grad += torch.sum(torch.einsum('bxt,bxyt->xyt', L, self.trace_rec), dim=2).clamp(-100, 100)
        self.w_out_grad += torch.einsum('tbo,brt->or', err, self.trace_out).clamp(-100, 100)


    def do_weight_update(self, tp):
        ''' Update the weights in the network

        Args:
            tp : time point
        '''
        if (tp>10000 and tp <10050) or (tp>20000 and tp <20050):
            refresh = True
            print('Refreshing')
        else:
            refresh = False


        assert self.method in ['sign', 'stochastic', 'multi-mem', 'mixed-precision', 'vanilla', 'accumulator'], "Invalid update method."
        
        if self.xbar: # PCM Weights
            if self.method =='sign':
                ''' Update PCM devices according to sign of the gradient
                '''
                # Weight update - Apply gradual SET to differential memristors (W -= sign(dW)) if dW > thr
                self.inp_xbar.write(tp=tp, mask=torch.stack((self.w_inp_grad < -self.grad_thr, self.w_inp_grad > self.grad_thr), dim=0).unsqueeze(1).expand(-1,self.xbar_n,-1,-1), perf=self.perf)
                self.rec_xbar.write(tp=tp, mask=torch.stack((self.w_rec_grad < -self.grad_thr, self.w_rec_grad > self.grad_thr), dim=0).unsqueeze(1).expand(-1,self.xbar_n,-1,-1), perf=self.perf)
                self.out_xbar.write(tp=tp, mask=torch.stack((self.w_out_grad < -self.grad_thr, self.w_out_grad > self.grad_thr), dim=0).unsqueeze(1).expand(-1,self.xbar_n,-1,-1), perf=self.perf)
            
            elif self.method =='stochastic':
                ''' Gradual SET pulse as a function of gradient (not accumulated) amplitude.
                '''
                # Current differential conductance of XBar (perfect estimation)
                inp_G_est = (self.inp_xbar.G[0] - self.inp_xbar.G[1]).squeeze(0)
                rec_G_est = (self.rec_xbar.G[0] - self.rec_xbar.G[1]).squeeze(0)
                out_G_est = (self.out_xbar.G[0] - self.out_xbar.G[1]).squeeze(0)

                # Target G
                inp_G_tar = inp_G_est - self.lr_inp * self.w_inp_grad * self.inp_xbar.xbar_scale
                rec_G_tar = rec_G_est - self.lr_rec * self.w_rec_grad * self.rec_xbar.xbar_scale
                out_G_tar = out_G_est - self.lr_out * self.w_out_grad * self.out_xbar.xbar_scale
                
                self.inp_xbar.target_write(tp=tp, G_target=inp_G_tar, G_curr_est = inp_G_est, refresh=False, perf=self.perf, method='stochastic')
                self.rec_xbar.target_write(tp=tp, G_target=rec_G_tar, G_curr_est = rec_G_est, refresh=False, perf=self.perf, method='stochastic')
                self.out_xbar.target_write(tp=tp, G_target=out_G_tar, G_curr_est = out_G_est, refresh=False, perf=self.perf, method='stochastic')

            elif self.method =='multi-mem':
                ''' Multiple memristor per synapse, updated in turns.
                '''
                # Current differential conductance of XBar (perfect estimation)
                inp_G_est = (torch.mean(self.inp_xbar.G,1)[0] - torch.mean(self.inp_xbar.G,1)[1]).squeeze(0)
                rec_G_est = (torch.mean(self.rec_xbar.G,1)[0] - torch.mean(self.rec_xbar.G,1)[1]).squeeze(0)
                out_G_est = (torch.mean(self.out_xbar.G,1)[0] - torch.mean(self.out_xbar.G,1)[1]).squeeze(0)

                # Target G
                inp_G_tar = inp_G_est - self.lr_inp * self.w_inp_grad * self.inp_xbar.xbar_scale
                rec_G_tar = rec_G_est - self.lr_rec * self.w_rec_grad * self.rec_xbar.xbar_scale
                out_G_tar = out_G_est - self.lr_out * self.w_out_grad * self.out_xbar.xbar_scale
                
                self.inp_xbar.target_write(tp=tp, G_target=inp_G_tar, G_curr_est = inp_G_est, refresh=refresh, perf=self.perf, method='multi-mem')
                self.rec_xbar.target_write(tp=tp, G_target=rec_G_tar, G_curr_est = rec_G_est, refresh=refresh, perf=self.perf, method='multi-mem')
                self.out_xbar.target_write(tp=tp, G_target=out_G_tar, G_curr_est = out_G_est, refresh=refresh, perf=self.perf, method='multi-mem')

            elif self.method =='mixed-precision':
                ''' Accumulate gradient digitally and apply pulses as in mixed-precision paper
                ''' 
                eps = 0.125 # Step for W in [-1,1]
                self.inp_acc += self.lr_inp * self.w_inp_grad
                self.rec_acc += self.lr_rec * self.w_rec_grad
                self.out_acc += self.lr_out * self.w_out_grad

                grad_inp_quant = torch.floor_divide(self.inp_acc, eps) * eps # (100,4) in 0.125 steps [-1,1]
                grad_rec_quant = torch.floor_divide(self.rec_acc, eps) * eps # (100, 100)
                grad_out_quant = torch.floor_divide(self.out_acc, eps) * eps

                self.inp_acc -= grad_inp_quant
                self.rec_acc -= grad_rec_quant
                self.out_acc -= grad_out_quant

                # Current differential conductance of XBar (perfect estimation)
                inp_G_est = (self.inp_xbar.G[0] - self.inp_xbar.G[1]).squeeze(0)
                rec_G_est = (self.rec_xbar.G[0] - self.rec_xbar.G[1]).squeeze(0)
                out_G_est = (self.out_xbar.G[0] - self.out_xbar.G[1]).squeeze(0)

                # Target G
                inp_G_tar = inp_G_est - grad_inp_quant * self.inp_xbar.xbar_scale
                rec_G_tar = rec_G_est - grad_rec_quant * self.rec_xbar.xbar_scale
                out_G_tar = out_G_est - grad_out_quant * self.out_xbar.xbar_scale

                self.inp_xbar.target_write(tp=tp, G_target=inp_G_tar, G_curr_est = inp_G_est, refresh=False, perf=self.perf, method='mixed-precision')
                self.rec_xbar.target_write(tp=tp, G_target=rec_G_tar, G_curr_est = rec_G_est, refresh=False, perf=self.perf, method='mixed-precision')
                self.out_xbar.target_write(tp=tp, G_target=out_G_tar, G_curr_est = out_G_est, refresh=False, perf=self.perf, method='mixed-precision')

        else: # Digital Weights
            if self.method == 'accumulator':
                eps = 0.125 # Step for W in [-1,1] (Dynamic range=8 for PCM)
                self.inp_acc += self.lr_inp * self.w_inp_grad
                self.rec_acc += self.lr_rec * self.w_rec_grad
                self.out_acc += self.lr_out * self.w_out_grad

                grad_inp_quant = torch.floor_divide(self.inp_acc, eps) * eps
                grad_rec_quant = torch.floor_divide(self.rec_acc, eps) * eps
                grad_out_quant = torch.floor_divide(self.out_acc, eps) * eps

                self.inp_acc -= grad_inp_quant
                self.rec_acc -= grad_rec_quant
                self.out_acc -= grad_out_quant

                self.w_inp.data = torch.clamp(self.w_inp.data - grad_inp_quant, -1, 1)
                self.w_rec.data = torch.clamp(self.w_rec.data - grad_rec_quant, -1, 1)
                self.w_out.data = torch.clamp(self.w_out.data - grad_out_quant, -1, 1)
       
            elif self.method == 'vanilla':
                ''' Vanilla e-prop, only restriction is weights are clipped between -1,1.
                '''
                self.w_inp.data = torch.clamp(self.w_inp.data - self.lr_inp  * self.w_inp_grad, -1, 1)
                self.w_rec.data = torch.clamp(self.w_rec.data - self.lr_rec  * self.w_rec_grad, -1, 1)
                self.w_out.data = torch.clamp(self.w_out.data - self.lr_out  * self.w_out_grad, -1, 1)
