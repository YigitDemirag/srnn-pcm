""" Trainer for e-prop supporting PCM Xbar.

Author: Yigit Demirag, Melika Payvand, 2020 @ NCS, INI of ETH Zurich and UZH
"""
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from srnn import SRNN

class Sinusoids(Dataset):
    ''' Implements a target signal dataset by summing up four sinusoids whose phases and amplitudes are randomly generated.
    '''
    def __init__(self, seq_length=1000, num_samples=2, num_inputs=80, input_freq=50):
        ''' Initialize the dataset.

        Args:
          seq_length  : (ms) length of the input sequence
          num_samples : number of samples
          num_inputs  : input dimension
          input_freq  : (Hz) Poisson input spike rate

        '''
        self.seq_length   = seq_length
        self.num_inputs   = num_inputs
        self.num_samples  = num_samples
        self.freq_list    = torch.tensor([1, 2, 3, 5]) # (Hz) frequency of the sinusoids for target signal
        self.dt           = 1e-3 # (s) simulation timestep 
        self.t            = torch.arange(0, seq_length*self.dt, self.dt) # (s) time vector
        self.inp_freq     = input_freq 

        # Random input
        self.x = (torch.rand(self.num_samples, self.num_inputs, self.seq_length) < self.dt * self.inp_freq).float()

        # Randomized output amplitude and phase
        amplitude_list = torch.FloatTensor(self.num_samples, len(self.freq_list)).uniform_(0.5, 2)
        phase_list = torch.FloatTensor(self.num_samples, len(self.freq_list)).uniform_(0, 2 * math.pi)

        # Normalized sum of sinusoids
        self.y = torch.zeros(self.num_samples, self.seq_length)
        for i in range(self.num_samples):
          summed_sinusoid = sum([amplitude_list[i, ix] * torch.sin(2*math.pi*f*self.t + phase_list[i, ix]) for ix, f in enumerate(self.freq_list)])
          self.y[i, :] = summed_sinusoid/torch.max(torch.abs(summed_sinusoid))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]

def train(seed, inp_dim, out_dim, n_rec, thr, tau_rec, tau_out,
          lr_inp, lr_rec, lr_out, w_init_gain, n_t, n_b, gamma, dt, reg, f0,
          xbar, xbar_n, perf, xbar_res, xbar_scale, prob_scale, grad_thr, method, cuda, epochs):

    # fix seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # data Loader
    sinusoid_dataset = Sinusoids(seq_length=n_t, num_samples=6, num_inputs=inp_dim, input_freq=50)

    # parameters
    train_percentage = 50
    batch_size = 1

    train_size = int(len(sinusoid_dataset) * train_percentage/100)
    train_set, _ = random_split(sinusoid_dataset, [train_size, len(sinusoid_dataset)-train_size])
    train_data = DataLoader(train_set, batch_size, shuffle=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")

    srnn = SRNN(inp_dim, out_dim, n_rec, thr, tau_rec,
                tau_out, lr_inp, lr_rec, lr_out, w_init_gain, n_t, n_b,
                gamma, dt, reg, f0, xbar, xbar_n, perf,
                xbar_res, xbar_scale, prob_scale, grad_thr, method, device).to(device)

    mse_loss = nn.MSELoss()

    tp=0.; T0=38.6
    for epoch in range(epochs):
      srnn.eval()
      with torch.no_grad():
        for _, (x, y) in enumerate(train_data):
            x, y = x.to(device), y.to(device)
            y = y.permute(1,0).unsqueeze(-1)
            yhat = srnn(x, tp=tp) 
            srnn.calc_traces(x)
            srnn.acc_gradient(yhat - y)

            # next write after T0 seconds
            tp = tp + T0

            # weight update - Apply gradual SETs to differential memristors
            srnn.do_weight_update(tp=tp)

      # report
      if epoch%10 == 0:
        print(f'Epoch [{epoch}] - Loss :{mse_loss(yhat, y).item():.4f}')

        # stop the training if there is no WRITE update to recurrent layer
        if srnn.xbar and torch.sum(srnn.rec_xbar.count).cpu() < 2 and epoch == 50:
            break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--inp_dim', type=int, default=100, help='Input layer dimension')
    parser.add_argument('--out_dim', type=int, default=1, help='Output layer dimension')
    parser.add_argument('--n_rec', type=int, default=100, help='Number of recurrent units')
    parser.add_argument('--thr', type=float, default=0.1, help='Firing threshold')
    parser.add_argument('--tau_rec', type=float, default=30e-3, help='Recurrent units membrane leakage time constant (s)')
    parser.add_argument('--tau_out', type=float, default=30e-3, help='Output units membrane leakage time constant (s)')
    parser.add_argument('--lr_inp', type=float, default=2e-5, help='Learning rate for input layer')
    parser.add_argument('--lr_rec', type=float, default=5e-5, help='Learning rate for recurrent layer')
    parser.add_argument('--lr_out', type=float, default=2e-5, help='Learning rate for output layer')
    parser.add_argument('--w_init_gain', type=float, default=0.1, help='Weight initialization factor')
    parser.add_argument('--n_t', type=int, default=1000, help='Target signal duration (ms)')
    parser.add_argument('--n_b', type=int, default=1, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.3, help='Surrogate derivative parameter')
    parser.add_argument('--dt', type=float, default=1e-3, help='Simulation timestep (s)')
    parser.add_argument('--reg', type=float, default=1e-4, help='Firing rate regularization factor')
    parser.add_argument('--f0', type=float, default=12, help='Target firing rate')
    parser.add_argument("--xbar", default=False, type=lambda s: s.lower() == 'true', help='Enable PCM XBar simulation')
    parser.add_argument('--xbar_n', type=int, default=1, help='Number of positive and negative memristor pairs per synapse')
    parser.add_argument("--perf", default=False, type=lambda s: s.lower() == 'true', help='Enable performance mode')
    parser.add_argument('--xbar_res', type=int, default=3, help='Ideal memory device resolution in perf-mode')
    parser.add_argument('--xbar_scale', type=float, default=1, help='Scaling factor of matrix-vector-multiply operation')
    parser.add_argument('--prob_scale', type=float, default=1, help='Probability factor for stochastic update method')
    parser.add_argument('--grad_thr', type=float, default=0, help='Gradient threshold for sign-symmetry method')
    parser.add_argument('--method', type=str, default='vanilla', help='E-prop training methods i.e., sign, stochastic, multi-mem, mixed-precision, accumulator, vanilla')
    parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA for GPU training')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    args = parser.parse_args()

    train(seed=args.seed,
          inp_dim=args.inp_dim,
          out_dim=args.out_dim,
          n_rec=args.n_rec,
          thr=args.thr,
          tau_rec=args.tau_rec,
          tau_out=args.tau_out,
          lr_inp=args.lr_inp,
          lr_rec=args.lr_rec,
          lr_out=args.lr_out,
          w_init_gain=args.w_init_gain,
          n_t=args.n_t,
          n_b=args.n_b,
          gamma=args.gamma,
          dt=args.dt,
          reg=args.reg,
          f0=args.f0,
          xbar=args.xbar,
          xbar_n=args.xbar_n,
          perf=args.perf,
          xbar_res=args.xbar_res,
          xbar_scale=args.xbar_scale,
          prob_scale=args.prob_scale,
          grad_thr=args.grad_thr,
          method=args.method,
          cuda=args.cuda,
          epochs=args.epochs)
