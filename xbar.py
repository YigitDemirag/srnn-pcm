""" PCM crossbar array simulation framework.

Based on Nandakumar, S. R. et al. A phase-change memory model for neuromorphic computing. J Appl Phys 124, 152135 (2018)
author: Yigit Demirag, 2020 @ NCS, INI of ETH Zurich and UZH
"""
import math
import torch

class XBar:
    def __init__(self, G0=0.1, N=1, size=(1024, 1024), res=16, scale=12, prob_scale=120, device=torch.device('cuda')):
        """ Initializes the PCM crossbar (XBar) object.

        Args:
            G0         : Initial conductance value
            N          : Number of (+/-) pairs per synapse
            size       : Size of the crossbar array
            res        : Number of bits per PCM device in perf mode
            scale      : Scaling factor after matmul operation
            prob_scale : Scaling factor for probability calculation used in stochastic weight update rule
        """
        # Parameters
        self.a  = 2.6
        self.m1 = -0.084
        self.c1 = 0.880
        self.A1 = 1.40
        self.m2 = 0.091
        self.c2 = 0.260
        self.A2 = 2.15
        self.m3 = 0.03
        self.c3 = 0.13
        self.v  = 0.04
        self.G0 = G0
        self.Gmax = 12 # Maximum conductance value. Use 20 to replicate Fig 5.
        self.dt = 1e-3
        self.xbar_scale = scale
        self.xbar_res = res
        self.perf_inc = self.Gmax/(2**self.xbar_res) # Control number of steps between min/max conductances
        self.size = size
        self.xbar_n = N
        self.device = device
        self.Pmem = torch.ones(2, N, size[0], size[1], device=self.device) # (+/-, N, X, Y)
        self.tp = torch.zeros(2, N, size[0], size[1], dtype=torch.double, device=self.device) # (+/-, N, X, Y)
        self.count = torch.zeros(2, N, size[0], size[1], device=self.device) # (+/-, N, X, Y)
        self.tracker = torch.zeros(2, size[0], size[1], device=self.device) # (+/-, X, Y)
        self.G = torch.normal(self.G0, self.G0*0.1 , (2, N, size[0], size[1]), device=self.device).clamp(1e-2, self.Gmax)
        self.prob_scale  = prob_scale

    def write(self, tp, mask, perf):
        ''' Emulates a masked WRITE operation on the PCM crossbar.

        Args:
            tp   : Timing of the applied WRITE pulse
            mask : Mask for selecting PCM device
            perf : If True, perform WRITE operation in the performance mode
        '''
        self.Pmem[mask] = self.Pmem[mask] * math.exp(-1 / self.a) 

        # Write + noise
        mu_dgn  = self.m1 * self.G[mask] + (self.c1 + self.A1 * self.Pmem[mask])
        std_dgn = self.m2 * self.G[mask] + (self.c2 + self.A2 * self.Pmem[mask])
        dgn = mu_dgn + std_dgn * torch.randn(torch.sum(mask), device=self.device)
        if not perf:
            self.G[mask] = torch.clamp(self.G[mask] + dgn, 0.1, self.Gmax)
        else:
            self.G[mask] = torch.clamp(self.G[mask] + self.perf_inc, 0.1, self.Gmax) 
        self.count[mask] = self.count[mask] + 1
        self.tp[mask] = tp

    def read(self, t, T0, perf=False):
        ''' Emulates a READ operation on the PCM crossbar.

        Args:
            t     : Timing of the applied READ pulse
            T0    : Initial conductance read after WRITE pulse, constant value (For details, Nandakumar et al. 2018, Eq. 3)
            perf  : If True, perform READ operation in the performance mode
        '''
        # Drift
        Gd = self.G * torch.pow(((t - self.tp)/T0), -self.v).float()

        # Read noise
        std_nG = self.m3 * Gd + self.c3
        nG = torch.normal(torch.zeros_like(Gd), std_nG)

        Gn = torch.clamp(Gd + nG, 0.1, self.Gmax) / self.xbar_scale

        if perf:
            Gn = torch.clamp(self.G, 0.1, self.Gmax) / self.xbar_scale
        return Gn

    def reset(self, tp, mask, G0=0.1):
        ''' Emulates a masked RESET operation on the PCM crossbar.

        Args:
            mask : Mask for selecting PCM device
            G0   : Initial conductance value
        '''
        self.Pmem[mask] = 1
        self.tp[mask] = tp # Please read Issue #1 regarding the change in this line (this is corrected version). 
        self.count[mask] = 0
        self.G[mask] = torch.normal(G0, G0 * 0.1, ((torch.sum(mask),)), device=self.device).clamp(1e-2, self.Gmax)
        self.tracker[mask[:,0,:,:]] = 0 

    def G_to_numpulse(self, G_curr, G_target):
        ''' Calculates the number of WRITE pulses to apply for increasing conductance 
            from the current conductance (G_curr) to the target conductance (G_target).

        Args:
            G_curr  : Current conductance value
            G_target: Target conductance value

        Returns:
            numpulse: Number of WRITE pulses to apply
        '''
        P_target = 0.027 * torch.pow(G_target, 3) - 0.15 * torch.pow(G_target, 2)  + 0.81 * G_target
        P_curr = 0.027 * torch.pow(G_curr, 3) - 0.15 * torch.pow(G_curr, 2)  + 0.81 * G_curr
        numpulse = torch.floor(P_target-P_curr)
        return numpulse

    def target_write(self, tp, G_target, G_curr_est=None, refresh=False, perf=False, method='stochastic'):
        ''' Implements four different weight update mechanisms.

        Args:
            tp        : Timing of the applied WRITE pulses
            G_target  : Target conductance value
            G_curr_est: Current conductance estimate
            refresh   : If True, refresh the differential pairs matching the refresh criteria
            perf      : If True, perform WRITE operation in the performance mode
            method    : Synaptic update mechanisms i.e., `stochastic`, `multi-mem`, `mixed-precision`, `upd-ready`.
        '''

        if method == 'stochastic':
           # PARAMETERS
           reset_thr = 9 # (µS) condition indicating PCM conductance saturation
           eps = 0.75 # (µS) estimated minimum achivable conductance jump

           num_pulse = torch.zeros_like(self.G, device=self.device)
           prob = torch.zeros_like(self.G, device=self.device)
           dG = G_target - G_curr_est

           # refresh (if weights are saturated at high G)
           if refresh:
               amp_mask  = torch.logical_or(self.G[0] > reset_thr, self.G[1]>reset_thr)
               diff_mask = torch.abs(self.G[0] - self.G[1]) < (reset_thr/4)
               ref_mask  = torch.logical_and(amp_mask, diff_mask).unsqueeze(0).repeat(2,1,1,1)
               # back up weight
               G_tmp = self.G[0] - self.G[1]
               # reset the synapses
               self.reset(mask=ref_mask)

               # calculate number of pulses for each pair
               num_pulse[0] = ( G_tmp * (G_tmp>0) / eps).unsqueeze(0)
               num_pulse[1] = (-G_tmp * (G_tmp<0) / eps).unsqueeze(0)

               # load old weight values back to single PCM
               for _ in range(int(torch.max(num_pulse))):
                   update_mask = torch.logical_and(num_pulse > 0, ref_mask)
                   self.write(tp=tp, mask=update_mask, perf=perf)
                   num_pulse = torch.relu(num_pulse-1)

           # calculate update probability
           prob[0] = ( dG * (dG>0) / self.prob_scale).unsqueeze(0)
           prob[1] = (-dG * (dG<0) / self.prob_scale).unsqueeze(0)

           # calculate number of pulses to apply for each differential unit
           num_pulse[0]  = prob[0] > torch.rand(dG.shape, device=self.device)
           num_pulse[1]  = prob[1] > torch.rand(dG.shape, device=self.device)

           # apply WRITE pulses
           self.write(tp=tp, mask=num_pulse.bool(), perf=perf)

        if method == 'multi-mem':
           # PARAMETERS
           reset_thr = 9
           num_PCM = self.xbar_n

           num_pulse = torch.zeros((2, G_target.shape[0], G_target.shape[1]), device=self.device)
           update_mask = torch.zeros_like(self.G, device=self.device)
           dG = G_target - G_curr_est

           # calculate number of pulses for each synapse.
           eps_per_device = 0.75 / num_PCM
           num_pulse[0]  = ( dG * (dG > 0) / eps_per_device).int()
           num_pulse[1]  = (-dG * (dG < 0) / eps_per_device).int()
           # update
           for _ in range(int(torch.max(num_pulse))):
               update_mask = torch.zeros_like(self.G).bool()
               self.tracker = self.tracker + (num_pulse>0)
               for j in range(num_PCM):
                   update_mask[:,j,:,:] = torch.logical_and((num_pulse>0), (((self.tracker-1)%num_PCM)==j))
               self.write(tp=tp, mask=update_mask, perf=perf)
               num_pulse = torch.relu(num_pulse-1)

           # refresh
           if refresh:
                Gtmp = (torch.mean(self.G,1)[0] - torch.mean(self.G,1)[1])
                ref_mask = torch.logical_or(torch.mean(self.G,1)[0]> reset_thr, torch.mean(self.G,1)[1]> reset_thr)
                diff_mask = torch.abs(Gtmp)<(reset_thr/2)
                ref_mask = torch.logical_and(ref_mask, diff_mask)
                self.reset(mask=ref_mask.unsqueeze(0).repeat(2, num_PCM, 1, 1))

                # calculate number of pulses
                num_pulse = torch.zeros((2, Gtmp.shape[0], Gtmp.shape[1]), device=self.device)
                eps_per_device = 0.75 / num_PCM
                num_pulse[0]  = (( Gtmp * (Gtmp > 0) / eps_per_device) * ref_mask).int()
                num_pulse[1]  = ((-Gtmp * (Gtmp < 0) / eps_per_device) * ref_mask).int()

                # Update xbar
                for _ in range(int(torch.max(num_pulse))):
                    update_mask = torch.zeros_like(self.G).bool()
                    self.tracker = self.tracker + (num_pulse>0)
                    for j in range(num_PCM):
                        update_mask[:,j,:,:] = torch.logical_and((num_pulse>0), (((self.tracker-1)%num_PCM)==j))
                    self.write(tp=tp, mask=update_mask, perf=perf)
                    num_pulse = torch.relu(num_pulse-1)

        if method == 'mixed-precision':
           # PARAMETERS
           reset_thr = 6
           num_pulse = torch.zeros_like(self.G, device=self.device)
           dG = G_target - G_curr_est
           eps = 0.75

           # refresh (If weights are saturated at high G)
           if refresh:
               amp_mask  = torch.logical_or(self.G[0]>reset_thr, self.G[1]>reset_thr)
               diff_mask = torch.abs(self.G[0]-self.G[1])<(reset_thr/4)
               ref_mask  = torch.logical_and(amp_mask, diff_mask).unsqueeze(0).repeat(2,1,1,1)
               # back up weight
               G_tmp = self.G[0]-self.G[1]
               # reset the synapse
               self.reset(mask=ref_mask)

               # calculate number of pulses for each pair
               num_pulse[0] = ( G_tmp * (G_tmp>0) / eps).unsqueeze(0)
               num_pulse[1] = (-G_tmp * (G_tmp<0) / eps).unsqueeze(0)

               # load old  weight values back to single PCM
               for _ in range(int(torch.max(num_pulse))):
                   update_mask = torch.logical_and(num_pulse>0, ref_mask)
                   self.write(tp=tp, mask=update_mask, perf=perf)
                   num_pulse = torch.relu(num_pulse-1)

           # calculate number of pulses to apply for each differential unit
           num_pulse[0]  = ( dG * (dG>0) / eps).unsqueeze(0)
           num_pulse[1]  = (-dG * (dG<0) / eps).unsqueeze(0)

           # apply pulses
           for i in range(int(torch.max(num_pulse))):
               update_mask = num_pulse>0
               self.write(tp=tp, mask=update_mask, perf=perf)
               num_pulse = torch.relu(num_pulse-1)

        if method == 'upd-ready':
           # PARAMETERS
           reset_thr = 6
           eps = 0.75

           num_pulse = torch.zeros_like(self.G, device=self.device)
           dG = G_target - G_curr_est
           
           # update-ready scheme (RESET if targeted update is not possible in single shot)
           corr = torch.zeros_like(dG, device=self.device, dtype=torch.int64)
           corr[torch.sign(dG)==-1] = 1
           meme = torch.stack((corr, torch.abs(1-corr)))
           Gsign_pos = torch.gather(self.G, 0, meme.unsqueeze(1))[0]
           cond = torch.abs(dG) > (10 - Gsign_pos)
           reset_ind=torch.stack((torch.logical_and(cond, corr), torch.logical_and(torch.abs(1-corr), cond)))
           self.reset(mask=reset_ind)
           dG = G_target - (self.G[0]-self.G[1]).squeeze(0)

           # refresh (If weights are saturated at high G, rewrite)
           if refresh:
               amp_mask  = torch.logical_or(self.G[0]>reset_thr, self.G[1]>reset_thr)
               diff_mask = torch.abs(self.G[0]-self.G[1])<(reset_thr/4)
               ref_mask  = torch.logical_and(amp_mask, diff_mask).unsqueeze(0).repeat(2,1,1,1)
               # back up weight
               G_tmp = (self.G[0]-self.G[1]) # 1,3,3
               # reset the synapse
               self.reset(mask=ref_mask)

               # calculate number of pulses for each pair
               num_pulse[0] = ( G_tmp * (G_tmp>0) / eps).unsqueeze(0)
               num_pulse[1] = (-G_tmp * (G_tmp<0) / eps).unsqueeze(0)

               # load old  weight values back to single PCM
               for i in range(int(torch.max(num_pulse))):
                   update_mask = torch.logical_and(num_pulse>0, ref_mask)
                   self.write(tp=tp, mask=update_mask, perf=perf)
                   num_pulse = torch.relu(num_pulse-1)


           # calculate number of pulses to apply for each differential unit
           num_pulse[0]  = ( dG * (dG>0) / eps).unsqueeze(0)
           num_pulse[1]  = (-dG * (dG<0) / eps).unsqueeze(0)

           # apply pulses
           for i in range(int(torch.max(num_pulse))):
               update_mask = num_pulse>0
               self.write(tp=tp, mask=update_mask, perf=perf)
               num_pulse = torch.relu(num_pulse-1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--replicate', type=int, default=0)
    parser.add_argument('--perf', type=bool, default=False)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--write_method', type=str, default='Default')
    parser.add_argument('--target', type=float, default=6.1)
    parser.add_argument('--num_iter', type=int, default=10)
    parser.add_argument('--xbar_res', type=float, default=8)
    parser.add_argument('--xbar_n', type=int, default=1)
    parser.add_argument('--xbar_scale', type=float, default=1)
    parser.add_argument('--sample', type=int, default=100)
    parser.add_argument('--prob_scale', type=float, default=120.0)
    args = parser.parse_args()
    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")
    torch.cuda.empty_cache()

######### REPLICATE FIGURE 5 ###########
    if args.replicate == 5:
        sample = args.sample
        pulse  = 20
        T0 = 38.6; tp=0  
        Gtrace = torch.ones((sample, pulse), device=device)
        mask = torch.zeros(2,args.xbar_n,1000,1000, device=device).bool()
        Gf = torch.zeros_like(mask)
        xbar = XBar(N=args.xbar_n, size=(1000,1000), res=args.xbar_res, scale=args.xbar_scale, prob_scale=args.prob_scale, device=device)
        mask[0,0:args.xbar_n,2,3]=True

        for i in range(sample):
            for pulse_no in range(0,pulse,1):
                xbar.write(tp=tp, mask=mask, perf=args.perf)
                Gf = xbar.read(t=tp+T0, T0=T0, perf=args.perf)
                tp = tp + T0
                Gtrace[i, pulse_no] = torch.sum(Gf[mask],dim=0)
            xbar.reset(mask=torch.ones(2,args.xbar_n,1000,1000, device=device).bool())

        # Plotting
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(20, 20)
        ax1 = fig.add_subplot(gs[0:20,0:9])
        plt.errorbar(range(0,pulse,1), torch.mean(Gtrace,0).cpu(), yerr=torch.std(Gtrace,0).cpu(), fmt='o',
                    elinewidth=2, capsize=0, markersize=2)
        plt.xlabel('Pulse number')
        plt.ylabel('G (µS)')
        #plt.ylim([-1,13])
        plt.grid(True)

        ax2 = fig.add_subplot(gs[0:9,11:20])
        plt.plot(range(0,pulse,1),torch.mean(Gtrace,0).cpu(),'.-')
        plt.xlabel('Pulse number')
        plt.ylabel(r'$\mu_G (\mu S)$')
        #plt.ylim([0,10])
        plt.grid(True)

        ax3 = fig.add_subplot(gs[11:20,11:20])
        plt.plot(range(0,pulse,1),torch.std(Gtrace,0).cpu(),'.-')
        plt.xlabel('Pulse number')
        plt.ylabel(r'$\sigma_G (\mu S)$')
        plt.ylim([0,torch.max(torch.std(Gtrace,0).cpu())+1,])
        plt.grid(True)
        plt.show()

######### REPLICATE FIGURE 2 ###########
    if args.replicate == 2:
        # Figure 2 - Replicate
        xbar = XBar(N=args.xbar_n, size=(10,10), res=args.xbar_res, scale=args.xbar_scale, prob_scale=args.prob_scale, device=device)
        mask = torch.zeros(2, args.xbar_n, 10,10).bool()
        mask[0,0:args.xbar_n,1,4] = True

        xbar.reset(mask=mask)
        r = torch.zeros(800)
        for t in range(1,800,1):
            r[t]=torch.sum(xbar.read(t,T0=38.6,perf=args.perf)[mask], dim=0)

            if t % 38.5 == 0:
                xbar.write(tp=t,mask=mask,perf=args.perf)

        plt.plot(r[0:])
        #plt.ylim([0,12])
        plt.xlabel('Time (s)')
        plt.ylabel('G (µS)')
        plt.show()

######### REPLICATE FIGURE 7 ###########
    if args.replicate == 7:
        # Figure 7 - Replicate
        xbar = XBar(size=(100,100), N=args.xbar_n, res=args.xbar_res, scale=args.xbar_scale, prob_scale=args.prob_scale, device=device)
        mask = torch.ones(100, 100, device=device).repeat(2, args.xbar_n, 1, 1).bool()
        xbar.reset(mask=mask, G0=0.1)

        r = torch.zeros(300)
        for i in range(1, 300, 1):
            if i % 5 == 0 and i < 100:
                # Write updates conductance values at G[tp+T0].
                xbar.write(tp=i, mask=mask, perf=args.perf)
            r[i] = torch.mean(xbar.read(t=i, T0=38.6, perf=args.perf))
        plt.plot(r, '.-')
        plt.grid()
        #plt.ylim([0, 12])
        plt.show()

###### TEST TARGET WITH ITERATIVE PROGRAMMING  ###
    if args.write_method in ['multi-mem', 'upd-ready']:

        mean_mat = torch.zeros(21,21).to(device)
        std_mat = torch.zeros(21,21).to(device)
        for s in range(-10,11,1): # 21 x 21
            for t in range(-10,11,1):
                xbar = XBar(N=args.xbar_n, size=(100,100), res=args.xbar_res, scale=args.xbar_scale, prob_scale=args.prob_scale, device=device)
                if s<0:
                    xbar.G[1] = -s
                if s>0:
                    xbar.G[0] = s
                if s == 0:
                    xbar.G[0] = 0.1
                target = t*torch.ones((100,100)).to(device)
                source = s*torch.ones((100,100)).to(device)
                xbar.target_write(tp=1e-3, G_target=target, G_curr_est=source, refresh=True, perf=args.perf, method=args.write_method)
                # Read with mean(G, dim=1) as there is no scaling factor in direct access
                mean_mat[s+10,t+10] = torch.mean(torch.mean(xbar.G,1)[0]-torch.mean(xbar.G,1)[1])
                std_mat[s+10,t+10] = torch.std(torch.mean(xbar.G,1)[0]-torch.mean(xbar.G,1)[1])

        plt.figure(figsize=(16,12))
        plt.subplot(1,2,1)
        ax =sns.heatmap(mean_mat.cpu(), annot=True, xticklabels=list(range(-10,11,1)), yticklabels=list(range(-10,11,1)),annot_kws={"fontsize":8})
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.xlabel('target conductance ($\mu G$)')
        plt.ylabel('source conductance ($\mu G$)')
        err = torch.mean(torch.pow((torch.mean(mean_mat,0)-torch.arange(-10,11,1).to(device)),2))

        plt.title(f'Mean, (Error:{err:.2f})')
        plt.subplot(1,2,2)
        ax2 =sns.heatmap(std_mat.cpu(), annot=True, xticklabels=list(range(-10,11,1)), yticklabels=list(range(-10,11,1)),annot_kws={"fontsize":8})
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        plt.xlabel('target conductance ($\mu G$)')
        plt.ylabel('source conductance ($\mu G$)')
        plt.title('STD')

        plt.show()
        plt.tight_layout()
        s = args.write_method + '_' + str(args.xbar_n) + '_write_result.eps'
        plt.savefig(s, format='eps')
