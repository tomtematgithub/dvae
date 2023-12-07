import torch as t
import torch.nn as nn
import torch.distributions as td
import numpy as np
import random


class RBM(nn.Module):
    """
    This class implements an RBM prior in the latent space (version: 'bipartite latent-space RBM').
    """
    def __init__(self, latent_dim, batch_size, num_fantasy_particles, nW, len_x_train, num_epochs, use_sr_buffer, model_name, cuda, buffer_size=1024):
        super(RBM, self).__init__()
        self.latent_dim = latent_dim
        self.num_fantasy_particles = num_fantasy_particles
        self.pcd_state = t.zeros(self.num_fantasy_particles, latent_dim)
        if use_sr_buffer:
            self.sr_buffer = t.zeros(buffer_size, self.latent_dim)                                                                     
        self.b = t.zeros(latent_dim, 1)
        self.W = t.zeros(latent_dim // 2, latent_dim // 2)

        self.batch_size = batch_size
        self.len_x_train = len_x_train
        self.num_epochs = num_epochs
        self.use_sr_buffer = use_sr_buffer
        self.model_name = model_name
        self.buffer_size = buffer_size
        self.fp_array = np.zeros((num_epochs, int(np.ceil(len_x_train/batch_size))))
        self.fp_list = []
        self.pphe_array = np.zeros((num_epochs, int(np.ceil(len_x_train/batch_size))))
        self.nphe_array = np.zeros((num_epochs, int(np.ceil(len_x_train/batch_size))))
        self.pphe_list = []
        self.nphe_list = []
        self.b_array = np.zeros((num_epochs, latent_dim, 1))
        self.W_array = np.zeros((num_epochs, latent_dim // 2, latent_dim // 2))
        self.epoch = 0
        self.mb_counter = 0
        self.use_sr_buffer_flag = True

        if cuda:
            self.pcd_state = self.pcd_state.cuda()
            if use_sr_buffer:
                self.sr_buffer = self.sr_buffer.cuda()
            self.b = nn.Parameter(self.b.cuda(), requires_grad=True)
            if not nW:
                self.W = nn.Parameter(self.W.cuda(), requires_grad=True)
            else:
                self.W = self.W.cuda()
        else:
            self.b = nn.Parameter(self.b, requires_grad=True)
            if not nW:
                self.W = nn.Parameter(self.W, requires_grad=True)

        print('RBM init')

    def forward(self, x):
        return x

    def update_samples(self, n, cuda):
        if self.use_sr_buffer:
            # Pick 95% of fantasy states from the buffer, randomly set 5% of the randomly chosen fantasy states to 0 or 1 with equal probability
            if self.use_sr_buffer_flag:
                print('SR buffer is on!')
                self.use_sr_buffer_flag = False
            num_new = np.random.binomial(self.num_fantasy_particles, 0.05)
            probs = 0.5*t.ones(num_new, self.latent_dim)
            rand_states = td.bernoulli.Bernoulli(probs=probs).sample()
            old_states = t.reshape(t.cat(random.choices(self.sr_buffer, k=self.num_fantasy_particles-num_new), dim=0), (self.num_fantasy_particles-num_new, self.sr_buffer.shape[1]))
            if cuda:
                rand_states = rand_states.cuda()
                old_states = old_states.cuda()
            self.pcd_state = t.cat([rand_states, old_states], dim=0)
        # Split fantasy states into two (left and right) to serve as "visible" and "hidden" layers of an RBM model
        pcd_state_l, pcd_state_r = t.split(self.pcd_state, self.latent_dim // 2, dim=1)
        b_l, b_r = t.split(self.b, self.latent_dim // 2, dim=0)   # left and right biases
        for _ in range(n):
            # update right fantasy states
            pcd_state_r = self.gibbs_sample(pcd_state_l, b_r, self.W, transpose_W=False)    # pcd_state_r = Bernoulli( sigma( transpose(b_r) + pcd_state_l * W ) )
            # update left fantasy states
            pcd_state_l = self.gibbs_sample(pcd_state_r, b_l, self.W, transpose_W=True)     # pcd_state_l = Bernoulli( sigma( transpose(b_l) + pcd_state_r * transpose(W) ) )
        # Recombine left and right fantasy states
        self.pcd_state = t.cat((pcd_state_l, pcd_state_r), dim=1)

        if self.use_sr_buffer:
            # Add new fantasy states to the buffer and remove old ones if needed
            self.sr_buffer = t.cat([self.sr_buffer, self.pcd_state], dim=0)
            self.sr_buffer = self.sr_buffer[-self.buffer_size:]

    @staticmethod
    def gibbs_sample(z_in, b, W, transpose_W):
        W = t.transpose(W, 0, 1) if transpose_W else W
        logits = t.transpose(b, 0, 1) + t.matmul(z_in, W)
        z_out = td.bernoulli.Bernoulli(logits=logits).sample()
        return z_out

    def energy(self, z, b, W):
        # Split states into two (left and right) to serve as "visble" and "hidden" layers of an RBM model
        z_l, z_r = t.split(z, self.latent_dim // 2, dim=1)
        energy = -(t.matmul(z, b) + t.sum(t.matmul(z_l, W) * z_r, dim=1, keepdim=True))
        return energy

    def log_prob(self, zeta, training_mode):
        positive_phase_energy = self.positive_phase_energy(zeta)    # phi+eng = - ( z * b + sum( z_l * W * z_r ) )
        negative_phase_energy = self.negative_phase_energy()        # phi-eng = - ( pcd_state * b + sum( pcd_state_l * W * pcd_state_r ) )
        log_p_z = -(positive_phase_energy - negative_phase_energy)   # log p(z) = - ( phi+eng - phi-eng )
        
        if training_mode:
            self.mb_counter += 1
            self.fp_list.append(t.mean(self.pcd_state).cpu().detach().numpy())
            self.pphe_list.append(t.mean(positive_phase_energy).cpu().detach().numpy())
            self.nphe_list.append(negative_phase_energy.cpu().detach().numpy())

        if self.mb_counter >= int(np.ceil(self.len_x_train/self.batch_size)):
            self.fp_array[self.epoch, :] = self.fp_list
            self.pphe_array[self.epoch, :] = self.pphe_list
            self.nphe_array[self.epoch, :] = self.nphe_list
            self.b_array[self.epoch, :, :] = self.b.cpu().detach().numpy()
            self.W_array[self.epoch, :, :] = self.W.cpu().detach().numpy()
            self.fp_list = []
            self.pphe_list = []
            self.nphe_list = []
            self.mb_counter = 0
            self.epoch += 1
            print(self.epoch)
            print(self.num_epochs)
            if self.epoch >= self.num_epochs:
                np.savez('zt_'+self.model_name, fantasy_particles=self.fp_array, positive_phase_energy=self.pphe_array, negative_phase_energy=self.nphe_array)
                np.savez('bW_'+self.model_name, bias_array=self.b_array, weight_array=self.W_array)
        
        return log_p_z

    def positive_phase_energy(self, zeta):
        positive_phase_energy = self.energy(zeta, self.b, self.W)    # clamped to latent states (z)
        return t.mean(positive_phase_energy, dim=-1)

    def negative_phase_energy(self):
        negative_phase_energy = self.energy(self.pcd_state, self.b, self.W)   # free-running
        return t.mean(negative_phase_energy)
    

class RBM2(nn.Module):
    """
    This class implements an RBM prior in the latent space (version: 'RBM with augmented positive phase').
    """
    def __init__(self, latent_dim, batch_size, num_fantasy_particles, nW, len_x_train, num_epochs, use_sr_buffer, model_name, cuda, buffer_size=1024):
        super(RBM2, self).__init__()
        self.latent_dim = latent_dim
        self.num_fantasy_particles = num_fantasy_particles
        self.nW = nW
        self.hidden = t.zeros(batch_size*2, self.latent_dim)          # hidden sample clamped to latent states (z) of base model
        self.persistent_visible = t.zeros(self.num_fantasy_particles, self.latent_dim)   # visible fantasy states
        self.persistent_hidden = t.zeros(self.num_fantasy_particles, self.latent_dim)    # hidden fantasy states
        if use_sr_buffer:
            self.sr_buffer = t.zeros(buffer_size, self.latent_dim)
        self.a = t.zeros(self.latent_dim, 1)                    # bias of visible units
        self.b = t.zeros(self.latent_dim, 1)                    # bias of hidden units
        self.W = t.zeros(self.latent_dim, self.latent_dim)      # visible-hidden coupling matrix

        self.batch_size = batch_size
        self.len_x_train = len_x_train
        self.num_epochs = num_epochs
        self.use_sr_buffer = use_sr_buffer
        self.model_name = model_name
        self.buffer_size = buffer_size
        self.fp_array = np.zeros((num_epochs, int(np.ceil(len_x_train/batch_size))))
        self.fp_list = []
        self.pphe_array = np.zeros((num_epochs, int(np.ceil(len_x_train/batch_size))))
        self.nphe_array = np.zeros((num_epochs, int(np.ceil(len_x_train/batch_size))))
        self.pphe_list = []
        self.nphe_list = []
        self.a_array = np.zeros((num_epochs, latent_dim, 1))
        self.b_array = np.zeros((num_epochs, latent_dim, 1))
        self.W_array = np.zeros((num_epochs, latent_dim, 1))
        self.epoch = 0
        self.mb_counter = 0
        self.use_sr_buffer_flag = True

        if cuda:
            self.hidden = self.hidden.cuda()
            self.persistent_visible = self.persistent_visible.cuda()
            self.persistent_hidden = self.persistent_hidden.cuda()
            if use_sr_buffer:
                self.sr_buffer = self.sr_buffer.cuda()
            self.a = nn.Parameter(self.a.cuda(), requires_grad=True)
            self.b = nn.Parameter(self.b.cuda(), requires_grad=True)
            if not nW:
                self.W = nn.Parameter(self.W.cuda(), requires_grad=True)
            else:
                self.W = self.W.cuda()
        else:
            self.a = nn.Parameter(self.a, requires_grad=True)
            self.b = nn.Parameter(self.b, requires_grad=True)
            if not nW:
                self.W = nn.Parameter(self.W, requires_grad=True)
        
        print('RBM2 init')

    def forward(self, x):
        return x

    def update_samples(self, zeta, n, cuda):
        # update hidden states (1 step)
        hidden = self.gibbs_sample(zeta, self.b, self.W, transpose_W=False)
        if self.use_sr_buffer:
            # Pick 95% of fantasy states from the buffer, randomly set 5% of the randomly chosen fantasy states to 0 or 1 with equal probability
            if self.use_sr_buffer_flag:
                print('SR buffer is on!')
                self.use_sr_buffer_flag = False
            num_new = np.random.binomial(self.num_fantasy_particles, 0.05)
            probs = 0.5*t.ones(num_new, self.latent_dim)
            rand_states = td.bernoulli.Bernoulli(probs=probs).sample()
            old_states = t.reshape(t.cat(random.choices(self.sr_buffer, k=self.num_fantasy_particles-num_new), dim=0), (self.num_fantasy_particles-num_new, self.sr_buffer.shape[1]))
            if cuda:
                rand_states = rand_states.cuda()
                old_states = old_states.cuda()
            persistent_visible = t.cat([rand_states, old_states], dim=0)
        else:
            persistent_visible = self.persistent_visible

        for _ in range(n):
            # update hidden states (n steps)
            persistent_hidden = self.gibbs_sample(persistent_visible, self.b, self.W, transpose_W=False)    # peristent_hidden = Bernoulli( transpose(b) + persistent_visible * W )
            # update visible states (n steps)
            persistent_visible = self.gibbs_sample(persistent_hidden, self.a, self.W, transpose_W=True)     # persistent_visible = Bernoulli( transpose(a) + persistent_hidden * transpose(W) )

        self.hidden = hidden
        self.persistent_visible = persistent_visible
        self.persistent_hidden = persistent_hidden

        if self.use_sr_buffer:
            # Add new fantasy states to the buffer and remove old ones if needed
            self.sr_buffer = t.cat([self.sr_buffer, persistent_visible], dim=0)
            self.sr_buffer = self.sr_buffer[-self.buffer_size:]

    @staticmethod
    def gibbs_sample(x, b, W, transpose_W):
        W = t.transpose(W, 0, 1) if transpose_W else W
        logits = t.transpose(b, 0, 1) + t.matmul(x, W)
        x_out = td.bernoulli.Bernoulli(logits=logits).sample()
        return x_out

    def energy(self, v, h, a, b, W):
        energy = -t.matmul(v, a) - t.matmul(h, b) - t.sum(t.matmul(v, W) * h, dim=1, keepdim=True)
        return energy

    def log_prob(self, zeta, training_mode):
        positive_phase_energy = self.positive_phase_energy(zeta)     # phi+eng = -zeta*a - hidden*b - sum( zeta * W * hidden )
        negative_phase_energy = self.negative_phase_energy()         # phi-eng = -persistent_visible*a - persistent_hidden*b - sum( persistent_visible * W * persistent_hidden )
        log_p_z = -(positive_phase_energy - negative_phase_energy)   # log p(z) = - ( phi+eng - phi-eng )

        if training_mode:
            self.mb_counter += 1
            self.fp_list.append(t.mean(self.persistent_visible).cpu().detach().numpy())
            self.pphe_list.append(t.mean(positive_phase_energy).cpu().detach().numpy())
            self.nphe_list.append(negative_phase_energy.cpu().detach().numpy())

        if self.mb_counter >= int(np.ceil(self.len_x_train/self.batch_size)):
            self.fp_array[self.epoch, :] = self.fp_list
            self.pphe_array[self.epoch, :] = self.pphe_list
            self.nphe_array[self.epoch, :] = self.nphe_list
            self.a_array[self.epoch, :, :] = self.a.cpu().detach().numpy()
            self.b_array[self.epoch, :, :] = self.b.cpu().detach().numpy()
            self.W_array[self.epoch, :, :] = t.reshape(self.W[:, 0], (self.W[:, 0].shape[0], 1)).cpu().detach().numpy() 
            self.fp_list = []
            self.pphe_list = []
            self.nphe_list = []
            self.mb_counter = 0
            self.epoch += 1
            print(self.epoch)
            print(self.num_epochs)
            if self.epoch >= self.num_epochs:
                np.savez('zt_'+self.model_name, fantasy_particles=self.fp_array, positive_phase_energy=self.pphe_array, negative_phase_energy=self.nphe_array)
                np.savez('bW_'+self.model_name, bias_array=(self.a_array, self.b_array), weight_array=self.W_array)

        return log_p_z

    def positive_phase_energy(self, zeta):
        positive_phase_energy = self.energy(zeta, self.hidden, self.a, self.b, self.W)    # clamped to latent states (z) of base model
        return t.mean(positive_phase_energy, dim=-1)

    def negative_phase_energy(self):
        negative_phase_energy = self.energy(self.persistent_visible, self.persistent_hidden, self.a, self.b, self.W)   # free-running
        return t.mean(negative_phase_energy)
    