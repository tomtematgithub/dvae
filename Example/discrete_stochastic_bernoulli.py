import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from discrete_stochastic_utils import *


eps = 1e-8


class Stochastic(nn.Module):
    """
    This class uses the gumbel-softmax trick to reparameterize the Bernoulli variable.
    """
    def reparametrize(self, log_alpha, lamb):

        u = Variable(t.rand(log_alpha.size()), requires_grad=False)
        u = t.min(t.stack((t.max(t.stack((t.full_like(u, eps), u), 0), 0)[0], t.full_like(u, 1-eps)), 0), 0)[0]

        if log_alpha.is_cuda: u = u.cuda()

        z = (log_alpha + t.log(u) - t.log(1 - u))/lamb    # unnormalized logit

        return z


class RelaxedBernoulliSample(Stochastic):
    """
    This class randomly samples from the Beranoulli latent space.
    """
    def __init__(self, input_dim, output_dim, lamb):

        super(RelaxedBernoulliSample, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim    # latent_dim
        self.lamb = lamb
        self.log_alpha = nn.Linear(input_dim, output_dim)
        self.logistic = nn.Sigmoid()

    def forward(self, x, training_mode):
        log_alpha = self.log_alpha(x)   # num_instances x latent_dim
        lamb = self.lamb
        smoothed = self.logistic

        z_sample = smoothed(self.reparametrize(log_alpha, lamb))

        if not training_mode:
            z_sample = t.where(t.gt(z_sample, 0.5), t.ones_like(z_sample), t.zeros_like(z_sample))

        return z_sample, log_alpha     # re-parameterized and smoothed posterior latent vector, estimated parameter vector of posterior distribution
                                       # all dimensions = number of instances x number of latent units

class Encoder(nn.Module):
    """
    This class creates the encoder model with CNN architecture.
    """
    def __init__(self, latent_dim, num_param, window_size, regress, lamb,
                 sample_layer=RelaxedBernoulliSample, filter_1=8, filter_2=16, filter_3=32, filter_4=64):

        super(Encoder, self).__init__()
        self.num_branch = 3
        self.conv1_1 = nn.Conv1d(num_param, filter_1, 3, padding=1)
        self.conv1_2 = nn.Conv1d(num_param, filter_1, 5, padding=2)
        self.conv1_3 = nn.Conv1d(num_param, filter_1, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(filter_1, track_running_stats=False)
        self.pool = nn.MaxPool1d(2)
        self.conv2_1 = nn.Conv1d(filter_1, filter_2, 3, padding=1)
        self.conv2_2 = nn.Conv1d(filter_1, filter_2, 5, padding=2)
        self.conv2_3 = nn.Conv1d(filter_1, filter_2, 7, padding=3)
        self.bn2 = nn.BatchNorm1d(filter_2, track_running_stats=False)
        self.conv3_1 = nn.Conv1d(filter_2, filter_3, 3, padding=1)
        self.conv3_2 = nn.Conv1d(filter_2, filter_3, 5, padding=2)
        self.conv3_3 = nn.Conv1d(filter_2, filter_3, 7, padding=3)
        self.bn3 = nn.BatchNorm1d(filter_3, track_running_stats=False)
        self.conv4_1 = nn.Conv1d(filter_3, filter_4, 3, padding=1)
        self.conv4_2 = nn.Conv1d(filter_3, filter_4, 5, padding=2)
        self.conv4_3 = nn.Conv1d(filter_3, filter_4, 7, padding=3)
        self.bn4 = nn.BatchNorm1d(filter_4, track_running_stats=False)
        self.conv5 = nn.Conv1d(filter_4, filter_4, int(np.floor(window_size/16)))
        self.bn5 = nn.BatchNorm1d(filter_4, track_running_stats=False)
        self.flat = nn.Flatten()
        if regress:
            self.sample1 = sample_layer(int(filter_4*self.num_branch), latent_dim//2, lamb)
            self.sample2 = sample_layer(int(filter_4*self.num_branch+latent_dim//2), latent_dim//2, lamb)
        else:
            self.sample = sample_layer(int(filter_4*self.num_branch), latent_dim, lamb)
        self.regress = regress
        self.regress_flag = True

    def forward(self, x, training_mode):

        x = x.type(t.cuda.FloatTensor) if x.is_cuda else x.type(t.FloatTensor)

        x1 = self.pool(F.relu(self.bn1(self.conv1_1(x))))
        x1 = self.pool(F.relu(self.bn2(self.conv2_1(x1))))
        x1 = self.pool(F.relu(self.bn3(self.conv3_1(x1))))
        x1 = self.pool(F.relu(self.bn4(self.conv4_1(x1))))
        x1 = F.relu(self.bn5(self.conv5(x1)))
        x1 = self.flat(x1)
        
        x2 = self.pool(F.relu(self.bn1(self.conv1_2(x))))
        x2 = self.pool(F.relu(self.bn2(self.conv2_2(x2))))
        x2 = self.pool(F.relu(self.bn3(self.conv3_2(x2))))
        x2 = self.pool(F.relu(self.bn4(self.conv4_2(x2))))
        x2 = F.relu(self.bn5(self.conv5(x2)))
        x2 = self.flat(x2)
        
        x3 = self.pool(F.relu(self.bn1(self.conv1_3(x))))
        x3 = self.pool(F.relu(self.bn2(self.conv2_3(x3))))
        x3 = self.pool(F.relu(self.bn3(self.conv3_3(x3))))
        x3 = self.pool(F.relu(self.bn4(self.conv4_3(x3))))
        x3 = F.relu(self.bn5(self.conv5(x3)))
        x3 = self.flat(x3)
        
        x_out = t.cat((x1, x2, x3), dim=1)

        if self.regress:
            if self.regress_flag:
                print('Hierarchical posterior is on!')
                self.regress_flag = False
            x_out_1 = self.sample1(x_out, training_mode)
            x_out_cat = t.cat((x_out, x_out_1[0]), dim=1)
            x_out_2 = self.sample2(x_out_cat, training_mode)
            x_cg = self.combine_groups(x_out_1[0], x_out_2[0])
            log_alpha_cg = self.combine_groups(x_out_1[1], x_out_2[1])
            x_out = x_cg, log_alpha_cg
        else:
            x_out = self.sample(x_out, training_mode)

        return x_out

    def combine_groups(self, g1, g2):
        g1_1, g1_2 = t.split(g1, g1.shape[1]//2, dim=1)
        g2_1, g2_2 = t.split(g2, g2.shape[1]//2, dim=1)
        g1 = t.cat((g1_1, g2_1), dim=1)
        g2 = t.cat((g1_2, g2_2), dim=1)
        g12 = t.cat((g1, g2), dim=1)
        return g12


class Decoder(nn.Module):
    """
    This class creates the decoder model with CNN architecture.
    """
    def __init__(self, latent_dim, num_param, window_size, metric,
                 filter_1=8, filter_2=16, filter_3=32, filter_4=64):

        super(Decoder, self).__init__()
        
        self.num_param = num_param
        self.window_size = window_size
        self.filter_4 = filter_4
        self.fc = nn.Linear(latent_dim, filter_4)
        self.deconv1_1 = nn.ConvTranspose2d(filter_4, filter_3, (num_param, int(np.floor(window_size/16))))
        self.bn1 = nn.BatchNorm2d(filter_3, track_running_stats=False)
        self.up1 = nn.Upsample(size=(num_param, int(np.floor(window_size/8))), mode='bilinear')
        self.deconv2_1 = nn.ConvTranspose2d(filter_3, filter_2, (1, 3), padding=(0, 1))
        self.deconv2_2 = nn.ConvTranspose2d(filter_3, filter_2, (1, 5), padding=(0, 2))
        self.deconv2_3 = nn.ConvTranspose2d(filter_3, filter_2, (1, 7), padding=(0, 3))
        self.bn2 = nn.BatchNorm2d(filter_2, track_running_stats=False)
        self.up2 = nn.Upsample(size=(num_param, int(np.floor(window_size/4))), mode='bilinear')
        self.deconv3_1 = nn.ConvTranspose2d(filter_2, filter_1, (1, 3), padding=(0, 1))
        self.deconv3_2 = nn.ConvTranspose2d(filter_2, filter_1, (1, 5), padding=(0, 2))
        self.deconv3_3 = nn.ConvTranspose2d(filter_2, filter_1, (1, 7), padding=(0, 3))
        self.bn3 = nn.BatchNorm2d(filter_1, track_running_stats=False)
        self.up3 = nn.Upsample(size=(num_param, int(np.floor(window_size/2))), mode='bilinear')
        self.deconv4_1 = nn.ConvTranspose2d(filter_1, 1, (1, 3), padding=(0, 1))
        self.deconv4_2 = nn.ConvTranspose2d(filter_1, 1, (1, 5), padding=(0, 2))
        self.deconv4_3 = nn.ConvTranspose2d(filter_1, 1, (1, 7), padding=(0, 3))
        self.bn4 = nn.BatchNorm2d(1, track_running_stats=False)
        self.up4 = nn.Upsample(size=(num_param, window_size), mode='bilinear')
        self.convlast = nn.Conv2d(3, 1, (1, 1))
        if metric == 'MSE':
            self.output_activation = nn.Identity()
        elif metric == 'BCE':
            self.output_activation = nn.Sigmoid()

    def forward(self, x):

        x1 = F.relu(self.fc(x))
        x1 = t.reshape(x1, (-1, self.filter_4, 1, 1))
        x1 = self.up1(F.relu(self.bn1(self.deconv1_1(x1))))
        x1 = self.up2(F.relu(self.bn2(self.deconv2_1(x1))))
        x1 = self.up3(F.relu(self.bn3(self.deconv3_1(x1))))
        x1 = self.up4(F.relu(self.bn4(self.deconv4_1(x1))))
        
        x2 = F.relu(self.fc(x))
        x2 = t.reshape(x2, (-1, self.filter_4, 1, 1))
        x2 = self.up1(F.relu(self.bn1(self.deconv1_1(x2))))
        x2 = self.up2(F.relu(self.bn2(self.deconv2_2(x2))))
        x2 = self.up3(F.relu(self.bn3(self.deconv3_2(x2))))
        x2 = self.up4(F.relu(self.bn4(self.deconv4_2(x2))))
        
        x3 = F.relu(self.fc(x))
        x3 = t.reshape(x3, (-1, self.filter_4, 1, 1))
        x3 = self.up1(F.relu(self.bn1(self.deconv1_1(x3))))
        x3 = self.up2(F.relu(self.bn2(self.deconv2_3(x3))))
        x3 = self.up3(F.relu(self.bn3(self.deconv3_3(x3))))
        x3 = self.up4(F.relu(self.bn4(self.deconv4_3(x3))))
        
        x_out = t.cat((x1, x2, x3), dim=1)

        x_out = t.reshape(self.output_activation(self.convlast(x_out)), (-1, self.num_param, self.window_size))

        return x_out


class VAE(nn.Module):
    """
    This class creates the VAE model with CNN architecture.
    """
    def __init__(self, latent_dim, num_param, window_size, regress, metric, lamb, kld):
        super(VAE, self).__init__()
        self.z_dim = latent_dim
        self.p_dim = num_param
        self.t_dim = window_size
        self.regress = regress
        self.metric = metric
        self.lamb = lamb
        self.kld = kld
        self.encoder = Encoder(self.z_dim, self.p_dim, self.t_dim, self.regress, self.lamb)
        self.decoder = Decoder(self.z_dim, self.p_dim, self.t_dim, self.metric)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def kld_(self, q, q_log_alpha):

        if self.kld == 'stochastic':
            
            p_log_alpha = t.full_like(q_log_alpha, 0)
            loss = nn.BCEWithLogitsLoss(reduction='none')
            log_p_z_x = -loss(q_log_alpha, q)
            log_p_z_x = t.sum(log_p_z_x, dim=1)
            log_p_z = -loss(p_log_alpha, q)
            log_p_z = t.sum(log_p_z, dim=1)
            kl = log_p_z_x - log_p_z

            log_prior_pf = t.mean(log_p_z)
            log_posterior_pf = t.mean(log_p_z_x)

        elif self.kld == 'analytic':
            
            phi_log_alpha = q_log_alpha
            alpha = t.exp(phi_log_alpha)
            kl = t.sum(t.div(t.mul(alpha, t.log(t.div(alpha+eps, alpha+1))) +
                             t.log(t.div(1, alpha+1)) +
                             t.mul(alpha+1, t.log(t.full_like(alpha, 2))), alpha+1), dim=-1)

            log_prior_pf = -t.sum(t.log(t.full_like(alpha, 2)), dim=-1)
            log_posterior_pf = t.sum(t.div(t.mul(alpha, t.log(t.div(alpha+eps, alpha+1))) +
                                           t.log(t.div(1, alpha+1)), alpha+1), dim=-1)

        return kl, t.mean(q), t.mean(log_prior_pf), t.mean(log_posterior_pf)

    def forward(self, x, training_mode, eval=False):
        z, z_log_alpha = self.encoder(x, training_mode)
        if not eval: self.kl_div = self.kld_(z, z_log_alpha)
        x_rec = self.decoder(z)
        return x_rec

    def encode(self, x, training_mode):
        z, z_log_alpha = self.encoder(x, training_mode)
        return z

    def sample(self, z):
        x_rec = self.decoder(z)
        return x_rec
