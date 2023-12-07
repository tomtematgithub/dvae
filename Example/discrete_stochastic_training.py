"""Performs training of a beta-CVAE."""


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from pathlib import Path
import torch as t
import numpy as np
import time
import argparse
import logging
from torch.autograd import Variable
from torch import linalg as LA
import discrete_stochastic_gaussian as dsg
import discrete_stochastic_bernoulli as dsb
import discrete_stochastic_rbm as dsr
from discrete_stochastic_utils import *
from torch.optim import lr_scheduler
from distutils import util


def product(lst_val):
    """Calculates product
    Input: list of values the product of which to calculate
    """
    product = 1 
    for el in lst_val:
        product *= el 
    return product


def train_model(model_type, model, optimizer, scheduler, logger, model_name, mode, train, valid, metric, beta, num_epochs, len_x_train, batch_size, num_param, window_size,
                kl_anneal=False, K=1, D=1, w_l2rbm=0., loss='discrete', save=True, verbose=True, kl_anneal_flag=True, w_l2rbm_flag=True):
    """Trains the model

    Args:
        model_type (string): type of prior model in the VAE's latent space, i.e., 'continuous', 'discrete', or 'boltzmann'
        model (torch neural network model): a model built with torch.nn
        optimizer (torch optimizer): optimizer to be used for training
        scheduler (torch scheduler): learning-rate scheduler to be used for training
        logger (Python logging object): to create a log file, for event logging
        model_name (string): name for saving the resulting model
        mode (string): whether to perform hyperparameter tuning and model selection ('validation') or to obtain performance metrics ('testing')
        train (torch dataloader): training dataloader
        valid (torch dataloader): validation dataloader
        metric (string): metric to be used for the reconstruction loss, i.e., 'BCE' or 'MSE'
        beta (float): hyperparameter beta
        num_epochs (int): number of epochs of training
        len_x_train (int): length of training set
        batch_size (int): size of minibatch
        num_param (int): number of input features
        window_size (int): length of time series
        kl_anneal (bool, optional): whether to implement KL-term annealing (warm up). Defaults to False.
        K (int, optional): how many times to evaluate the lower bound on the log-likelihood. Defaults to 1.
        D (int, optional): how many times to evaluate the gradient. Defaults to  1.
        w_l2rbm (float, optional): weight of L2 penalty of RBM weight decay. Defaults to 0.0.
        loss (string, optional): whether to use the unnormalized reconstruction loss or the continous Bernoulli. Defaults to 'discrete'.
        save (bool, optional): whether to save the trained model or not. Defaults to True.
        verbose (bool, optional): whether to print out the losses during training. Defaults to True.
        kl_anneal_flag (bool, optional): indicates whether to print message when KL-term annealing ('warm-up') is turned on. Defaults to True.
        w_l2rbm_flag (bool, optional): indicates whether to print value of L2 weight penalty on RBM weights when this feature is turned on. Defaults to True.
    """

    # Initialization of losses and accuracies (arrays)
    training_loss_array = np.zeros(num_epochs)
    training_lh_loss_array = np.zeros(num_epochs)
    training_kld_loss_array = np.zeros(num_epochs)
    training_l2rbm_loss_array = np.zeros(num_epochs)
    validation_loss_array = np.zeros(num_epochs)
    validation_lh_loss_array = np.zeros(num_epochs)
    validation_kld_loss_array = np.zeros(num_epochs)

    # Initialization of input data and reconstructed input data (arrays)
    data = np.zeros((len_x_train % batch_size, num_param, window_size))
    data_rec = np.zeros((len_x_train % batch_size, num_param, window_size))

    # Initialization of Kl-divergence diagnostics and latent space-visualization arrays
    latents_array = np.zeros((num_epochs, int(np.ceil(len_x_train/batch_size))))
    log_prior_array = np.zeros((num_epochs, int(np.ceil(len_x_train/batch_size))))
    log_posterior_array = np.zeros((num_epochs, int(np.ceil(len_x_train/batch_size))))

    iteration_accumulator = 0       # stores training time for an iteration

    # for each epoch of training
    for epoch in range(num_epochs):

        epoch_accumulator = 0       # stores training time for an epoch

        model.train()       # this activates the training mode

        # Initialization of losses (numbers; training)
        training_loss, training_lh_loss, training_kld_loss, l2rbm_loss = 0, 0, 0, 0

        # Initialization of lists for Kl-divergence diagnostics and latent-space visualization
        latents_list = []
        log_prior_list = []
        log_posterior_list = []

        # KL-term annealing ("warm-up") over 200 epochs
        if kl_anneal:
            if epoch < 200:
                if kl_anneal_flag:
                    print('KL-term annealing is on!')
                    kl_anneal_flag = False
                beta_hp = epoch/200 * beta
            else:
                beta_hp = beta
        else:
            beta_hp = beta

        # for each mini-batch: looping over training data
        for xt in train:
            
            xt = Variable(xt)            
            
            # this copies the data to gpu if available, otherwise the default is cpu.
            if cuda:
                xt = xt.cuda(device=0)

            # for multiple evaluation of ELBO
            xt_tiled = t.tile(xt, (K, 1, 1))
            xt_rec_tiled = t.zeros_like(xt_tiled)

            L_lh_train_arr = t.zeros(D)
            L_kld_train_arr = t.zeros(D)
            L_train_arr = t.zeros(D)

            start = time.time()
            
            # for each gradient evaluation
            for d in range(D):

                # Obtain the reconstruction for the minibatch
                if model_type == 'continuous':
                    xt_rec_tiled = model(xt_tiled)
                else:
                    xt_rec_tiled = model(xt_tiled, True)

                # Calculate training loss (reconstruction loss and ELBO)
                likelihood = -likelihood_loss(xt_tiled, xt_rec_tiled, metric, loss)
                elbo = likelihood - beta_hp * model.kl_div[0]

                L_lh_train = -t.mean(likelihood)
                L_kld_train = t.mean(model.kl_div[0])
                L_train = -t.mean(elbo)

                if cuda:
                    L_train = L_train.cuda(device=0)

                L_lh_train_arr[d] = L_lh_train
                L_kld_train_arr[d] = L_kld_train
                L_train_arr[d] = L_train

            stop = time.time()
            duration = stop - start
            epoch_accumulator += duration

            latents_list.append(model.kl_div[1].cpu().detach().numpy())
            log_prior_list.append(model.kl_div[2].cpu().detach().numpy())
            log_posterior_list.append(model.kl_div[3].cpu().detach().numpy())
            
            L_lh_train_mean = t.mean(L_lh_train_arr)
            L_kld_train_mean = t.mean(L_kld_train_arr)
            L_train_mean = t.mean(L_train_arr)

            # Penalty on RBM weight matrix
            if model_type == 'boltzmann':
                if w_l2rbm_flag and w_l2rbm > 0:
                    print('The weight of the penalty applied to the RBM weight matrix is '+str(w_l2rbm)+'!')
                    w_l2rbm_flag = False
                weight_penalty = w_l2rbm*t.pow(LA.norm(model.rbm.W), 2)

            training_lh_loss += L_lh_train_mean.item()
            training_kld_loss += L_kld_train_mean.item()
            if model_type == 'boltzmann':
                l2rbm_loss += weight_penalty.item()
            training_loss += L_train_mean.item()

            start = time.time()

            if cuda:
                L_train_mean = L_train_mean.cuda(device=0)
            if model_type == 'boltzmann':
                L_train_mean += weight_penalty

            # this performs backprapagaton and opimizer's step
            L_train_mean.backward()            # getting gradients w.r.t. parameters

            optimizer.step()        # updating parameters
            optimizer.zero_grad()   # to prevent gradient accumulation over batches

            stop = time.time()
            duration = stop - start
            epoch_accumulator += duration

        iteration_accumulator += epoch_accumulator

        latents_array[epoch, :] = latents_list
        log_prior_array[epoch, :] = log_prior_list
        log_posterior_array[epoch, :] = log_posterior_list

        # Evaluation of performance on training data
        mt = len(train)
        training_lh_loss_array[epoch] = training_lh_loss/mt
        training_kld_loss_array[epoch] = training_kld_loss/mt
        training_loss_array[epoch] = training_loss/mt
        training_l2rbm_loss_array[epoch] = l2rbm_loss/mt

        # Log epoch
        logger.info("epoch: "+str(epoch+1)+"\n")

        # Log losses
        logger.info("total training loss = "+str(training_loss/mt))
        logger.info("likelhood loss = "+str(training_lh_loss/mt))
        logger.info("kl-divergence loss = "+str(training_kld_loss/mt))
        logger.info("rbm weight-penalty loss = "+str(l2rbm_loss/mt)+"\n")

        if verbose:
            print("Epoch: {},  Duration: {:0.2f} s".format(epoch+1, epoch_accumulator))
            print("[Training]\t\t L_train: {:.4f}, L_lh_train: {:.4f}, L_kld_train: {:.4f}".format(training_loss_array[epoch], training_lh_loss_array[epoch], training_kld_loss_array[epoch]))

        if mode != 'testing':
            # Validation (no weight updates)
            model.eval()

            # Initialization of losses (numbers; validation)
            validation_loss, validation_lh_loss, validation_kld_loss = 0, 0, 0

            # looping over validation data
            for xv in valid:
                # creating PyTorch Variables for each data entry
                xv = Variable(xv)
                # this copies the data to gpu if available, otherwise the default is cpu.
                if cuda: 
                    xv = xv.cuda(device=0)
                # computing the x_rec loss
                if model_type == 'continuous':
                    xv_rec = model(xv)
                else:
                    xv_rec = model(xv, False)

                # Calculate validation loss
                likelihood = -likelihood_loss(xv, xv_rec, metric, loss)
                elbo = likelihood - beta_hp * model.kl_div[0]

                L_lh_valid = -t.mean(likelihood)
                L_kld_valid = t.mean(model.kl_div[0])
                L_valid = -t.mean(elbo)

                validation_lh_loss += L_lh_valid.item()
                validation_kld_loss += L_kld_valid.item()
                validation_loss += L_valid.item()

            # Evaluation of performance on validation data
            mv = len(valid)
            validation_lh_loss_array[epoch] = validation_lh_loss/mv
            validation_kld_loss_array[epoch] = validation_kld_loss/mv
            validation_loss_array[epoch] = validation_loss/mv
            if verbose:
                print("[Validation]\t\t L_valid: {:.4f}, L_lh_valid: {:.4f}, L_kld_valid: {:.4f}".format(validation_loss_array[epoch], validation_lh_loss_array[epoch], validation_kld_loss_array[epoch]))

        scheduler.step(training_loss/mt)

    data = xt.cpu().detach().numpy()
    x_rec = xt_rec_tiled[range(len(xt)), :, :]
    data_rec = x_rec.cpu().detach().numpy()

    logger.info("total training time = "+str(int(np.round(iteration_accumulator)))+" s\n\n")

    if save:
        # Save the trained model
        t.save(model.state_dict(), ("./"+model_name+".pth"))

    # saving the training losses and accuracies
    np.savez(model_name, training_lh_loss=training_lh_loss_array,
                         training_kld_loss=training_kld_loss_array,
                         training_l2rbm_loss=training_l2rbm_loss_array,
                         training_loss=training_loss_array,
                         validation_lh_loss=validation_lh_loss_array,
                         validation_kld_loss=validation_kld_loss_array,
                         validation_loss=validation_loss_array)

    # Save input data and reconstructed input data
    np.savez('x_'+model_name, x=data, x_rec=data_rec)

    # Save latent variables
    np.savez('z_'+model_name, latents=latents_array, log_prior=log_prior_array, log_posterior=log_posterior_array)


if __name__ == '__main__':

    # Test for GPU availability
    cuda = t.cuda.is_available()

    t.autograd.set_detect_anomaly(True)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        type=str,
                        default='boltzmann',
                        choices=['continuous', 'discrete', 'boltzmann'],
                        help='model type: continuous = Gaussian, discrete = Bernoulli, boltzmann = RBM')

    parser.add_argument('-e',
                        type=int,
                        default=400,
                        help='number of epochs for training')

    parser.add_argument('-md',
                        type=str,
                        default='testing',
                        choices=['validation', 'testing'],
                        help='validation mode for hyperparameter tuning and model selection or testing mode for obtaining performance metrics')

    parser.add_argument('-dr',
                        type=str,
                        default="./",
                        help='directory to the data, use "./" if the data is in the same directory as the source code')

    parser.add_argument('-td',
                        type=str,
                        default='DASHlink_binary_Flaps_withAnomalies',
                        help='what training data to use')

    parser.add_argument('-bs',
                        type=int,
                        default=128,
                        help='mini-batch size')

    parser.add_argument('-l',
                        type=int,
                        default=32,
                        help='latent space dimension')

    parser.add_argument('-v',
                        type=lambda x:bool(util.strtobool(x)),
                        default=True,
                        choices=[False, True],
                        help='whether to print out losses per epoch during training')

    parser.add_argument('-mtrc',
                        type=str,
                        default='BCE',
                        choices=['BCE', 'MSE'],
                        help='metric to be used for the reconstruction loss, i.e., BCE or MSE')

    parser.add_argument('-ka',
                        type=lambda x:bool(util.strtobool(x)),
                        default=False,
                        choices=[False, True],
                        help='whether to implement KL-term annealing (warm up)')

    parser.add_argument('-kl',
                        type=str,
                        default='stochastic',
                        choices=['stochastic', 'analytic'],
                        help='whether to use the stochstic approximation or the anlaytic expression'+
                             'for calculating the KL divergence of the Bernoulli model')

    parser.add_argument('-rg',
                        type=lambda x:bool(util.strtobool(x)),
                        default=False,
                        choices=[False, True],
                        help='whether to implement a conditional approximate posterior')

    parser.add_argument('-k',
                        type=int,
                        default=1,
                        help='how many times to evaluate the lower bound on the log-likelihood')

    parser.add_argument('-d',
                        type=int,
                        default=1,
                        help='how many times to evaluate the gradient')

    parser.add_argument('-ls',
                        type=str,
                        default='discrete',
                        choices=['discrete', 'continuous'],
                        help='whether to use the unnormalized reconstruction loss or the continous Bernoulli')

    parser.add_argument('-b',
                        type=float,
                        default=30.0,
                        help='value of hyperparameter beta')

    parser.add_argument('-lm',
                        type=float,
                        default=0.1,
                        help='temperature of concrete distribution')

    parser.add_argument('-ns',
                        type=int,
                        default=10,
                        help='number of samples to find anomaly-score threshold and test set-label predictions')

    parser.add_argument('-op',
                        type=float,
                        default=4.5005,
                        help='percentage of outliers in training set')

    parser.add_argument('-dt',
                        type=str,
                        default='none',
                        choices=['none', 'sqrt', 'log', 'log2', 'log10', 'inv', 'asinh', 'logit', 'boxcox', 'yeojohnson'],
                        help='transformation applied to anomaly scores (to make them more similar to a normal distirbution)')

    parser.add_argument('-i',
                        type=int,
                        default=1,
                        help='number of iterations to repeat training and validation')

    parser.add_argument('-lg',
                        type=str,
                        default='discrete_stochastic.log',
                        help='name of log file')

    parser.add_argument('-dv',
                        type=lambda x:bool(util.strtobool(x)),
                        default=False,
                        choices=[False, True],
                        help='boolean flag to determine variability between latent units and inactive latent units')

    parser.add_argument('-rv',
                        type=int,
                        default=2,
                        choices=[1, 2],
                        help='version of RBM model')

    parser.add_argument('-fp',
                        type=int,
                        default=500,
                        help='number of fantasy particles')

    parser.add_argument('-pc',
                        type=int,
                        default=25,
                        help='length of persistent chain')

    parser.add_argument('-wl2',
                        type=float,
                        default=0.0,
                        help='weight of L2 penalty of RBM weight decay')

    parser.add_argument('-nw',
                        type=lambda x:bool(util.strtobool(x)),
                        default=False,
                        choices=[False, True],
                        help='whether to set RBM couplings to zero')

    parser.add_argument('-srb',
                        type=lambda x:bool(util.strtobool(x)),
                        default=False,
                        choices=[False, True],
                        help='whether to use a sampling replay buffer')

    args = parser.parse_args()

    """
    Define constants
    """
    model_type = args.m
    num_epochs = args.e
    mode = args.md
    dir2data = args.dr
    dataset = args.td
    batch_size = args.bs
    latent_dim = args.l
    verbose = args.v
    metric = args.mtrc
    kl_anneal = args.ka
    kld_type = args.kl
    regress = args.rg
    K = args.k
    D = args.d
    loss = args.ls
    beta = args.b
    lamb = args.lm
    num_sample = args.ns
    perc_out = args.op
    data_transformation = args.dt
    iterations = args.i
    logfile = args.lg
    dist_var_zs = args.dv
    v_rbm = args.rv
    num_fantasy_particles = args.fp
    len_persistent_chain = args.pc
    w_l2rbm = args.wl2
    nW = args.nw
    sr_buffer = args.srb

    kl_anneal_flag = True
    w_l2rbm_flag = True

    # Create logger
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), filename=logfile)
    logger = logging.getLogger("Discrete-Stochastic-Log")
    logger.info("initializing...")

    """
    Loading the data
    """
    # the data should be 
    ## 1- normalized, if you normalize using minmax scaling, you can
    ##### use any of the metrics, but if you use standard scaling,
    ##### only use MSE as a metric, since BCE won't be applicable.

    ## 2- divided into three sets of training, validation, and testing

    ## 3- data format should be in #_instances * window_size * num_param
    logger.info("loading the data...")
    if dir2data == './':
        train_data = np.load(dataset+"_train.npz")
        valid_data = np.load(dataset+"_valid.npz")
    else:
        train_data = np.load(Path(dir2data+"/"+dataset+"_train.npz"))
        valid_data = np.load(Path(dir2data+"/"+dataset+"_valid.npz"))
    x_train = train_data['data']
    x_valid = valid_data['data']
    if mode == 'testing':
        x_train = np.concatenate((x_train, x_valid))

    len_x_train = len(x_train)

    # In this part, we identify whether the data is normalized using minmax scaler
    ## or standard scaler. This will be used for parameterizing the model, specifically
    ### in the last layer of the decoder, where we use sigmoid activation.
    if metric == 'BCE':
        min_value = np.min(x_train)
        max_value = np.max(x_train)
        # We round the minimum and maximum values used to test for standard scaling because the minimum and maximum
        # values used for standard scaling are derived from the (original) training set and in testing mode the training
        # and validation sets are combined for training. Since a few values of the validation set might slightly exceed
        # the interval [0, 1], values of the new combined training set might exceed them as well.
        # The likelihood_loss function (in discrete_stochastic_utils.py) takes care of any exceedances by clamping the
        # input data and the reconstructed input data. 
        if np.round(min_value, 1) < 0 or np.round(max_value, 1) > 1:
            logger.info("Due to standard scaling of the data, only the MSE can be used as error metric.")
            metric = 'MSE'

    # Determine length of time series and number of attributes
    window_size = np.shape(x_train)[1]
    num_param = np.shape(x_train)[2]

    # Transposition of second and third dimensions because in PyTorch the feature dimension comes before the time dimension
    x_train = np.transpose(x_train, axes=(0, 2, 1))
    x_valid = np.transpose(x_valid, axes=(0, 2, 1))
    training = Dataset(x_train)
    validation = Dataset(x_valid)

    # for each iteration
    for i in range(iterations):
        logger.info("iteration... "+str(i+1)+"\n")
        print("\nIteration:", i+1)

        model_name = "discrete_stochastic_mtype_"+model_type+\
                     "_epochs"+str(num_epochs)+\
                     "_data_"+dataset+\
                     "_bs"+str(batch_size)+\
                     "_l"+str(latent_dim)+\
                     "_b"+str(beta)+\
                     "_pc"+str(len_persistent_chain)+\
                     "_wd"+str(w_l2rbm)+\
                     "_iter"+str(i+1)

        # Load the separate training and validation dataloader for PyTorch
        train, valid = get_dataset(training, validation, batch_size=batch_size)

        """
        Building the model
        """
        logger.info("building the model...")

        # Setting up the model, the optimizer, and the learning-rate scheduler
        if model_type == 'continuous':
            model = dsg.VAE(latent_dim, num_param, window_size, regress, metric)
        elif model_type == 'discrete':
            model = dsb.VAE(latent_dim, num_param, window_size, regress, metric, lamb, kld_type)
        elif model_type == 'boltzmann':
            model = dsr.VAE(latent_dim, num_param, window_size, regress, metric, lamb, v_rbm, batch_size, num_fantasy_particles, len_persistent_chain, nW, len_x_train, num_epochs, sr_buffer, model_name)
        if cuda: model = model.cuda()

        optimizer = t.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


        """
        Training the model
        """
        logger.info("training...\n")
        train_model(model_type, model, optimizer, scheduler, logger, model_name, mode, train, valid, metric, beta, num_epochs, len_x_train, batch_size, num_param, window_size,
                    kl_anneal=kl_anneal, K=K, D=D, w_l2rbm=w_l2rbm, loss=loss, save=True, verbose=verbose, kl_anneal_flag=kl_anneal_flag, w_l2rbm_flag=w_l2rbm_flag)
