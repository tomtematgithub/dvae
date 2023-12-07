"""Determines anomaly threshold (based on training data),
   anomaly scores (based on evaluation data), and
   data labels (based on unsupervised inference).

Computes and plots performance metrics:
    - Accuracy
    - Precision, recall, and F1-score
    - Training and validation loss over epochs
    - Confusion matrix
    - ROC curve
    - Precision/recall curve

Plots
    - Losses and accuracies
    - Latent-space configuration and reconstruction
    - KL-divergence diagnostics and latent-space visualizations
    - The first 16 components of the RBM bias vector and the first 16 elements of the first column of the RBM's weight matrix
      (if model_type == 'boltzmann')
"""


from pathlib import Path
import torch as t
import numpy as np
import argparse
import discrete_stochastic_gaussian as dsg
import discrete_stochastic_bernoulli as dsb
import discrete_stochastic_rbm as dsr
from discrete_stochastic_utils import *
from scipy.stats import norm, boxcox, yeojohnson
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from distutils import util
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.ticker import MaxNLocator
from seaborn import heatmap, cm
import pickle
import inflect


def subplot_layout(x):
    """Dictionary for layout of subplots (x = number of subplots)."""
    return {
        1: [1, 1],
        2: [1, 2],
        3: [1, 3],
        4: [2, 2],
        5: [2, 3],
        6: [2, 3],
        7: [2, 4],
        8: [2, 4],
        9: [3, 3],
        10: [3, 4],
        11: [3, 4],
        12: [3, 4],
        13: [3, 5],
        14: [3, 5],
        15: [3, 5],
        16: [4, 4],
    } [x]

def find_score(x, num_sample, model_type, model, data, transformation, metric='BCE', loss='discrete'):
    """Find anomaly score

    Args:
        x (numpy array): input data
        num_sample (int): number of times to reconstruct data
        model_type (string): type of prior model in the VAE's latent space, i.e., 'continuous', 'discrete', or 'boltzmann'
        model (torch model): trained model
        data (string): 'training' or 'evaluation'
        transformation (string): type of data transformation (to be applied to anomaly scores)
        metric (str): loss metric ('MSE' or 'BCE')
        loss (str): unnormalized reconstruction loss or continous Bernoulli ('discrete' or 'continuous')

    Returns:
        Anomlay score for each data point (one-dimensional numpy array)
    """

    num_x = np.shape(x)[0]    # number of instances
    anomaly_score = np.zeros(num_x)
    anomaly_interim_score = np.zeros((num_x, num_sample))

    # Loop to sample anomaly scores multiple times and average them afterwards
    for j in range(num_sample):
        if data == 'training':
            print('j =', j+1)
        else:
            print('jj =', j+1)

        # Reconstruction
        if model_type == 'continuous':
            x_rec = model(t.tensor(x), eval=True)
        else:
            x_rec = model(t.tensor(x), False, eval=True)

        # Apply data transformation (if transformation != 'none') and determine likelihood loss (anomaly scores)
        if transformation == 'none':                                                                               # Matrix of anomaly scores ([training instances x anomaly-score samples])
            anomaly_interim_score[:, j] = likelihood_loss(t.tensor(x), x_rec, metric, loss).detach().numpy()   # Matrix the columns of which contain the anomaly scores from the different samples
        elif transformation == 'sqrt':
            lh_loss = likelihood_loss(t.tensor(x), x_rec, metric, loss)
            lh_loss = t.sqrt(lh_loss)
            anomaly_interim_score[:, j] = lh_loss.detach().numpy()
        elif transformation == 'log':
            lh_loss = likelihood_loss(t.tensor(x), x_rec, metric, loss)
            lh_loss = t.log(lh_loss)
            anomaly_interim_score[:, j] = lh_loss.detach().numpy()
        elif transformation == 'log2':
            lh_loss = likelihood_loss(t.tensor(x), x_rec, metric, loss)
            lh_loss = t.log2(lh_loss)
            anomaly_interim_score[:, j] = lh_loss.detach().numpy()
        elif transformation == 'log10':
            lh_loss = likelihood_loss(t.tensor(x), x_rec, metric, loss)
            lh_loss = t.log10(lh_loss)
            anomaly_interim_score[:, j] = lh_loss.detach().numpy()
        elif transformation == 'inv':
            lh_loss = likelihood_loss(t.tensor(x), x_rec, metric, loss)                                                                                                     
            lh_loss = t.reciprocal(lh_loss)
            # unity = t.ones_like(lh_loss)
            # lh_loss = unity/lh_loss  
            anomaly_interim_score[:, j] = lh_loss.detach().numpy()
        elif transformation == 'asinh':
            lh_loss = likelihood_loss(t.tensor(x), x_rec, metric, loss)
            lh_loss = t.asinh(lh_loss)
            anomaly_interim_score[:, j] = lh_loss.detach().numpy()
        elif transformation == 'logit':
            lh_loss = likelihood_loss(t.tensor(x), x_rec, metric, loss)
            lh_loss = t.logit(lh_loss)
            anomaly_interim_score[:, j] = lh_loss.detach().numpy()
        elif transformation == 'boxcox':
            lh_loss = likelihood_loss(t.tensor(x), x_rec, metric, loss).detach().numpy()
            lh_loss = boxcox(lh_loss)
            anomaly_interim_score[:, j] = lh_loss        
        elif transformation == 'yeojohnson':
            lh_loss = likelihood_loss(t.tensor(x), x_rec, metric, loss).detach().numpy()
            lh_loss = yeojohnson(lh_loss)
            anomaly_interim_score[:, j] = lh_loss 
        
    anomaly_score = np.mean(anomaly_interim_score, axis=1)   # Mean over columns ([training instances])

    return anomaly_score

def determine_metrics(true_labels, predicted_labels):
    """Compute performance metrics.

    Positional arguments:
        true class labels [numpy array of 0s (nominal) and 1s (anomalous)]
        predicted class labels [numpy array of 0s (nominal) and 1s (anomalous)]

    Returns: Accuracy, precision, recall, F1 scores
    """

    # Calculate accuracy
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    accuracy = np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat)

    # Calculate precision and recall
    info = precision_recall_fscore_support(true_labels, predicted_labels)
    precision = info[0][1]
    recall = info[1][1]
    fscore = info[2][1]

    return accuracy, precision, recall, fscore

def get_error(X, k, mi, im, ni):
    """Function that calculates the error for each value of K"""
    error = np.zeros(k)

    # first initiate the K-Means/-Modes
    if model_type == 'continuous':
        km = KMeans(n_clusters=k)
    elif model_type == 'discrete' or model_type == 'boltzmann':
        km = KModes(n_clusters=k, max_iter=mi, init=im, n_init=ni)

    # first initiate the K-Means
    km = KMeans(n_clusters=k)
    
    # fit the K-Means to data
    km.fit(X)
    
    # loop over each cluster
    for l in range(k):
        # select the data that are assinged to the cluster l
        data = X[km.labels_==l]
        # find the center of the cluster
        data_mean = km.cluster_centers_[l]
        # calculate the sum of sqaured distance between each point in the cluster and the center of the cluster
        error[l] = np.sum((data - data_mean)**2)
           
    return np.sum(error)

def plot_losses(training_total_loss, training_lh_loss, training_kld_loss, training_l2rbm_loss,
                validation_total_loss, validation_lh_loss, validation_kld_loss,
                model_name):
    """Make subplots of losses over epochs.

    Positional arguments: various training and validation losses (arrays), model name (string)
    """

    if model_type == 'boltzmann':

        plt.figure(figsize=(17,8))

        plt.subplot(2, 2, 1)
        ax = plt.gca()
        plt.plot(range(1, len(training_total_loss)+1), training_total_loss, label='Training')
        if mode != 'testing':
            plt.plot(range(1, len(training_total_loss)+1), validation_total_loss, label='Validation')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title('Total Loss')

        plt.subplot(2, 2, 2)
        ax = plt.gca()
        plt.plot(range(1, len(training_total_loss)+1), training_lh_loss, label='Training')
        if mode != 'testing':
            plt.plot(range(1, len(training_total_loss)+1), validation_lh_loss, label='Validation')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title('Reconstruction Loss')

        plt.subplot(2, 2, 3)
        ax = plt.gca()
        plt.plot(range(1, len(training_total_loss)+1), beta*training_kld_loss, label='Training')
        if mode != 'testing':
            plt.plot(range(1, len(training_total_loss)+1), beta*validation_kld_loss, label='Validation')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title('KL-Divergence Loss (weighted)')

        plt.subplot(2, 2, 4)
        ax = plt.gca()
        plt.plot(range(1, len(training_total_loss)+1), training_l2rbm_loss, label='Training')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title('Loss due to L2-Penalty on RBM Coupling Weights')

    else:    
        plt.figure(figsize=(25,4))

        plt.subplot(1, 3, 1)
        ax = plt.gca()
        plt.plot(range(1, len(training_total_loss)+1), training_total_loss, label='Training')
        if mode != 'testing':
            plt.plot(range(1, len(training_total_loss)+1), validation_total_loss, label='Validation')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title('Total Loss')

        plt.subplot(1, 3, 2)
        ax = plt.gca()
        plt.plot(range(1, len(training_total_loss)+1), training_lh_loss, label='Training')
        if mode != 'testing':
            plt.plot(range(1, len(training_total_loss)+1), validation_lh_loss, label='Validation')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title('Reconstruction Loss')

        plt.subplot(1, 3, 3)
        ax = plt.gca()
        plt.plot(range(1, len(training_total_loss)+1), training_kld_loss, label='Training')
        if mode != 'testing':
            plt.plot(range(1, len(training_total_loss)+1), validation_kld_loss, label='Validation')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.title('KL-Divergence Loss (unweighted)')

    plt.savefig('losses_'+model_name+'.png')
    plt.close()

def plot_latent_space_diagnostics(subs, rows, cols, inds, arr1, arr2, leg, title, name):
    """"Figure with subplots for KL-divergence diagnostics and latent-space visualization"""
    plt.figure(figsize=(15,6))
    for i in range(subs):
        plt.subplot(rows, cols, i+1)
        plt.scatter(inds, arr1[i, :], s=marker_size, color='mediumseagreen', alpha=transparency)
        plt.scatter(inds, arr2[i, :], s=marker_size, color='mediumpurple', alpha=transparency)
        plt.legend([leg[0], leg[1]])
    plt.suptitle(title)
    plt.savefig(name+'_'+model_name+'.png')
    plt.close()

def plot_bW(num_params, params, name):
    """Figures plotting up to the first sixteen components of the bias vector and
       elements of the first column of the weight matrix over epochs of training"""
    ie = inflect.engine()
    in_words = ie.number_to_words(num_params)
    x = subplot_layout(num_params)
    row = x[0]; col = x[1]
    plt.figure(figsize=(25,10))
    for i in range(num_params):
        plt.subplot(row, col, i+1)
        ax = plt.gca()
        plt.plot(range(1, params.shape[1]+1), params[i, :])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if name == 'b':
        plt.suptitle('First '+in_words.capitalize()+' Components of the Bias Vector over Epochs of Training')
    else:
        plt.suptitle('First '+in_words.capitalize()+' Elements of the First Column of the Weight Matrix over Epochs of Training')
    plt.savefig(name+'_'+model_name+'.png')
    plt.close()
    

if __name__ == '__main__':

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
                        help='number of samples to find anomaly-score threshold and test-label predictions')

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

    model_name_storage_flag = True

    """
    Loading the data
    """

    if dir2data == './':
        train_data = np.load(dataset+"_train.npz")
        valid_data = np.load(dataset+"_valid.npz")
        test_data = np.load(dataset+"_test.npz")
    else:
        train_data = np.load(Path(dir2data+"/"+dataset+"_train.npz"))
        valid_data = np.load(Path(dir2data+"/"+dataset+"_valid.npz"))
        test_data = np.load(Path(dir2data+"/"+dataset+"_test.npz"))
    x_train = train_data['data']
    x_valid = valid_data['data']
    y_valid = valid_data['label']
    x_test = test_data['data']
    y_test = test_data['label']

    if mode == 'testing':
        x_eval = x_test
        y_eval = y_test
        x_train = np.concatenate((x_train, x_valid))
    else:
        x_eval = x_valid
        y_eval = y_valid

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
            print("Due to standard scaling of the data, only the MSE can be used as error metric.")
            metric = 'MSE'

    # Define window size (for 1D convolutional kernels) and number of attributes
    window_size = np.shape(x_eval)[1]
    num_param = np.shape(x_eval)[2]

    # Transposition of second and third dimensions because in PyTorch the feature dimension comes before the time dimension
    x_train = np.transpose(x_train, axes=(0, 2, 1))
    x_eval = np.transpose(x_eval, axes=(0, 2, 1))

    # Instantiate and pre-allocate loss arrays
    training_lh_loss = np.zeros((iterations, num_epochs))
    training_kld_loss = np.zeros((iterations, num_epochs))
    training_loss = np.zeros((iterations, num_epochs))
    training_l2rbm_loss = np.zeros((iterations, num_epochs))
    validation_lh_loss = np.zeros((iterations, num_epochs))
    validation_kld_loss = np.zeros((iterations, num_epochs))
    validation_loss = np.zeros((iterations, num_epochs))

    # Instantiate and pre-allocate performance arrays
    acc = np.zeros(iterations)
    pr = np.zeros(iterations)
    rc = np.zeros(iterations)
    f1 = np.zeros(iterations)

    # Instantiate and pre-allocate arrays for KL-divergence diagnostics and latent-space visualization
    positive_array_flat = np.zeros((iterations, num_epochs*int(np.ceil(len_x_train/batch_size))))
    negative_array_flat = np.zeros((iterations, num_epochs*int(np.ceil(len_x_train/batch_size))))
    log_prior_array_flat = np.zeros((iterations, num_epochs*int(np.ceil(len_x_train/batch_size))))
    log_posterior_array_flat = np.zeros((iterations, num_epochs*int(np.ceil(len_x_train/batch_size))))
    positive_energy_array_flat = np.zeros((iterations, num_epochs*int(np.ceil(len_x_train/batch_size))))
    negative_energy_array_flat = np.zeros((iterations, num_epochs*int(np.ceil(len_x_train/batch_size))))

    # Lists to collect variables over iterations
    true_labels = []
    predicted_labels = []
    reconstruction_errors = []

    # Make histograms of anomaly scores
    x = subplot_layout(iterations)
    row = x[0]; col = x[1]
    if row == col:
        plt.figure(figsize=(25,15))
    else:
        plt.figure(figsize=(15,8))

    # for each iteration
    for i in range(iterations):

        print('Iteration: '+str(i+1)+' of '+str(iterations))
        
        # Evaluation of performance on test data
        model_name = "discrete_stochastic_mtype_"+model_type+\
                     "_epochs"+str(num_epochs)+\
                     "_data_"+dataset+\
                     "_bs"+str(batch_size)+\
                     "_l"+str(latent_dim)+\
                     "_b"+str(beta)+\
                     "_pc"+str(len_persistent_chain)+\
                     "_wd"+str(w_l2rbm)+\
                     "_iter"+str(i+1)

        # setting up the model
        if model_type == 'continuous':
            model = dsg.VAE(latent_dim, num_param, window_size, regress, metric)
        elif model_type == 'discrete':
            model = dsb.VAE(latent_dim, num_param, window_size, regress, metric, lamb, kld_type)
        elif model_type == 'boltzmann':
            model = dsr.VAE(latent_dim, num_param, window_size, regress, metric, lamb, v_rbm, batch_size, num_fantasy_particles, len_persistent_chain, nW, len_x_train, num_epochs, sr_buffer, model_name, cuda=False)

        print('Loading model \''+model_name+'.pth\'.')

        # Load saved model and map it to CPU
        model.load_state_dict(t.load((model_name+".pth"), map_location=t.device('cpu')), strict=False)
        
        # Evaluation (no weight updates)
        model.eval()

        # Load losses
        losses = np.load(model_name+".npz")
        training_lh_loss[i, :] = losses['training_lh_loss']
        training_kld_loss[i, :] = losses['training_kld_loss']
        training_l2rbm_loss[i, :] = losses['training_l2rbm_loss']
        training_loss[i, :] = losses['training_loss']
        validation_lh_loss[i, :] = losses['validation_lh_loss']
        validation_kld_loss[i, :] = losses['validation_kld_loss']
        validation_loss[i, :] = losses['validation_loss']

        # Load arrays for KL-divergence diagnostics and latent-space visualization
        full_data_positive = np.load("z_"+model_name+".npz")
        positive_array = full_data_positive['latents']
        log_prior_array = full_data_positive['log_prior']
        log_posterior_array = full_data_positive['log_posterior']
        positive_array_flat[i, :] = positive_array.flatten()
        log_prior_array_flat[i, :] = log_prior_array.flatten()
        log_posterior_array_flat[i, :] = log_posterior_array.flatten()
        if model_type == 'boltzmann':
            full_data_negative = np.load("zt_"+model_name+".npz")
            negative_array = full_data_negative['fantasy_particles']
            positive_energy_array = full_data_negative['positive_phase_energy']
            negative_energy_array = full_data_negative['negative_phase_energy']
            negative_array_flat[i, :] = negative_array.flatten()
            positive_energy_array_flat[i,:] = positive_energy_array.flatten()
            negative_energy_array_flat[i,:] = negative_energy_array.flatten()
        
        """
        Finding anomaly scores and the threshold for anomaly detection
        """
        scale = norm.ppf(1 - perc_out/100)      # quantile function (inverse CDF): returns z-score of standard normal distribution
        anomaly_score_train = find_score(x_train, num_sample, model_type, model, 'training', data_transformation, metric, loss)
        if data_transformation == 'inv':
            threshold = np.mean(anomaly_score_train) - scale*np.std(anomaly_score_train, ddof=1)   # thr = E[anomaly_score] - scale*std(anomaly_score)
        else:
            threshold = np.mean(anomaly_score_train) + scale*np.std(anomaly_score_train, ddof=1)   # thr = E[anomaly_score] + scale*std(anomaly_score)

        """
        Test the model
        """
        anomaly_score_eval = find_score(x_eval, num_sample, model_type, model, 'evaluation', data_transformation, metric, loss)
            
        pred_eval = np.zeros(len(y_eval))

        n = anomaly_score_eval[y_eval==0.]    # nominal
        p = anomaly_score_eval[y_eval==1.]    # anomalous

        # Make subplots of anomaly scores
        if data_transformation == 'inv':
            # If the inverse transformation is used on the anomaly scores,
            # the historgram is reversed (reflected about the highest anomaly score),
            # so that anomalies appear on the right 
            max_ase = np.max(anomaly_score_eval)
            n = max_ase - n
            p = max_ase - p
            threshold_refl = max_ase - threshold
        else:
            threshold_refl = threshold
        # threshold4g = '{0:1.4g}'.format(threshold_refl)      # threshold with 4 significant digits excluding trailing zeros
        threshold4g = format(threshold_refl, '#.4g')           # threshold with 4 significant digits including trailing zeros
        plt.subplot(row, col, i+1)
        ax = plt.gca()
        plt.hist(n, bins=80, alpha=0.75, color='blue', label='Nominal')
        plt.hist(p, bins=80, alpha=0.9, color='red', label='Anomalous')
        plt.axvline(x=threshold_refl, ymax = 1, c='k', lw='1.2', ls='dashed', alpha=1, label='Threshold')
        plt.legend(fontsize=12, loc='upper right', framealpha=1)
        plt.text(0.74, 0.64, 'threshold =\n'+str(threshold4g), fontsize=12, ha='left', va='top', transform=ax.transAxes)
        print('threshold =', threshold)

        if data_transformation == 'inv':
            pred_eval[anomaly_score_eval <= threshold] =  1   # Logical vector (component = 1 if anomaly score is less than thr, 0 otherwise)
            max_ase = np.max(anomaly_score_eval)
            anomaly_score_eval = max_ase - anomaly_score_eval
        else:
            pred_eval[anomaly_score_eval >= threshold] =  1   # Logical vector (component = 1 if anomaly score exceeds thr, 0 otherwise)

        # Determine performance metrics
        acc[i], pr[i], rc[i], f1[i]  = determine_metrics(y_eval, pred_eval)

        true_labels.append(y_eval)
        predicted_labels.append(pred_eval)
        reconstruction_errors.append(anomaly_score_eval)

        if dist_var_zs == True and model_type == 'boltzmann':
            # Determine distinct encodings, number of active units,
            # encodings of active units, and variances of latent units
            if model_type == 'continuous':
                z = model.encode(t.tensor(x_eval))
            else:
                z = model.encode(t.tensor(x_eval), False)

            z = z.detach().numpy()

            # Determine distinct latent-space encodings for evaluation set
            dist_zs = []
            for l in range(len(z)):
                if not any((z[l, :] == z_el).all() for z_el in dist_zs):
                    dist_zs.append(z[l, :])
            if len(dist_zs) <= 1000:
                description_dist_zs = '\n\nDistinct latent-space encodings:\n'
                dist_zs_to_print = dist_zs
            else:
                description_dist_zs = '\n\n'
                dist_zs_to_print = []

            # Determine the variance of the value of each unit z over the test set
            var_zs0 = np.zeros(z.shape[1])
            var_zs = np.zeros((2, z.shape[1]))
            var_zs0 = np.var(z, axis=0, ddof=1)
            idx_var_zs = np.arange(z.shape[1], dtype=np.int32)
            var_zs = np.transpose(np.stack([idx_var_zs, var_zs0]))

            # Print results to text file
            with open('dist_var_zs_'+model_name+'.txt', 'wt') as text_file:
                print('Number of distinct latent-space encodings for evaluation set = '+str(len(dist_zs))+
                    '\nNumber of active units = '+str(len(var_zs0[var_zs0>0.01]))+
                    '\nLatent-space encodings of active units:\n'+str(z[:, var_zs0>0.01])+
                    '\n\nVariances of latent units over evaluation set:\n'+str(var_zs)+
                    description_dist_zs+str(dist_zs_to_print), file=text_file)

    # Complete subplots of anomaly scores
    plt.suptitle('Test Set Anomaly Scores')
    plt.savefig('error_xrec_'+model_name+'_'+data_transformation+'.png')
    plt.close()

    # Determine averages and standard deviations of losses
    training_lh_loss_avg = np.mean(training_lh_loss, axis=0)
    training_kld_loss_avg = np.mean(training_kld_loss, axis=0)
    training_l2rbm_loss_avg = np.mean(training_l2rbm_loss, axis=0)
    training_loss_avg = np.mean(training_loss, axis=0)
    validation_lh_loss_avg = np.mean(validation_lh_loss, axis=0)
    validation_kld_loss_avg = np.mean(validation_kld_loss, axis=0)
    validation_loss_avg = np.mean(validation_loss, axis=0)
    training_lh_loss_std = np.std(training_lh_loss, axis=0, ddof=1)
    training_kld_loss_std = np.std(training_kld_loss, axis=0, ddof=1)
    training_l2rbm_loss_std = np.std(training_l2rbm_loss, axis=0, ddof=1)
    training_loss_std = np.std(training_loss, axis=0, ddof=1)
    validation_lh_loss_std = np.std(validation_lh_loss, axis=0, ddof=1)
    validation_kld_loss_std = np.std(validation_kld_loss, axis=0, ddof=1)
    validation_loss_std = np.std(validation_loss, axis=0, ddof=1)

    true_labels_flattened = np.concatenate(true_labels).ravel()
    predicted_labels_flattened = np.concatenate(predicted_labels).ravel()
    reconstruction_errors_flattened = np.concatenate(reconstruction_errors).ravel()
    
    # ROC and precision/recall curves for computed anomaly scores
    fpr_test, tpr_test, _ = roc_curve(true_labels_flattened, reconstruction_errors_flattened, pos_label=1)
    pr_test, rc_test, _ = precision_recall_curve(true_labels_flattened, reconstruction_errors_flattened, pos_label=1)

    # Determine averages of performance metrics
    acc_avg = np.mean(acc)
    pr_avg = np.mean(pr)
    rc_avg = np.mean(rc)
    f1_avg = np.mean(f1)

    # Determine standard deviations of performance metrics
    acc_std = np.std(acc, ddof=1)
    pr_std = np.std(pr, ddof=1)
    rc_std = np.std(rc, ddof=1)
    f1_std = np.std(f1, ddof=1)

    # Print metrics to text file
    with open('metrics_'+model_name+'_'+data_transformation+'.txt', 'wt') as text_file:
        print('Accuracy = '+str(np.round(acc_avg, 4))+
              '\nStandard deviation of accuracy = '+str(np.round(acc_std, 5))+
              '\nPrecision = '+str(np.round(pr_avg, 4))+
              '\nStandard deviation of precision = '+str(np.round(pr_std, 5))+
              '\nRecall = '+str(np.round(rc_avg, 4))+
              '\nStandard deviation of recall = '+str(np.round(rc_std, 5))+
              '\nF1 score = '+str(np.round(f1_avg, 4))+
              '\nStandard deviation of F1 score = '+str(np.round(f1_std, 5)), file=text_file)

    with open('raw_metrics_'+model_name+'_'+data_transformation+'.txt', 'wt') as text_file:
        print('Accuracy = '+str(np.round(acc, 4))+
              '\nPrecision = '+str(np.round(pr, 4))+
              '\nRecall = '+str(np.round(rc, 4))+
              '\nF1-score = ' +str(np.round(f1, 4)), file=text_file)

    # Save losses, metrics, and scores
    np.savez('losses_'+model_name,
             training_lh_loss_avg=training_lh_loss_avg,
             training_kld_loss_avg=training_kld_loss_avg,
             training_l2rbm_loss_avg=training_l2rbm_loss_avg,
             training_loss_avg=training_loss_avg,
             validation_lh_loss_avg=validation_lh_loss_avg,
             validation_kld_loss_avg=validation_kld_loss_avg,
             validation_loss_avg=validation_loss_avg,
             training_lh_loss_std=training_lh_loss_std,
             training_kld_loss_std=training_kld_loss_std,
             training_l2rbm_loss_std=training_l2rbm_loss_std,
             training_loss_std=training_loss_std,
             validation_lh_loss_std=validation_lh_loss_std,
             validation_kld_loss_std=validation_kld_loss_std,
             validation_loss_std=validation_loss_std)
    np.savez('perf_met_'+model_name+'_'+data_transformation,
             acc_avg=acc_avg, acc_std=acc_std, pr_avg=pr_avg, pr_std=pr_std,
             rc_avg=rc_avg, rc_std=rc_std, f1_avg=f1_avg, f1_std=f1_std)
    np.savez('scores_'+model_name+'_'+data_transformation,  anomaly_score=reconstruction_errors_flattened, true_label=true_labels_flattened,
             predicted_label=predicted_labels_flattened)

    # Plot results

    # History of training and validation losses
    fig = plt.figure(figsize=(15,6))
    ax = fig.add_subplot(111)
    if model_type == 'boltzmann':
        ax.plot(range(1, len(training_loss_avg)+1), training_loss_avg, label='Training Total')
        ax.plot(range(1, len(training_loss_avg)+1), training_lh_loss_avg, label='Training Reconstruction')
        ax.plot(range(1, len(training_loss_avg)+1), beta*training_kld_loss_avg, label='Training KL-Divergence (weighted)')
        ax.plot(range(1, len(training_loss_avg)+1), training_l2rbm_loss_avg, label='Loss due to L2-Penalty on RBM Coupling Weights')
        if mode != 'testing':
            ax.plot(range(1, len(training_loss_avg)+1), validation_loss_avg, label='Validation Total')
            ax.plot(range(1, len(training_loss_avg)+1), validation_lh_loss_avg, label='Validation Reconstruction')
            ax.plot(range(1, len(training_loss_avg)+1), beta*validation_kld_loss_avg, label='Validation KL-Divergence (weighted)')
        ax.set_xticks(range(1, len(training_loss_avg)+1))
    else:
        ax.plot(range(1, len(training_loss_avg)+1), training_loss_avg, label='Training Total')
        ax.plot(range(1, len(training_loss_avg)+1), training_lh_loss_avg, label='Training Reconstruction')
        ax.plot(range(1, len(training_loss_avg)+1), training_kld_loss_avg, label='Training KL-Divergence (unweighted)')
        if mode != 'testing':
            ax.plot(range(1, len(training_loss_avg)+1), validation_loss_avg, label='Validation Total')
            ax.plot(range(1, len(training_loss_avg)+1), validation_lh_loss_avg, label='Validation Reconstruction')
            ax.plot(range(1, len(training_loss_avg)+1), validation_kld_loss_avg, label='Validation KL-Divergence (unweighted)')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlim(0, len(training_loss_avg)+1) # consistent scale
    ax.legend()
    if num_epochs % 8 == 0: 
        ax.xaxis.set_major_locator(MaxNLocator(9, integer=True))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(11, integer=True))
    fig.tight_layout()
    fig.savefig('history_'+model_name+'.png', bbox_inches='tight')
    plt.close(fig)

    plot_losses(training_loss_avg, training_lh_loss_avg, training_kld_loss_avg, training_l2rbm_loss_avg,
                validation_loss_avg, validation_lh_loss_avg, validation_kld_loss_avg,
                model_name)

    # Confusion matrix
    plt.figure(figsize=(9,7.5))
    conf_mat = confusion_matrix(true_labels_flattened, predicted_labels_flattened)
    conf_mat = np.round(conf_mat / (iterations))
    heatmap(conf_mat, square=True, annot=True, cbar=True, xticklabels=['Nominal', 'Anomalous'],
            yticklabels=['Nominal', 'Anomalous'], cmap=cm.rocket_r, fmt='g')
    plt.tight_layout()
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.title('Confusion Matrix of the Detector')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.gcf().subplots_adjust(left=0.2)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.savefig('cm_'+model_name+'_'+data_transformation+'.png')
    plt.close()

    # ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr_test, tpr_test)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.text(0.5, 0.3, 'Testing AUC  = {}'.format(np.round(auc(fpr_test, tpr_test),3)), fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig('roc_'+model_name+'_'+data_transformation+'.png')
    plt.close()

    # Precision/recall curve
    plt.figure(figsize=(8,6))
    plt.plot(rc_test, pr_test)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.text(0.2, 0.2, 'Testing AUC  = {}'.format(np.round(auc(rc_test, pr_test),3)), fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig('pr_'+model_name+'_'+data_transformation+'.png')
    plt.close()

    # Latent-space configuration
    if model_type == 'continuous':
        z = model.encode(t.tensor(x_eval))
    else:
        z = model.encode(t.tensor(x_eval), False)

    # Create colormap
    values = range(8)
    accent = cm = plt.get_cmap('Accent')
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=accent)
    colorVal = [scalarMap.to_rgba(i) for i in values] 

    # Save predicted scores and class labels
    with open('scores_t_'+model_name+'_'+data_transformation+'.pkl', 'wb') as f:
        pickle.dump(pred_eval, f)
    with open('y_eval_cl_'+model_name+'.pkl', 'wb') as f:
        pickle.dump(y_eval, f) 

    # Visualize latent space by means of classes and clusters
    z = z.detach().numpy()

    # T-SNE plot
    p = 10; ee = 12; lr = 200   # t-SNE variables
    tsne = TSNE(n_components=2, perplexity=p, early_exaggeration=ee, learning_rate=lr, init='pca').fit_transform(z)
    with open('tsne_'+model_name+'.pkl', 'wb') as f:
        pickle.dump(tsne, f)
    plt.figure(figsize=(18,5))
    # true labels
    plt.subplot(1, 3, 1)
    for i in range(2):
        indices = np.where(y_eval==i)[0]
        plt.scatter(tsne[indices, 0], tsne[indices, 1], color=colorVal[i], alpha=0.5)
    plt.legend(['Nominal', 'Anomalous'])
    plt.title('True Classes')
    # predicted labels
    plt.subplot(1, 3, 2)
    for i in range(2):
        indices = np.where(pred_eval==i)[0]
        plt.scatter(tsne[indices, 0], tsne[indices, 1], color=colorVal[i], alpha=0.5)
    plt.legend(['Nominal', 'Anomalous'])
    plt.title('Predicted Classes')

    # Cluster plot
    k = 2
    mi = 100; im = 'Cao'; ni = 5       # k-modes variables
    if model_type == 'continuous':
        km = KMeans(n_clusters=k)
    elif model_type == 'discrete' or model_type == 'boltzmann':
        km = KModes(n_clusters=k, max_iter=mi, init=im, n_init=ni)
    clusters = km.fit_predict(z)
    plt.subplot(1, 3, 3)
    for i in range(k):
        indices = np.where(clusters==i)[0]
        plt.scatter(tsne[indices, 0], tsne[indices, 1], color=colorVal[i], alpha=0.5)
    plt.legend(['k = 1', "k = 2"])
    plt.title('Clusters')
    plt.savefig('lsc_'+model_name+'.png')
    plt.close()

    # # Elbow plot
    # # define a range of K values to test
    # k_values = np.arange(1, kmax+1, 1)
    # errors = np.zeros(len(k_values))
    # for i in range(len(k_values)):
    #     errors[i] = get_error(z, k_values[i], mi, im, ni)
    # plt.figure(figsize=(8, 6))
    # plt.plot(k_values, errors, marker='s', markersize=6, markerfacecolor='green')
    # plt.xlabel('Number of clusters (k)', fontsize=14)
    # plt.ylabel('Sum of Squared Errors', fontsize=14)
    # plt.xticks(k_values)
    # plt.tick_params(axis='both', which='major', labelsize=14)
    # plt.savefig('elbow_'+model_name+'.png')
    # plt.close()

    # Compare input data and reconstructed input data
    data = np.load("x_"+model_name+".npz")
    x = data['x']
    x_rec = data['x_rec']

    variable_names = ['Corrected AOA', 'Barometric Altitude', 'Computed Airspeed', 'TE Flap Position', 'Glideslope Deviation', 'Core Speed AVG', 'Pitch Angle', 'Roll Angle', 'True Heading', 'Wind Speed']

    plt.figure(figsize=(30,8))
    num_subplots = np.min([x.shape[1], 10])
    for i in range(num_subplots):
        plt.subplot(2, 5, i+1)
        plt.plot(x[0, i, :], label='x')
        plt.plot(x_rec[0, i, :], label='x_rec')
        plt.legend()
        plt.title(variable_names[i])
    plt.savefig('reconstruction_'+model_name+'.png')
    plt.close()

    indices = np.arange(1, len(positive_array_flat[0, :])+1)

    marker_size = 25
    transparency = 0.25

    x = subplot_layout(iterations)
    nrow = x[0]; ncol = x[1]

    nrow = 1; ncol = 1

    nplot = nrow*ncol

    # Make figures for KL-divergence diagnostics and latent-space visualization
    plot_latent_space_diagnostics(nplot, nrow, ncol, indices, positive_array_flat, negative_array_flat, ('Positive Phase', 'Negative Phase'), 'Latents', 'z')
    plot_latent_space_diagnostics(nplot, nrow, ncol, indices, log_prior_array_flat, log_posterior_array_flat, ('Log Prior', 'Log Posterior'), 'Log Probability Functions', 'lpf')
    if model_type == 'boltzmann':
        plot_latent_space_diagnostics(nplot, nrow, ncol, indices, positive_energy_array_flat, negative_energy_array_flat, ('Positive Phase', 'Negative Phase'), 'Energy', 'ez')

        # Plot up to the first 16 components of the bias vector and elements of the first column of the weight matrix over epochs
        parameters = np.load("bW_"+model_name+".npz")
        if v_rbm==1:
            bias_array = parameters['bias_array']
        else:
             bias_array = parameters['bias_array'][0]
        weight_array = parameters['weight_array']

        num_bW = np.min([bias_array.shape[1], 16])

        bias_arr = np.zeros((num_bW, num_epochs))
        weight_arr = np.zeros((num_bW, num_epochs))

        for i in range(num_bW):
            bias_arr[i, :] = bias_array[:, i, 0]
            weight_arr[i, :] = weight_array[:, i, 0]

        plot_bW(num_bW, bias_arr, 'b')
        plot_bW(num_bW, weight_arr, 'W')
