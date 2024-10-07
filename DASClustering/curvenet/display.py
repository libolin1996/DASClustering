import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import os
import yaml
import re
import imageio
from yaml import Loader
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.interpolate import CubicHermiteSpline


def show_clusters(epoch, dir_output,
                  save='clusters.png'):
    """Show seismic data in scattering latent space."""

    if dir_output is None:
        dir_output = os.path.join(dir_summary, tag)
    with h5py.File(dir_output + '/clusters.h5', 'r') as hf:
        latent = hf['epoch_{:05d}'.format(epoch)]['features'][()]
        hot = hf['epoch_{:05d}'.format(epoch)]['hot'][()]

    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams.update({'font.size': 20})
    for h in np.unique(hot):
        ax.scatter(*latent[hot == h].T, '.', s=15, alpha=.4, label='Cluster'+str(h+1))
    plt.rcParams.update({'font.size': 16})
    ax.set_xlabel('First latent variable', fontsize=22, labelpad=20) 
    ax.set_ylabel('Second latent variable', fontsize=22, labelpad=20)  
    ax.set_zlabel('Third latent variable', fontsize=22, labelpad=20)  
    ax.legend()
    ax.grid()
    plt.tight_layout()
    if save is not None:
        fig.savefig(os.path.join(dir_output, save), dpi=300)
    return fig, ax

def show_clustering_loss(dir_output, save='clustering_loss.png'):

    data = np.loadtxt(dir_output + '/loss_clustering.txt')
    data = data/2-0.3
    fig, ax = plt.subplots(1)
    ax.plot(data[1:])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Clustering Loss')
    ax.grid()
    return fig,ax

def show_weights(n_features_all, epochs, dir_output, save='Weights.png'):

    data = np.loadtxt(dir_output + '/Weights.txt')
    fig, ax = plt.subplots(1)
    for i in [0,1,2,10,11]:
        tmp = list()
        for j in range(epochs):
            tmp.append(i+j*n_features_all)
        ax.plot(data[tmp], label='Weight'+str(i+1))
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Weights')
    ax.grid()
    fig.savefig(os.path.join(dir_output, save), dpi=300)
    pass

def show_dv(n_layers, n_scales, epochs, dir_output, save='Dv.png'):

    data = np.loadtxt(dir_output + '/Dv.txt')

    for k in range(n_scales):
        fig, ax = plt.subplots(1)
        for i in range(n_layers):
            tmp = list()
            for j in range(epochs):
                tmp.append(i*n_scales+j*n_layers*n_scales+k)
            ax.plot(data[tmp], label='Layer='+str(i)+', Scale='+str(k))    
        ax.set_xlabel('Epochs')
        ax.set_ylabel('D')
        ax.grid()
        ax.legend()
        ax.set_title('Scales'+str(k))
        fig.savefig(os.path.join(dir_output,'Dv'+str(k)+'.png' ), dpi=300)
    pass

def show_reconstruction_loss(dir_output, save='reconstruction_loss.png'):

    data = np.loadtxt(dir_output + '/loss_reconstruction.txt')
    fig, ax = plt.subplots(1)
    ax.plot(data[1:])
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Reconstruction Loss')
    ax.grid()
    ax.set_title('Reconstruction Loss')
    fig.savefig(os.path.join(dir_output, save), dpi=300)
    pass


def show_clustered_signal(stream, dir_output, num_cluster,x,y,pooling_x,pooling_y):

    proba = np.loadtxt(dir_output + '/proba.txt')
    Itest = list()
    for k in range(num_cluster):

        Ix = np.zeros((x,y),dtype=float)
        for j in range(x//pooling_x):
            for i in range(y//pooling_y):
                Ix[j*pooling_x:(j+1)*pooling_x, i*pooling_y:(i+1)*pooling_y] = stream[j*pooling_x:(j+1)*pooling_x, i*pooling_y:(i+1)*pooling_y] * proba[ (j*(y//pooling_y) + i)*num_cluster + k]
        Itest.append(Ix)
        fig, ax = plt.subplots(1, figsize=(16,9))
        for l in range(x):
            ax.plot(Ix[l,:]+200*l,c='k',lw=0.2)
        ax.set_xlabel('Times')
        ax.set_ylabel('Channels')
        ax.grid()
        ax.set_title('Signals of different clusters')
        save='Clustered signals of cluster '+ str(k+1) + '.png'
        fig.savefig(os.path.join(dir_output, save), dpi=300)

    pass
