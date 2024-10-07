import h5py
import logging
import numpy as np
import os
import yaml

from termcolor import colored
from tqdm import trange
from yaml import Loader
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_arguments(yaml_file):

    yaml_base_tmp = yaml_file.split(os.sep)
    yaml_base =  yaml_base_tmp[-1]

    tag, _ = os.path.splitext(yaml_base)
    args = yaml.load(open(yaml_file).read(), Loader=Loader)
    args['summary']['tag'] = os.path.join(tag)
    args['summary']['yaml_file'] = yaml_file
    return args


class Summary():

    def __init__(self, args):

        self.__dict__ = args
        self.path = os.path.join(self.path, self.tag)
        self.mkdir()
        self.epoch = 0
        self.save_scat = None if self.save_scat == 0 else self.save_scat
        pass

    def mkdir(self, path=None):
        """Make directory, and clean it if already exsit."""
        if path is None:
            path = self.path
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info('{} (created)'.format(path))
        else:
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))
            logging.info('{} (cleaned)'.format(path))
        pass

    def save_args(self, file_name='args.yaml'):
        """Duplicate yaml arguments in the summary directory."""
        file_args = os.path.join(self.path, file_name)
        os.popen('cp {} {}'.format(self.yaml_file, file_args))
        logging.info('{} (done)'.format(file_args))
        pass

    def save_hot(self, features, gmm, pca, dtype=np.float32):
        """Save clustering results."""
        file_name = os.path.join(self.path, 'clusters.h5')
        with h5py.File(file_name, 'a') as file:
            g = file.create_group('epoch_{:05d}'.format(self.epoch))
            g.create_dataset('hot', data=gmm.predict(features))
            g.create_dataset('proba', data=gmm.predict_proba(features))
            g.create_dataset('features', data=features)
            g.create_dataset('means', data=gmm.means_.astype(dtype))
            g.create_dataset('covariance', data=gmm.covariances_.astype(dtype))
            g.create_dataset('eigenvalues', data=pca.explained_variance_)
        logging.debug('{} (done)'.format(file_name))
        pass


    def save_scalar(self, base_name, value):
        """Save clustering results."""
        base_file = '{}.txt'.format(base_name)
        file_name = os.path.join(self.path, base_file)
        with open(file_name, 'a') as file:
            file.write('{}\n'.format(value))
        pass

    def save_proba(self, base_name, epoch, dir_output, num_cluster):
        """Save clustering results."""
        base_file = '{}.txt'.format(base_name)
        file_name = os.path.join(self.path, base_file)
        with h5py.File(dir_output + '/clusters.h5', 'r') as hf:
            proba = hf['epoch_{:05d}'.format(epoch)]['proba'][()]

        with open(file_name, 'a') as file:
            for a in range(num_cluster):
                for i, p in enumerate(proba):
                    file.write(('{}'.format(p[a])).strip("[]"))
                    file.write('\n')
        pass

    def save_Weights(self, base_name, value):
        """Save clustering results."""
        base_file = '{}.txt'.format(base_name)
        file_name = os.path.join(self.path, base_file)
        with open(file_name, 'a') as file:
            for i, p in enumerate(value):
                file.write(('{}'.format(p)).strip("[]"))
                file.write('\n')
        pass


    def watch(self, epoch, epochs=None):
        """Set current epoch."""
        self.epoch = epoch
        self.epochs = 1 if epochs is None else epochs
        pass
