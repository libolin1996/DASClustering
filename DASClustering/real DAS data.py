import oct2py
oc = oct2py.Oct2Py()
import fdct_tf
import curvenet as cn
import numpy as np
import imageio
import os
import scipy.io
import noise
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
import tensorflow as tf
from matplotlib import pyplot as plt
from obspy import read
from scipy.io import savemat
from scipy.signal import butter, sosfiltfilt

pooling_channels = 1
pooling_time = 1
n_batch_channel = 1
n_batch_time = 1
space = 1
noise_level = 1
sampling_rate = 200

args = cn.io.parse_arguments('./example_real.yaml.py')
n_pca = args['pca']['n_components']
n_clusters = args['gmm_init']['n_components']
summary = cn.io.Summary(args['summary'])
summary.save_args()

stream = cn.data.read_data(**args['data']['load'])
stream.filter(**args['data']['filter'])
stream.taper(**args['data']['taper'])
stream = np.array(stream)
stream = stream[200:999, 175:300]
#stream = stream[220:1019, 3210:3340]

sos = butter(4, 0.3, output='sos', btype='high')
stream = sosfiltfilt(sos, stream, axis=1)

scale = 20.0
world = np.zeros((np.shape(stream)[0], np.shape(stream)[1]))
for i in range(np.shape(stream)[0]):
    for j in range(np.shape(stream)[1]):
        world[i][j] = noise.pnoise2(i/scale,
                                     j/scale,
                                     octaves=6,
                                     persistence=0.5,
                                     lacunarity=2.0,
                                     repeatx=1024,
                                     repeaty=1024,
                                     base=42)
stream += noise_level*world
stream_all = stream
batch_channel_size = np.array(stream.shape)[0]//n_batch_channel
batch_time_size = np.array(stream.shape)[1]//n_batch_time


plt.rcParams.update({'font.size': 32})
plt.figure(figsize=(16, 5))
vm = 15
plt.imshow(stream_all, aspect='auto', cmap='gray', extent=[0, stream_all.shape[1]/sampling_rate, 0, stream_all.shape[0]], vmin=-vm,vmax=vm, origin='lower')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=32)
cbar.set_label('Strain rate (/s)', fontsize=40)
plt.xlabel('Time (s)', fontsize=44)
plt.ylabel('Distance (m)', fontsize=44)
plt.yticks(np.arange(0, stream_all.shape[0] + 1, 200))
plt.savefig('./summaries/example_real.yaml/stream_imshow.pdf', format='pdf', bbox_inches='tight')
plt.clf()

#parameters:
n_layers = 2
n_scales = 2
n_coarses = 8
fre_per = 1/2
ratio = 10
ruler = min(batch_channel_size,batch_time_size)
lt = [1] + [1 / (2**i) for i in range(n_scales-1, 0, -1)]
lt = lt * n_layers
D = ruler*fre_per*np.array(lt)
D = D/ratio

g = tf.Graph()
config = tf.compat.v1.ConfigProto()
with tf.compat.v1.Session(graph=g, config=config) as sess:

    with g.as_default():

        Dv = tf.Variable(D, name='Dv', trainable=True)
        init = tf.compat.v1.placeholder(tf.float64, shape=(batch_channel_size,batch_time_size))
        status = 0
        C = fdct_tf.fdct_wrapping_var(init,0,1,n_scales,n_coarses,Dv*ratio,fre_per,status)   
        n_features = len(C)
        C_tmp = C
        for i in range(1, n_layers):
            status += 1
            C_tmp = fdct_tf.fdct_wrapping_var(C_tmp[0],0,1,n_scales,n_coarses,Dv*ratio,fre_per,status)
            for j in range(len(C_tmp)):
                C.append(C_tmp[j])
        n_features_all = len(C)
        [channel_size_new, time_size_new] = C[0].shape
        pooleds = tf.image.resize(tf.expand_dims(init, axis=-1), (channel_size_new, time_size_new))
        for i in range(n_features_all):
            pooled = tf.image.resize(tf.expand_dims(tf.cast(tf.math.real(C[i]), dtype=tf.float32), axis=-1), (channel_size_new, time_size_new))
            pooleds = tf.concat([pooleds,pooled], axis = 2)
        n_features_all += 1

        Weight = (np.ones(n_features_all)).astype('float')
        Weights = tf.Variable(Weight, name='Weights', trainable=True)
        a = list()
        bias = 1
        tmp0 = 0
        for i in range(1, n_features_all):
            print(i)
            tmp0 = tmp0 + Weights[i]
        tmp = pooleds[:,:,0] * tf.cast((n_features_all + bias - tmp0) * Weights[0], dtype=tf.float32)
        a.append(tmp)
        for i in range(1, n_features_all):
            tmp = pooleds[:,:,i] * tf.cast(Weights[i], dtype=tf.float32)
            if (i-1)%n_features==0:
                tmp *= bias/1
            a.append(tmp)
        Weighted0 = tf.stack(a)
        b = list()
        for i in range(n_features_all):
            tmp = Weighted0[i,:,:]/tf.reduce_max(tf.reduce_max(Weighted0[i,:,:],axis=0),axis=0)
            b.append(tmp)
        Weighted = tf.stack(b)
        Weighted = tf.image.resize(tf.transpose(Weighted0,[1,2,0]), (batch_channel_size//pooling_channels, batch_time_size//pooling_time))
        Weighted = tf.transpose(Weighted,[2,0,1])
        Weighted = tf.reshape(Weighted, [Weighted.get_shape().as_list()[0], -1])
        Weighted = tf.transpose(Weighted)
        Weighted = tf.cast(Weighted, dtype=tf.float64)
        C_w = tf.compat.v1.placeholder(tf.float64, (n_features_all, n_pca))
        C_bar = tf.compat.v1.placeholder(tf.float64, shape=(1, n_features_all))
        C_proj = tf.matmul(Weighted - C_bar, C_w)

        mu = tf.compat.v1.placeholder(tf.float64, shape=(n_clusters, n_pca))
        cov = tf.compat.v1.placeholder(tf.float64, shape=(n_clusters, n_pca, n_pca))
        tau = tf.compat.v1.placeholder(tf.float64, shape=(n_clusters,))
        loss, cat = cn.layer.gmm(mu, cov, tau, C_proj, **args['gmm'])
        y = tf.argmax(cat, axis=1)
        #y: which cluster is every point belongs to, (64,1)
        learn_rate = tf.compat.v1.placeholder(tf.float64)
        optimizer = tf.compat.v1.train.AdamOptimizer(learn_rate)
        minimizer = optimizer.minimize(loss)

    sess.run(tf.compat.v1.global_variables_initializer())
    with tf.device('/CPU:0'):

        cost = 0
        pca_op = PCA(n_components = n_pca)
        gmm_op = GaussianMixture(**args['gmm_init'])
        epochs = args['learning']['epochs']
        learning_rate = args['learning']['rate']

        for epoch in range(epochs):

            print('epoch:',epoch)
            # Waitbar
            summary.watch(epoch, epochs)
            learning_rate /= 1.0015
            pooleds_b = list()
            for b_c in range(n_batch_channel):
                for b_t in range(n_batch_time):
                    stream_batch = stream[ b_c * batch_channel_size : (b_c+1) * batch_channel_size , b_t * batch_time_size : (b_t+1) * batch_time_size ]
                    feed = {
                            init: stream_batch
                        }
                    pooleds_2 = sess.run(pooleds, feed_dict=feed)
                    pooleds_b.append(pooleds_2)
            pooleds_all_tmp = np.concatenate(pooleds_b, axis=2)
            pooleds_all_tmp1 = pooleds_all_tmp
            pooleds_all_tmp = resize(pooleds_all_tmp,(channel_size_new//pooling_channels, time_size_new//pooling_time, pooleds_all_tmp.shape[2] ))
            pooleds_all_tmp = pooleds_all_tmp.reshape(-1,pooleds_all_tmp.shape[2])
            pooleds_all_tmp *= 10000
            pooleds_all_tmp[np.isnan(pooleds_all_tmp)] = np.log(args['eps_log'])
            pooleds_all_tmp[np.isinf(pooleds_all_tmp)] = np.log(args['eps_log'])
            pcaed = pca_op.fit_transform(pooleds_all_tmp)
            if epoch==0:
                pcaed0 = pcaed
            scat_w = pca_op.components_.T        
            gmm_op.fit(pcaed)

            means = gmm_op.means_.astype(np.float64)
            weights = gmm_op.weights_.astype(np.float64)
            covariances = gmm_op.covariances_.astype(np.float64)
            summary.save_hot(pcaed, gmm_op, pca_op)
            summary.save_scalar('loss_clustering', cost)
            summary.save_Weights('Weights', Weights.eval())
            summary.save_Weights('Dv', Dv.eval())

            cost = [0]
            feed = {
                            init: stream,
                            learn_rate: learning_rate, C_w: scat_w, 
                            C_bar: np.reshape(pca_op.mean_, (1, -1)),
                            tau: weights, cov: covariances, mu: means
                        }
            sess.run(minimizer, feed_dict=feed)
            c = sess.run(loss, feed_dict=feed)
            cost.append(c)
            cost = np.mean(cost)

cn.display.show_clustering_loss(dir_output='./summaries/example_real.yaml')
n_f_all = 1+n_layers*9
cn.display.show_weights(n_f_all, epochs, dir_output='./summaries/example_real.yaml')
cn.display.show_dv(n_layers, n_scales, epochs, dir_output='./summaries/example_real.yaml')
fig, ax = cn.display.show_clusters(epoch=0,dir_output='./summaries/example_real.yaml',save='clusters_epoch=0.png')
cn.display.show_clusters(epoch=epochs-1,dir_output='./summaries/example_real.yaml',save='clusters_epoch=' + str(epochs-1) + '.png')
summary.save_proba('proba', epoch=epochs-1, dir_output='./summaries/example_real.yaml', num_cluster = args['gmm_init']['n_components'])
summary.save_proba('proba0', epoch=0, dir_output='./summaries/example_real.yaml', num_cluster = args['gmm_init']['n_components'])
proba = np.loadtxt('./summaries/example_real.yaml/proba.txt')
proba0 = np.loadtxt('./summaries/example_real.yaml/proba0.txt')
C_new = oc.fdct_wrapping(stream_all,1,1,n_scales,n_coarses)

nbangles = n_coarses * 2**(np.ceil((n_scales - np.arange(n_scales, 1, -1))/2))
nbangles = np.insert(nbangles,0,1)
nbangles = nbangles.astype(int)
num_point = (channel_size_new//pooling_channels)*(time_size_new//pooling_time)
xtest_l=[]
xtest0_l=[]
fig_all, axes = plt.subplots(int(n_clusters//2),2)
for k in range(n_clusters):
    C_new0 = oc.fdct_wrapping(stream_all,1,1,n_scales,n_coarses)
    C_new00 = oc.fdct_wrapping(stream_all,1,1,n_scales,n_coarses)
    p = (proba[ k*num_point : (k+1)*num_point ]).reshape(channel_size_new//pooling_channels,time_size_new//pooling_time)
    p0 = (proba0[ k*num_point : (k+1)*num_point ]).reshape(channel_size_new//pooling_channels,time_size_new//pooling_time)
    for jj in range(n_scales):
        for ii in range(nbangles[jj]):
            [Cn1,Cn2] = C_new0[0, jj][0, ii].shape
            p = resize(p,(Cn1,Cn2))
            p0 = resize(p0,(Cn1,Cn2))
            for j in range(Cn1):
                for i in range(Cn2):
                    C_new0[0, jj][0, ii][j, i] = C_new0[0, jj][0, ii][j, i] * p[j, i]
                    C_new00[0, jj][0, ii][j, i] = C_new00[0, jj][0, ii][j, i] * p0[j, i]
    xtest = oc.ifdct_wrapping(C_new0,1,stream_all.shape[0],stream_all.shape[1])
    fig, ax = plt.subplots(1, figsize=(16,6))
    plt.imshow(xtest, aspect='auto', cmap='gray', extent=[0, stream_all.shape[1]/sampling_rate, 0, stream_all.shape[0]])
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('Distance(m)')
    plt.savefig('./summaries/example_real.yaml/Clustered signals of cluster '+ str(k+1) + ' imshow.png')
    plt.clf()
    xtest_l.append(xtest)
    xtest0 = oc.ifdct_wrapping(C_new00,1,stream_all.shape[0],stream_all.shape[1])
    plt.clf()
    plt.imshow(np.real(xtest0), aspect='auto', cmap='gray', extent=[0, stream_all.shape[1]/sampling_rate, 0, stream_all.shape[0]], vmin=-vm,vmax=vm, origin='lower')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('Distance(m)')
    plt.savefig('./summaries/example_real.yaml/Original clustered signals of cluster '+ str(k+1) + ' imshow.png')
    plt.clf()
    im = axes.flatten()[k].imshow(np.real(xtest0), aspect='auto', cmap='gray', extent=[0, stream_all.shape[1]/sampling_rate, 0, stream_all.shape[0]])
    axes.flatten()[k].set_xlabel('Time (s)')
    axes.flatten()[k].set_ylabel('Distance(m)')
    xtest0_l.append(xtest0)


fig_all.colorbar(im, ax=axes.ravel().tolist())
fig_all.savefig('./summaries/example_real.yaml/Original clustered signals of different clusters imshow.png')
plt.clf()

xtest_ll = xtest_l
xtest_l = xtest0_l
x_cluster = list()
x_cluster1 = xtest_l[0] + xtest0_l[3] + xtest_l[5]+ xtest_l[9]
#x_cluster1 = xtest_l[2] + xtest0_l[4] + xtest_l[5] + xtest0_l[6] + xtest_l[8] + xtest_l[9]
#x_cluster4 = xtest0_l[1]
x_cluster2 = stream - x_cluster1

x_cluster.append(x_cluster2)
x_cluster.append(x_cluster1)

num_cluster_fin = 2
fig_all, axes = plt.subplots(num_cluster_fin,1,sharex=True, sharey=True)
for k in range(num_cluster_fin):
    fig, ax = plt.subplots(1, figsize=(16,9))
    for l in range(np.array(stream_all.shape)[0]): 
        ax.plot(x_cluster[k][l,:]+space*l,c='k',lw=0.2)

    ax.set_xlabel('Times')
    ax.set_ylabel('Distance(m)')
    ax.grid()
    ax.set_title('Signals of different clusters')
    save='The clustered signals of cluster '+ str(k+1) + '.png'
    fig.savefig(os.path.join('./summaries/example_real.yaml', save), dpi=300)
    plt.clf()
    plt.imshow(x_cluster[k], aspect='auto', cmap='gray', extent=[0, stream_all.shape[1]/sampling_rate, 0, stream_all.shape[0]], vmin=-vm, vmax=vm)
    plt.colorbar(orientation='horizontal')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance(m)')
    plt.savefig('./summaries/example_real.yaml/The clustered signals of cluster '+ str(k+1) + ' imshow.png')
    plt.clf()

    im = axes.flatten()[k].imshow(x_cluster[k], aspect='auto', cmap='gray', extent=[0, stream_all.shape[1]/sampling_rate, 0, stream_all.shape[0]], vmin=-vm, vmax=vm)
    pos = axes[k].get_position()
    new_pos = [pos.x0 + 0.06, pos.y0 + 0.06, pos.width * 0.8, pos.height * 0.8]
    axes[k].set_position(new_pos)

axes[-1].set_xlabel('Time (s)')
fig_all.text(0.1, 0.5, 'Distance(m)', va='center', rotation='vertical')
cbar_ax = fig_all.add_axes([0.85, 0.1, 0.03, 0.8])
fig_all.colorbar(im, ax=axes.ravel().tolist(), cax=cbar_ax)
fig_all.savefig('./summaries/example_real.yaml/The clustered signals of different clusters imshow.png')
plt.clf()


x_clustern = list()
x_clustern.append(xtest_ll[4] + xtest_ll[8])
x_clustern.append(stream-x_clustern[0])
fig_all, axes = plt.subplots(num_cluster_fin,1,sharex=True, sharey=True)
for k in range(num_cluster_fin):
    im = axes.flatten()[k].imshow(x_clustern[k], aspect='auto', cmap='gray', extent=[0, stream_all.shape[1]/sampling_rate, 0, stream_all.shape[0]], vmin=-vm, vmax=vm)
    pos = axes[k].get_position()
    new_pos = [pos.x0 + 0.06, pos.y0 + 0.06, pos.width * 0.8, pos.height * 0.8]
    axes[k].set_position(new_pos)
axes[-1].set_xlabel('Time (s)')
fig_all.text(0.1, 0.5, 'Distance(m)', va='center', rotation='vertical')
cbar_ax = fig_all.add_axes([0.85, 0.1, 0.03, 0.8])
fig_all.colorbar(im, ax=axes.ravel().tolist(), cax=cbar_ax)
fig_all.savefig('./summaries/example_real.yaml/The clustered signals of different clusters of wavelet imshow.png')
plt.clf()


with open('x_cluster.txt', 'w') as f:
    for item in x_cluster:
        f.write(f"{item}\n")

with open('x_clustern.txt', 'w') as f:
    for item in x_clustern:
        f.write(f"{item}\n")


plt.rcParams.update({'font.size': 8})
fig_all, axes = plt.subplots(num_cluster_fin, 2, sharex=True, sharey=True)

for k in range(num_cluster_fin):
    im = axes[k, 0].imshow(x_cluster[k], aspect='auto', cmap='gray', extent=[0, stream_all.shape[1]/sampling_rate, 0, stream_all.shape[0]], vmin=-vm, vmax=vm, origin='lower')
    plt.gca().invert_yaxis()
    pos = axes[k, 0].get_position()
    new_pos = [pos.x0 + 0.024, pos.y0 + 0.205 - 0.2+0.2*k, pos.width * 0.92 , pos.height * 0.5]
    axes[k, 0].set_position(new_pos)

for k in range(num_cluster_fin):
    im = axes[k, 1].imshow(x_clustern[k], aspect='auto', cmap='gray', extent=[0, stream_all.shape[1]/sampling_rate, 0, stream_all.shape[0]], vmin=-vm, vmax=vm, origin='lower')
    plt.gca().invert_yaxis()
    pos = axes[k, 1].get_position()
    new_pos = [pos.x0 - 0.026, pos.y0 + 0.205 - 0.2+0.2*k, pos.width * 0.92 , pos.height * 0.5]
    axes[k, 1].set_position(new_pos)

fig_all.text(0.445, 0.24, 'Time (s)', va='center', fontsize=12)
fig_all.text(0.054, 0.5, 'Distance (m)', va='center', rotation='vertical', fontsize=12)
fig_all.text(0.983, 0.5, 'Strain rate (/s)', va='center', ha='right', rotation='vertical', fontsize=12)

cbar_ax = fig_all.add_axes([0.87, 0.27, 0.025, 0.45])
cbar=fig_all.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)

fig_all.savefig('./summaries/example_real.yaml/NEW The clustered signals of different clusters.pdf', format='pdf', bbox_inches='tight')
plt.clf()
