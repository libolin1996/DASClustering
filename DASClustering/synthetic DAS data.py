import oct2py
oc = oct2py.Oct2Py()
import fdct_tf
import curvenet as cn
import numpy as np
import os
import scipy.io
import noise
import h5py
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
import tensorflow as tf
from matplotlib import pyplot as plt
from obspy import read
from scipy.io import savemat


pooling_channels = 1
pooling_time = 2
n_batch_channel = 1
n_batch_time = 1
space = 50
noise_level = 15
sampling_rate = 100

args = cn.io.parse_arguments('./example.yaml.py')
n_pca = args['pca']['n_components']
n_clusters = args['gmm_init']['n_components']
summary = cn.io.Summary(args['summary'])
summary.save_args()
stream = oc.syn_data_new()
stream_lin = oc.syn_data_lin()
stream += stream_lin
stream = stream.T
stream *= 100000
np.random.seed(42)

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
stream = resize(stream, (120, 800))
batch_channel_size = np.array(stream.shape)[0]//n_batch_channel
batch_time_size = np.array(stream.shape)[1]//n_batch_time

plt.rcParams.update({'font.size': 32})
fig, ax = plt.subplots(figsize=(16, 9))
im = ax.imshow(stream_all, aspect=0.03, cmap='gray', extent=[0, stream_all.shape[1]/sampling_rate, 0, stream_all.shape[0]], vmin=-400, vmax=500)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Strain rate (/s)', fontsize=40)
ax.set_xlabel('Time (s)', fontsize=44)
ax.set_ylabel('Distance (m)', fontsize=44)
cbar.ax.set_position([cbar.ax.get_position().x0, ax.get_position().y0, cbar.ax.get_position().width, ax.get_position().height])
cbar.ax.tick_params(labelsize=24)
plt.savefig('./summaries/example.yaml/stream_imshow.pdf', format='pdf', bbox_inches='tight')
plt.clf()

#parameters:
n_layers = 3
n_scales = 2
n_coarses = 8
fre_per = 1/2
ratio = 10

ruler = min(batch_channel_size,batch_time_size)
lt = [1] + [1 / (2**i) for i in range(n_scales-1, 0, -1)]
lt = lt * n_layers
#lt=[1,1/4,1/2]
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
            tmp0 = tmp0 + Weights[i]
        tmp = pooleds[:,:,0] * tf.cast((n_features_all + bias - tmp0) * Weights[0], dtype=tf.float32) #/tf.reduce_max(tf.reduce_max(pooleds[:,:,0],axis=0),axis=0)
        a.append(tmp)
        for i in range(1, n_features_all):
            tmp = pooleds[:,:,i] * tf.cast(Weights[i], dtype=tf.float32) #/tf.reduce_max(tf.reduce_max(pooleds[:,:,i],axis=0),axis=0)
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

            summary.watch(epoch, epochs)
            learning_rate /= 1.002
            pooleds_b = list()
            C_all = list()
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



n_c = np.arange(3, 7)

models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(pcaed0)
          for n in n_c]
fig, ax = plt.subplots(1, figsize=(16,9))
ax.plot(n_c, [silhouette_score(pcaed0, m.predict(pcaed0)) for m in models])
ax.set_xlabel('Number of Components', fontsize=20)
ax.set_ylabel('Silhouette Score', fontsize=20)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
fig.savefig(os.path.join('./summaries/example.yaml', 'Silhouette Analysis.png'), dpi=300)

fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
axes = axes.flatten()

for idx, n_clu in enumerate(n_c):

    ax1 = axes[idx]
    ax1.set_xlim([-0.2, 1])
    ax1.set_ylim([0, len(pcaed0) + (n_clu + 1) * 10])

    clusterer = GaussianMixture(n_clu, covariance_type='full', random_state=0).fit(pcaed0)
    cluster_labels = clusterer.predict(pcaed0)

    silhouette_avg = silhouette_score(pcaed0, cluster_labels)
    print(
        "For n_clusters =",
        n_clu,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    sample_silhouette_values = silhouette_samples(pcaed0, cluster_labels)

    y_lower = 10
    colors = ['#00FF00', '#FF0088', '#0000FF', '#00FFFF', '#FFFF00', '#8800FF', '#FF8800', '#00FF88', '#FF0000', '#FF00FF']
    colors = colors[:n_clu]
    for i in range(n_clu):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = colors[i]
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        if n_clu > 4 and i == 1:
            alpha=-130
        elif n_clu == 5 and i == 4:
            alpha=-110
        else:
            alpha=0
        ax1.text(-0.05, y_lower + alpha + 0.5 * size_cluster_i, str(i + 1), fontsize=20)

        y_lower = y_upper #+ 10  # 10 for the 0 samples


    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.set_title(f"The number of clusters is {n_clu}", fontsize=28)

fig.text(0.04, 0.5, 'Cluster label', va='center', rotation='vertical', fontsize=32)
fig.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.15, wspace=0.15, hspace=0.2)
fig.savefig(os.path.join('./summaries/example.yaml', 'Silhouette plots 2x2.pdf'), 
            format='pdf', bbox_inches='tight', dpi=300)
plt.clf()

fig, ax = cn.display.show_clustering_loss(dir_output='./summaries/example.yaml')
fig.savefig('./summaries/example.yaml/clustering_loss.pdf', format='pdf', bbox_inches='tight')

plt.rcParams.update({'font.size': 12})
fig, ax = cn.display.show_clusters(epoch=0,dir_output='./summaries/example.yaml')
fig.subplots_adjust(right=0.85)
fig.savefig('./summaries/example.yaml/clusters_epoch=0.pdf', format='pdf')
fig, ax = cn.display.show_clusters(epoch=epochs-1,dir_output='./summaries/example.yaml')
fig.subplots_adjust(right=0.85)
fig.savefig('./summaries/example.yaml/clusters_epoch=' + str(epochs-1) + '.pdf', format='pdf')
plt.rcParams.update({'font.size': 16})

summary.save_proba('proba', epoch=epochs-1, dir_output='./summaries/example.yaml', num_cluster = args['gmm_init']['n_components'])
proba = np.loadtxt('./summaries/example.yaml/proba.txt')
C_new = oc.fdct_wrapping(stream_all,1,1,n_scales,n_coarses)

nbangles = n_coarses * 2**(np.ceil((n_scales - np.arange(n_scales, 1, -1))/2))
nbangles = np.insert(nbangles,0,1)
nbangles = nbangles.astype(int)
num_point = (channel_size_new//pooling_channels)*(time_size_new//pooling_time)
plt.clf()
for k in range(n_clusters):
    C_new0 = oc.fdct_wrapping(stream_all,1,1,n_scales,n_coarses)
    p = (proba[ k*num_point : (k+1)*num_point ]).reshape(channel_size_new//pooling_channels,time_size_new//pooling_time)
    for jj in range(n_scales):
        for ii in range(nbangles[jj]):
            [Cn1,Cn2] = C_new0[0, jj][0, ii].shape
            p = resize(p,(Cn1,Cn2))
            for j in range(Cn1):
                for i in range(Cn2):
                    C_new0[0, jj][0, ii][j, i] = C_new0[0, jj][0, ii][j, i] * p[j, i]
    xtest = oc.ifdct_wrapping(C_new0,1,stream_all.shape[0],stream_all.shape[1])
    plt.imshow(xtest, aspect=0.035, cmap='gray', extent=[0, stream_all.shape[1]/sampling_rate, 0, stream_all.shape[0]])
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('Distance(m)')
    plt.savefig('./summaries/example.yaml/Clustered signals of cluster '+ str(k+1) + ' imshow.png')
    plt.clf()
