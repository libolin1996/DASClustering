data:
    load:
        decimation: 8
    filter:
        type: highpass
        freq: 0.1
        corners: 2
    taper:
        max_percentage: 0.5
        max_length: 60
    batch:
        batch_size: 16
        patch_shape: 16384

eps_norm: 0.001
eps_log: 0.0005

learning:
    epochs: 51
    rate: 0.00006

pca:
    n_components: 3

gmm:
    gmm_type: natural
    trainable: False

gmm_init:
    n_components: 4
    max_iter: 10
    covariance_type: full
    warm_start: True
    random_state: 1

summary:
    path: ./summaries/
    save_scat: 1000

