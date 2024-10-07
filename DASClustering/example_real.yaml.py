data:
    load:
        decimation: 8
        file_data: ./FORGE_78-32_iDASv3-P11_UTC190423213209.segy
        #file_data: ./FORGE_78-32_iDASv3-P11_UTC190426070723.segy
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
    rate: 0.01

pca:
    n_components: 3

gmm:
    gmm_type: natural
    trainable: False

gmm_init:
    n_components: 10
    max_iter: 10
    covariance_type: full
    warm_start: True
    random_state: 1

summary:
    path: ./summaries/
    save_scat: 1000

