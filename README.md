# SSLforDSS

Here we provide the codes using constrastive predictive coding (cpc) [1] as a self-supervised learning method (ssl) to increase sample-efficiency of DeepSongSegmenter package [2] which was developed for segmentation of animal songs.

## Instructions
To use this repository you should be already familiar with dss package (https://github.com/janclemenslab/deepss). Moreover, you should have access to some animals' song data.

Contrastive predictive coding was developed as a universal un- or self-supervised learning approach to extract useful representations from high-dimensional data. It accomplishes this by predicting the future in latent space using powerful autoregressive models, nagative samples, and a probabilistic contrastive loss.
For more information regarding this approach, please refer to the original paper [1]. We used the implementation provided in this repository: https://github.com/SPEECHCOG/pc_models_analysis.

Here we bring the results for three different animals' songs data: 1) multi-channel flies data, 2) single-channel flies data, 3) single-channel birds data.
The aim of SSLforDSS is to come up with the best representations across deep hierarchy using unlabeled data to decrease the amount of required labeled data for subsequent training.

Download the files of this repository and run them in an environment which dss was already installed. There is no need to any other python modules to be installed.

Here are the brief descriptions of the five python modules of this repository. To run the codes over above data, just uncomment the designated lines.

1) `cpc_ssl_dss`: it provides the classes and methods for contrastive predictive coding model.
2) `train_ssl_dss`: it performs contrastive predictive coding on our available data. The parameters used for dss and cpc are optimized for each type of data to have the best performance on the available data.
3) `predict_ssl_dss`: using the weights obtained from cpc training, it predicts the performance of dss network on test set.
4) `train_predict_ssl_dss`: it runs training for dss network by changing the fraction of data used for training in the following three cases: 1) starting the dss network from scratch 2) starting the dss network using the cpc pretrained weights and let them to be trained, 3) starting the dss network using the cpc pretrained weights but freeze them (in this case, the ratio of the parameters to be trained to the whole parameters of the network is less than 0.001).
5) `val_loss_ssl_dss`: it plots the validation loss over training epochs using the above training data and conditions.

## References
1. Oord, A. Van Den, Li, Y., & Vinyals, O. (2019). Representation Learning with Contrastive Predictive Coding. ArXiv.
2. Steinfath, E., Palacios, A., Rottschäfer, J., Yuezak, D., & Clemens, J. (2021). Fast and accurate annotation of acoustic signals with deep neural networks. ArXiv.
