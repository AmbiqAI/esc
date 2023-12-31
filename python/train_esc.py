"""
Training script for ESC
"""
import os
import re
import logging
import argparse
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from nnsp_pack.nn_module import NeuralNetClass, lstm_states, tf_round
from nnsp_pack.tfrecord_converter_esc import tfrecords_pipeline
from nnsp_pack.loss_functions import cross_entropy
from nnsp_pack.converter_fix_point import fakefix_tf
from nnsp_pack.calculate_feat_stats_vad import feat_stats_estimator
from nnsp_pack.load_nn_arch import load_nn_arch, setup_nn_folder
import c_code_table_converter

DIM_TARGET = 5
physical_devices = tf.config.list_physical_devices('GPU')
WEIGHTS=np.ones(DIM_TARGET, dtype=np.float32)
WEIGHTS[1] = 2
WEIGHTS[4] = 3
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: # pylint: disable=bare-except
    pass

SHOW_STEPS          = True
DISPLAY_HISTOGRAM   = False

@tf.function
def train_kernel(   feat, mask, target, states, net,
                    optimizer,
                    training    = True,
                    quantized   = False):
    """
    Training kernel
    """
    with tf.GradientTape() as tape:
        est, states = net(   feat, mask, states,
                            training = training,
                            quantized = quantized)

        est_target  = tf.nn.softmax(est)

        target = tf.one_hot(target, DIM_TARGET, dtype=tf.float32)
        ave_loss, steps = cross_entropy(
                            target,
                            est_target,
                            weights=WEIGHTS,
                            masking = mask)

    if training:
        gradients = tape.gradient(ave_loss, net.trainable_variables)

        gradients_clips = [ tf.clip_by_norm(grad, 1) for grad in gradients ]
        optimizer.apply_gradients(zip(gradients_clips, net.trainable_variables))

    return est_target, states, ave_loss, steps

def epoch_proc( net,
                optimizer,
                dataset,
                fnames,
                batchsize,
                timesteps,
                training,
                zero_state,
                norm_mean,
                norm_inv_std,
                num_dnsampl     = 1,
                num_context     = 6,
                quantized=False
                ):
    """
    Training for one epoch
    """
    total_batches = int(np.ceil(len(fnames) / batchsize))
    net.reset_stats()
    for batch, data in enumerate(dataset):
        tf.print(f"\r {batch}/{total_batches}: ",
                        end = '')

        feats, masks, target, _ = data

        batchsize0, total_steps, dim_feat = feats.shape
    
        if 0:
            idx = 231
            feat = feats[idx,:,:].numpy()
            tar = target[idx,:].numpy() * 25
            plt.figure(1)
            plt.clf()
            plt.imshow(    feat.T,
                            origin      = 'lower',
                            cmap        = 'pink_r',
                            aspect      = 'auto')
            plt.plot(tar)
            plt.show()
        total_steps_extend = int(np.ceil(total_steps / timesteps) * timesteps)

        steps_pad = total_steps_extend - total_steps

        shape_feats = [[0, batchsize - batchsize0], [0, steps_pad], [0, 0]]
        paddings_feats = tf.constant(shape_feats)

        shape_target = [[0, batchsize - batchsize0], [0, steps_pad]]
        paddings_target = tf.constant(shape_target)

        shape_mask = [[0, batchsize - batchsize0], [0, steps_pad], [0, 0]]
        paddings_mask = tf.constant(shape_mask)

        # initial input: 2^-15 in time domain
        shape = (batchsize0, num_context-1, dim_feat)
        padddings_tsteps_zeros = tf.constant(
                        np.full(shape, np.log10(2**-15)),
                        dtype = tf.float32)

        feats = tf.concat([padddings_tsteps_zeros, feats], 1)
        feats = (feats - norm_mean) * norm_inv_std
        feats = fakefix_tf(feats, 16, 8)

        feats = tf.pad(feats, paddings_feats, "CONSTANT")

        target = tf.pad(target, paddings_target, "CONSTANT")

        masks = tf.pad(masks, paddings_mask, "CONSTANT")

        states = lstm_states(net, batchsize, zero_state= zero_state)

        for i in range( int(np.round(total_steps / timesteps))):
            start_fr = i * timesteps
            end_fr = start_fr + timesteps
            feat_part   = feats[:,start_fr:end_fr + num_context-1,:]
            mask_part   = masks[:,start_fr:end_fr:num_dnsampl,:]
            target_part = target[:,start_fr:end_fr:num_dnsampl]

            tmp = train_kernel(
                    feat_part,
                    mask_part,
                    target_part,
                    states,
                    net,
                    optimizer,
                    training    = training,
                    quantized   = quantized)

            est_target, states, ave_loss, steps = tmp
            est_target = tf.math.argmax(est_target, axis=2)

            net.update_cost_steps(ave_loss, steps)

            net.update_accuracy(
                tf.cast(target_part, tf.int64),
                est_target,
                mask_part,
                DIM_TARGET)

        net.show_loss(
            net.stats['acc_loss'],
            net.stats['acc_matchCount'],
            net.stats['acc_steps'],
            SHOW_STEPS)

    tf.print('\n', end = '')

def main(args):
    """
    main function to train neural network training
    """
    batchsize       = args.batchsize
    timesteps       = args.timesteps
    num_epoch       = args.num_epoch
    epoch_loaded    = args.epoch_loaded
    quantized       = args.quantized

    tfrecord_list = {   'train' : args.train_list,
                        'test'  : args.test_list}

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    arch = load_nn_arch(args.nn_arch)
    neurons, drop_rates, layer_types, activations, num_context, num_dnsampl, scalar_output, len_filter, len_lookahead = arch # pylint: disable=unbalanced-tuple-unpacking

    folder_nn = setup_nn_folder(args.nn_arch)

    dim_feat = neurons[0]

    nn_train = NeuralNetClass(
        neurons     = neurons,
        layer_types = layer_types,
        dropRates   = drop_rates,
        activations = activations,
        batchsize   = batchsize,
        nDownSample = num_dnsampl,
        kernel_size = num_context,
        dim_target  = DIM_TARGET)

    nn_infer = NeuralNetClass(
        neurons     = neurons,
        layer_types = layer_types,
        activations = activations,
        batchsize   = batchsize,
        nDownSample = num_dnsampl,
        kernel_size = num_context,
        dim_target  = DIM_TARGET)

    if epoch_loaded == 'random':
        epoch_loaded = -1

        loss = {'train' : np.zeros(num_epoch+1),
                'test' : np.zeros(num_epoch+1)}

        acc  = {'train' : np.zeros(num_epoch+1),
                'test' : np.zeros(num_epoch+1)}
        epoch1_loaded = epoch_loaded + 1
    else:
        if epoch_loaded == 'latest':
            checkpoint_dir = f'{folder_nn}/checkpoints'
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            nn_train.load_weights(latest)
            tmp = re.search(r'_ep(\d)+', latest)
            epoch_loaded = int(re.sub(r'_ep','',tmp.group(0)))
            epoch1_loaded = epoch_loaded + 1
        else:
            nn_train.load_weights(
                f'{folder_nn}/checkpoints/model_checkpoint_ep{epoch_loaded}')
            epoch1_loaded = epoch_loaded + 1

        print(f"Model at epoch {epoch1_loaded - 1} is retrieved")

        with open(os.path.join(folder_nn, 'nn_loss.pkl'), "rb") as file:
            loss = pickle.load(file)
        with open(os.path.join(folder_nn, 'nn_acc.pkl'), "rb") as file:
            acc = pickle.load(file)

        ax_handle = plt.subplot(2,1,1)
        ax_handle.plot(loss['train'][0: epoch_loaded])
        ax_handle.plot(loss['test'][0: epoch_loaded])
        ax_handle.legend(['train', 'test'])
        ax_handle.grid(True)
        ax_handle.set_title(f'Loss and accuracy upto epoch {epoch_loaded}. Close it to continue')

        ax_handle = plt.subplot(2,1,2)
        ax_handle.plot(acc['train'][0: epoch_loaded])
        ax_handle.plot(acc['test'][0: epoch_loaded])
        ax_handle.legend(['train', 'test'])
        ax_handle.set_xlabel('Epochs')

        print(f"(train) max acc epoch = {np.argmax(acc['train'])}")
        print(f"(test)  max acc epoch = {np.argmax(acc['test'])}")

        ax_handle.grid(True)

        plt.show()

    nn_train.duplicated_to(nn_infer, logger)

    fnames = {}
    for tr_set in ['train', 'test']:
        with open(tfrecord_list[tr_set], 'r') as file: # pylint: disable=unspecified-encoding
            try:
                lines = file.readlines()
            except:# pylint: disable=bare-except
                print(f'Can not find the list {tfrecord_list[tr_set]}')
            else:
                fnames[tr_set] = [line.strip() for line in lines]

    _, dataset = tfrecords_pipeline(
            fnames['train'],
            dim_feat,
            batchsize = batchsize,
            is_shuffle = True)

    _, dataset_tr = tfrecords_pipeline(
            fnames['train'],
            dim_feat,
            batchsize = batchsize)

    _, dataset_te = tfrecords_pipeline(
            fnames['test'],
            dim_feat,
            batchsize = batchsize)

    if os.path.exists(f'{folder_nn}/stats.pkl'):
        with open(os.path.join(folder_nn, 'stats.pkl'), "rb") as file:
            stats = pickle.load(file)
    else:
        stats = feat_stats_estimator(
                dataset_tr, fnames['train'],
                batchsize, dim_feat, folder_nn)

    nn_np = c_code_table_converter.tf2np(nn_infer, quantized=quantized)
    if DISPLAY_HISTOGRAM:
        c_code_table_converter.draw_nn_hist(nn_np)

    for epoch in range(epoch1_loaded, num_epoch):
        t_start = tf.timestamp()
        tf.print(f'\n(EP {epoch})\n', end = '')

        # Training phase
        epoch_proc( nn_train,
                    optimizer,
                    dataset,
                    fnames['train'],
                    batchsize,
                    timesteps,
                    training        = True,
                    zero_state      = False,
                    norm_mean       = stats['nMean_feat'],
                    norm_inv_std    = stats['nInvStd'],
                    num_dnsampl     = num_dnsampl,
                    num_context     = num_context,
                    quantized       = quantized)

        nn_train.duplicated_to(
                nn_infer,
                logger)

        # Computing Training loss
        epoch_proc( nn_infer,
                    optimizer,
                    dataset_tr,
                    fnames['train'],
                    batchsize,
                    timesteps,
                    training        = False,
                    zero_state      = True,
                    norm_mean       = stats['nMean_feat'],
                    norm_inv_std    = stats['nInvStd'],
                    num_dnsampl     = num_dnsampl,
                    num_context     = num_context,
                    quantized       = quantized)

        nn_infer.show_confusion_matrix(DIM_TARGET, logger)
        cmat_train = nn_infer.confusion_mat.numpy()
        loss['train'][epoch] = nn_infer.stats['acc_loss'] / nn_infer.stats['acc_steps']
        loss['train'][epoch] /= nn_infer.neurons[-1]

        acc['train'][epoch] = nn_infer.stats['acc_matchCount'] / nn_infer.stats['acc_steps']

        # Computing Testing loss
        epoch_proc( nn_infer,
                    optimizer,
                    dataset_te,
                    fnames['test'],
                    batchsize,
                    timesteps,
                    training            = False,
                    zero_state          = True,
                    norm_mean           = stats['nMean_feat'],
                    norm_inv_std        = stats['nInvStd'],
                    num_dnsampl         = num_dnsampl,
                    num_context         = num_context,
                    quantized           = quantized)

        nn_infer.show_confusion_matrix(DIM_TARGET, logger)
        cmat_test = nn_infer.confusion_mat.numpy()
        loss['test'][epoch] = nn_infer.stats['acc_loss'] / nn_infer.stats['acc_steps']
        loss['test'][epoch] /= nn_infer.neurons[-1]

        acc['test'][epoch] = nn_infer.stats['acc_matchCount'] / nn_infer.stats['acc_steps']

        nn_train.save_weights(f'{folder_nn}/checkpoints/model_checkpoint_ep{epoch}')

        with open(os.path.join(folder_nn, 'nn_loss.pkl'), "wb") as file:
            pickle.dump(loss, file)
        with open(os.path.join(folder_nn, 'nn_acc.pkl'), "wb") as file:
            pickle.dump(acc, file)

        print(f"(train) max acc epoch = {np.argmax(acc['train'][: epoch+1])}")
        print(f"(test)  max acc epoch = {np.argmax(acc['test'][: epoch+1])}")
        tf.print('Epoch spent ', tf_round(tf.timestamp() - t_start), ' seconds')

        plt.figure(2)
        plt.subplot(2,1,1)
        plt.imshow(cmat_train/np.sum(cmat_train, axis = -1,keepdims=True), cmap='gray_r')
        plt.colorbar()
        plt.title(f'train (epoch {epoch})')

        plt.subplot(2,1,2)
        plt.imshow(cmat_test/np.sum(cmat_test, axis = -1,keepdims=True), cmap='gray_r')
        plt.colorbar()
        plt.title('test')
        plt.savefig(f'{folder_nn}/confusedMat.png')
        plt.clf()

        plt.figure(1)
        ax_handle = plt.subplot(2,1,1)
        ax_handle.plot(loss['train'][: epoch+1])
        ax_handle.plot(loss['test'][: epoch+1])
        ax_handle.legend(['train', 'test'])
        ax_handle.grid(True)

        ax_handle = plt.subplot(2,1,2)
        ax_handle.plot(acc['train'][: epoch+1])
        ax_handle.plot(acc['test'][: epoch+1])
        ax_handle.legend(['train', 'test'])
        ax_handle.grid(True)
        ax_handle.set_xlabel('Epochs')
        plt.savefig(f'{folder_nn}/hist_loss_acc.png')
        plt.clf()

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(
        description='Training script for kws model')

    argparser.add_argument(
        '-a',
        '--nn_arch',
        default='nn_arch/def_esc_nn_arch.txt',
        help='nn architecture')

    argparser.add_argument(
        '-st',
        '--train_list',
        default='data/train_tfrecords.csv',
        help='train_list')

    argparser.add_argument(
        '-ss',
        '--test_list',
        default='data/test_tfrecords.csv',
        help='test_list')

    argparser.add_argument(
        '-b',
        '--batchsize',
        default=500,
        type=int,
        help='Batch size for training and validation')

    argparser.add_argument(
        '-t',
        '--timesteps',
        default=500,
        type=int,
        help='rnn timesteps for training and validation')

    argparser.add_argument(
        '-q',
        '--quantized',
        default = False,
        type=bool,
        help='is post quantization?')

    argparser.add_argument(
        '-l',
        '--learning_rate',
        default = 4 * 10**-4,
        type=float,
        help='learning rate')

    argparser.add_argument(
        '-e',
        '--num_epoch',
        type=int,
        default=3000,
        help='Number of epochs to train')

    argparser.add_argument(
        '--epoch_loaded',
        default= "random",
        help='epoch_loaded = \'random\': weight table is randomly generated, \
              epoch_loaded = \'latest\': weight table is loaded from the latest saved epoch result \
              epoch_loaded = 10  \
              (or any non-negative integer): weight table is loaded from epoch 10')

    main(argparser.parse_args())
