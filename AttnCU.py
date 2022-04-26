import numpy as np
import scipy.io as sio
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import h5py
import hdf5storage
from ops import *
from utils import *
from functools import reduce


class Attn_ComUnq():

    def __init__(self, sess, data_dir, batch_size, hidden_v, hidden_a, hidden_t, LSTM_hid, text_out,
                 Filters_AVT, Filters_AT, Filters_VT, Filters_AV):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          data_dir: path to the director of the dataset
        """
        self.sess       = sess
        self.y_dim      = 1

        self.data_dir   = data_dir
        self.batch_size = batch_size

        self.hv             = hidden_v
        self.ha             = hidden_a
        self.ht             = hidden_t
        self.LSTM_hid       = LSTM_hid
        self.t_out          = text_out
        self.Filters_AVT    = Filters_AVT
        self.Filters_AV     = Filters_AV
        self.Filters_AT     = Filters_AT
        self.Filters_VT     = Filters_VT

        #### This is for tri
        self.c_bn_a = batch_norm(name  ='cbn_a')
        self.c_bn1_a = batch_norm(name ='cbn1_a')
        self.c_bn2_a = batch_norm(name='cbn2_a')
        self.c_bn3_a = batch_norm(name='cbn3_a')

        self.c_bn_v = batch_norm(name='cbn_v')
        self.c_bn1_v = batch_norm(name='cbn1_v')
        self.c_bn2_v = batch_norm(name='cbn2_v')
        self.c_bn3_v = batch_norm(name='cbn3_v')

        self.conv_bn  = batch_norm(name='conv_bn')
        self.conv_bn1 = batch_norm(name='conv_bn1')
        self.conv_bn2 = batch_norm(name='conv_bn2')
        self.conv_bn3 = batch_norm(name='conv_bn3')

        #### This is for AT
        self.c_bn_at_a = batch_norm(name  ='cbn_at')
        self.c_bn1_at_a = batch_norm(name ='cbn1_at')
        self.c_bn2_at_a = batch_norm(name='cbn2_at')
        self.c_bn3_at_a = batch_norm(name='cbn3_at')

        self.conv_bn_at  = batch_norm(name='conv_bn_at')
        self.conv_bn1_at = batch_norm(name='conv_bn1_at')
        self.conv_bn2_at = batch_norm(name='conv_bn2_at')
        self.conv_bn3_at = batch_norm(name='conv_bn3_at')

        #### This is for VT
        self.c_bn_vt_v = batch_norm(name='cbn_vt')
        self.c_bn1_vt_v = batch_norm(name='cbn1_vt')
        self.c_bn2_vt_v = batch_norm(name='cbn2_vt')
        self.c_bn3_vt_v = batch_norm(name='cbn3_vt')


        self.conv_bn_vt  = batch_norm(name='conv_bn_vt')
        self.conv_bn1_vt = batch_norm(name='conv_bn1_vt')
        self.conv_bn2_vt = batch_norm(name='conv_bn2_vt')
        self.conv_bn3_vt = batch_norm(name='conv_bn3_vt')

        #### This is for AV
        self.c_bn_av_v = batch_norm(name='cbn_av_v')
        self.c_bn1_av_v = batch_norm(name='cbn1_av_v')
        self.c_bn2_av_v = batch_norm(name='cbn2_av_v')
        self.c_bn3_av_v = batch_norm(name='cbn3_av_v')

        self.c_bn_av_a = batch_norm(name='cbn_av_a')
        self.c_bn1_av_a = batch_norm(name='cbn1_av_a')
        self.c_bn2_av_a = batch_norm(name='cbn2_av_a')
        self.c_bn3_av_a = batch_norm(name='cbn3_av_a')

        self.conv_bn_av  = batch_norm(name='conv_bn_av')
        self.conv_bn1_av = batch_norm(name='conv_bn1_av')
        self.conv_bn2_av = batch_norm(name='conv_bn2_av')
        self.conv_bn3_av = batch_norm(name='conv_bn3_av')


        # batch normalization for unique Parts
        self.v_bn = batch_norm(name='video_subnet')
        self.v1_bn = batch_norm(name='video_subnet1')
        self.v2_bn = batch_norm(name='video_subnet2')
        self.v3_bn = batch_norm(name='video_subnet3')
        self.a_bn = batch_norm(name='audio_subnet')
        self.a1_bn = batch_norm(name='audio_subnet1')
        self.a2_bn = batch_norm(name='audio_subnet2')

        self.c_bn = batch_norm(name='conv_bn')
        self.c_bn01 = batch_norm(name='conv_bn01')
        self.c_bn02 = batch_norm(name='conv_bn02')
        self.c_bn1 = batch_norm(name='conv_bn1')
        self.c_bn11 = batch_norm(name='conv_bn11')
        self.c_bn2 = batch_norm(name='conv_bn2')
        self.c_bn21 = batch_norm(name='conv_bn21')
        self.c_bn3 = batch_norm(name='conv_bn3')
        self.c_bn31 = batch_norm(name='conv_bn31')


        self.build_model()

    def build_model(self):

        # audio_data, text_data, video_data, _, _ , _, _, _, _, _, _, _, = self.load_mosei()
        audio_data, text_data, video_data, *_ = self.load_mosei()

        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        video_dim = video_data.shape

        audio_dim = audio_data.shape

        text_dim = text_data.shape
        print('text dim --->',text_dim)

        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        self.video_inputs = tf.placeholder(tf.float32, [None, video_dim[1], video_dim[2], video_dim[3]],
                                           name='video_data')

        self.audio_inputs = tf.placeholder(tf.float32, [None, audio_dim[1], audio_dim[2], audio_dim[3]],
                                           name='audio_data')

        self.text_inputs = tf.placeholder(tf.float32, [None, text_dim[1], text_dim[2]], name='text_data')

        self.drop_ratio = tf.placeholder(tf.float32, [ ], name='dratio')

        self.drop_LSTM = tf.placeholder(tf.float32, [1], name='drLSTM')


        #### Loss from 3D CNN
        self.D_logits_3D, self.D0 = self.TFConv_train(self.video_inputs, self.audio_inputs, self.text_inputs,
                                             self.hv, self.ha, self.ht, self.LSTM_hid, self.t_out,
                                             dropl=self.drop_ratio, reuse=False)

        self.D_logits_AT, self.AT_0 = self.TFConv_train_AT(self.audio_inputs, self.text_inputs, self.ha, self.ht, self.LSTM_hid,
                                                self.t_out, dropl=self.drop_ratio, reuse=False)

        self.D_logits_VT, self.VT_0  = self.TFConv_train_VT(self.video_inputs, self.text_inputs, self.hv, self.ht, self.LSTM_hid,
                                                self.t_out, dropl=self.drop_ratio, reuse=False)

        self.D_logits_AV, self.AV_0  = self.TFConv_train_AV(self.video_inputs, self.audio_inputs, self.hv, self.ha,
                                                            dropl=self.drop_ratio, reuse=False)

        self.D_logits_FMA, self.h0_A = self.FMA_train(self.audio_inputs, self.ha, dropl=self.drop_ratio, reuse=False)

        self.D_logits_FMV, self.h0_V  = self.FMV_train(self.video_inputs, self.hv, dropl=self.drop_ratio, reuse=False)

        self.D_logits_FMT, self.h0_T  = self.FMT_train(self.text_inputs, self.ht, self.LSTM_hid, self.t_out,
                                            dropl=self.drop_ratio, reuse=False)


        #### This part is logits

        self.D_logits_3D_, self.D0_ = self.TFConv_test(self.video_inputs, self.audio_inputs, self.text_inputs,
                                             self.hv, self.ha,self.ht, self.LSTM_hid, self.t_out)

        self.D_logits_AT_, self.AT_0_  = self.TFConv_test_AT(self.audio_inputs, self.text_inputs, self.ha, self.ht, self.LSTM_hid,
                                                self.t_out)

        self.D_logits_VT_, self.VT_0_ = self.TFConv_test_VT(self.video_inputs, self.text_inputs, self.hv, self.ht, self.LSTM_hid,
                                                self.t_out)

        self.D_logits_AV_, self.AV_0_ = self.TFConv_test_AV(self.video_inputs, self.audio_inputs, self.hv, self.ha)

        self.D_logits_FMA_, self.h0_A_  = self.FMA_test(self.audio_inputs, self.ha)

        self.D_logits_FMV_, self.h0_V_  = self.FMV_test(self.video_inputs, self.hv)

        self.D_logits_FMT_, self.h0_T_  = self.FMT_test(self.text_inputs, self.ht, self.LSTM_hid, self.t_out)

        #### Used for attention fusion
        self.logits = self.fusion_train(self.h0_A, self.h0_V, self.h0_T, self.AT_0, self.VT_0, self.AV_0, self.D0, reuse=False)
        self.logits_ = self.fusion_test(self.h0_A_, self.h0_V_, self.h0_T_, self.AT_0_, self.VT_0_, self.AV_0_, self.D0_)

        #### Losss fucntion for training DeepCU
        self.cnn_loss = tf.losses.absolute_difference(labels=self.y, predictions=self.logits)
        self.Accuracy = tf.reduce_sum(tf.abs(tf.subtract(self.y, tf.clip_by_value(self.logits_, -3.0, 3.0))))
        self.Pred = tf.clip_by_value(self.logits_, -3.0, 3.0)
        self.diff = tf.abs(tf.subtract(self.y, tf.clip_by_value(self.logits_, -3.0, 3.0)))

        self.saver = tf.train.Saver()


    #### Train the Network

    def train(self, config):

        if (config.Optimizer == "Adam"):
            cnn_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=config.beta1) \
                .minimize(self.cnn_loss)
        elif (config.Optimizer == "RMS"):
            cnn_optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cnn_loss)
        else:
            cnn_optim = tf.train.MomentumOptimizer(self.learning_rate, config.momentum).minimize(self.cnn_loss)

        tf.global_variables_initializer().run()

        Au_trdat, Tx_trdat, Vi_trdat, Au_trlab, Au_tsdat, Tx_tsdat, Vi_tsdat, Au_tslab, Au_vdat, \
                        Tx_vdat, Vi_vdat, Au_vlab = self.load_mosei()

        train_batches = Au_trdat.shape[0] // self.batch_size
        val_batches = Au_vdat.shape[0] // self.batch_size

        left_index_train = Au_trdat.shape[0] - (train_batches * config.batch_size)
        left_index_val = Au_vdat.shape[0] - (val_batches * config.batch_size)

        dropout_list = np.arange(0.5, 0.75, 0.05)

        for drop1 in dropout_list:

            tf.global_variables_initializer().run()
            seed = 20

            print("dropout ratio --->", drop1)

            #### Start training the model
            lr              = config.learning_rate

            for epoch in range(config.epoch):
                seed += 1

                if np.mod(epoch + 1, 10) == 0:
                    lr = lr - lr * 0.1

                random_index    = np.random.RandomState(seed=seed).permutation(Au_trdat.shape[0])
                train_data_au   = Au_trdat[random_index]
                train_data_vi   = Vi_trdat[random_index]
                train_data_tx   = Tx_trdat[random_index]
                train_lab_au    = Au_trlab[random_index]

                for idx in range(train_batches):
                    batch_au = train_data_au[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_vi = train_data_vi[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_tx = train_data_tx[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = train_lab_au[idx * config.batch_size:(idx + 1) * config.batch_size]

                    _ = self.sess.run([cnn_optim],
                                      feed_dict={
                                          self.audio_inputs: batch_au,
                                          self.video_inputs: batch_vi,
                                          self.text_inputs: batch_tx,
                                          self.y: batch_labels,
                                          self.drop_ratio: drop1,
                                          self.learning_rate: lr
                                      })

                ##### Printing Loss on each epoch to monitor convergence
                ##### Apply Early stoping procedure to report results

                print("epoch", epoch)

                Val_Loss    = 0.0
                Tr_Loss     = 0.0

                random_index = np.random.permutation(Au_vdat.shape[0])
                VAL_data_au = Au_vdat[random_index]
                VAL_data_vi = Vi_vdat[random_index]
                VAL_data_tx = Tx_vdat[random_index]
                VAL_lab_au = Au_vlab[random_index]

                for idx in range(val_batches):
                    batch_au = VAL_data_au[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_vi = VAL_data_vi[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_tx = VAL_data_tx[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = VAL_lab_au[idx * config.batch_size:(idx + 1) * config.batch_size]

                    Val_Loss += self.Accuracy.eval({
                        self.audio_inputs: batch_au,
                        self.video_inputs: batch_vi,
                        self.text_inputs: batch_tx,
                        self.y: batch_labels
                    })

                batch_au = VAL_data_au[-left_index_val:]
                batch_vi = VAL_data_vi[-left_index_val:]
                batch_tx = VAL_data_tx[-left_index_val:]
                batch_labels = VAL_lab_au[-left_index_val:]

                Val_Loss += self.Accuracy.eval({
                    self.audio_inputs: batch_au,
                    self.video_inputs: batch_vi,
                    self.text_inputs: batch_tx,
                    self.y: batch_labels
                })

                Val_MAE = Val_Loss / (Au_vdat.shape[0])

                ### Check the training loss
                random_index = np.random.permutation(Au_trdat.shape[0])
                train_data_au = Au_trdat[random_index]
                train_data_vi = Vi_trdat[random_index]
                train_data_tx = Tx_trdat[random_index]
                train_lab_au = Au_trlab[random_index]

                for idx in range(train_batches):
                    batch_au = train_data_au[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_vi = train_data_vi[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_tx = train_data_tx[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = train_lab_au[idx * config.batch_size:(idx + 1) * config.batch_size]

                    Tr_Loss += self.Accuracy.eval({
                        self.audio_inputs: batch_au,
                        self.video_inputs: batch_vi,
                        self.text_inputs: batch_tx,
                        self.y: batch_labels
                    })

                batch_au = train_data_au[-left_index_train:]
                batch_vi = train_data_vi[-left_index_train:]
                batch_tx = train_data_tx[-left_index_train:]
                batch_labels = train_lab_au[-left_index_train:]

                Tr_Loss += self.Accuracy.eval({
                    self.audio_inputs: batch_au,
                    self.video_inputs: batch_vi,
                    self.text_inputs: batch_tx,
                    self.y: batch_labels
                })

                Train_MAE = Tr_Loss / (Au_trdat.shape[0])

                print(" ******* MOSI Results ************ ")
                print("Train MAE ---->", Train_MAE)
                print("VAl MAE ---->", Val_MAE)


        print('********** Iterations Terminated **********')


    #### Code for 3D convolution

    def TFConv_train(self, data_v, data_a, data_t, hidden_v, hidden_a, hidden_t, LSTM_hid, text_out, dropl, reuse=False):
        with tf.variable_scope("TFConv_3D") as scope:
            if reuse:
                scope.reuse_variables()

            #### Text network for text network
            LSTM_cell_t = tf.contrib.rnn.BasicLSTMCell(LSTM_hid, name='text')
            LSTM_cell_t = tf.contrib.rnn.DropoutWrapper(LSTM_cell_t, output_keep_prob=1.0 - dropl, seed=4)
            _, state_t = tf.nn.dynamic_rnn(LSTM_cell_t, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state_t.h, text_out, 'h0_t1'))
            h0t1 = tf.layers.dropout(h0t1, dropl, seed=4)
            h0t2 = tf.nn.relu(linear(h0t1, hidden_t, 'h0_t2'))

            ### Audio Network

            data_a = tf.transpose(data_a, perm=[0,2,3,1])
            print(' Audio shape_data --->',data_a.shape)

            conv0_a  = tf.nn.relu(self.c_bn_a(conv2d_revised(data_a, [5, 3, 8,  16], [1, 1, 3, 1], name='conv_0_a')))
            conv1_a = tf.nn.relu(self.c_bn1_a(conv2d_revised(conv0_a, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1_a')))
            conv1_a = tf.layers.dropout(conv1_a, dropl)
            conv2_a = tf.nn.relu(self.c_bn2_a(conv2d_revised(conv1_a, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2_a')))
            conv2_a = tf.layers.dropout(conv2_a, dropl)
            conv3_a = tf.nn.relu(self.c_bn3_a(conv2d_revised(conv2_a, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3_a')))
            print('conv3_a --->', conv3_a.shape)
            flatten_a = tf.reshape(conv3_a, [-1,conv3_a.shape[1],conv3_a.shape[-1]])
            print('flatten_a --->', flatten_a.shape)

            LSTM_cell_a = tf.contrib.rnn.BasicLSTMCell(32 ,name='audio')
            LSTM_cell_a = tf.contrib.rnn.DropoutWrapper(LSTM_cell_a, output_keep_prob=1.0 - dropl, seed=4)
            _, state_a = tf.nn.dynamic_rnn(LSTM_cell_a, flatten_a, dtype=tf.float32)

            print('state.h --->',state_a.h.shape)
            h0_a = tf.nn.relu(linear(state_a.h, hidden_a, 'h0_lin_a'))

            ######  Video Network

            data_v = tf.transpose(data_v, perm=[0,2,3,1])
            print('Video shape_data --->',data_v.shape)

            conv0_v  = tf.nn.relu(self.c_bn_v(conv2d_revised(data_v, [5, 3, 8,  16], [1, 1, 3, 1], name='conv_0_v')))
            conv1_v = tf.nn.relu(self.c_bn1_v(conv2d_revised(conv0_v, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1_v')))
            conv1_v = tf.layers.dropout(conv1_v, dropl)
            conv2_v = tf.nn.relu(self.c_bn2_v(conv2d_revised(conv1_v, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2_v')))
            conv2_v = tf.layers.dropout(conv2_v, dropl)
            conv3_v = tf.nn.relu(self.c_bn3_v(conv2d_revised(conv2_v, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3_v')))

            print('conv3_v --->', conv3_v.shape)
            flatten_v = tf.reshape(conv3_v, [-1,conv3_v.shape[1],conv3_v.shape[-1]])
            print('flatten_v --->', flatten_v.shape)

            LSTM_cell_v = tf.contrib.rnn.BasicLSTMCell(32 ,name='video')
            LSTM_cell_v = tf.contrib.rnn.DropoutWrapper(LSTM_cell_v, output_keep_prob=1.0 - dropl, seed=4)
            _, state_v = tf.nn.dynamic_rnn(LSTM_cell_v, flatten_v, dtype=tf.float32)

            print('state_v.h --->',state_v.h.shape)
            h0_v = tf.nn.relu(linear(state_v.h, hidden_v, 'h0_lin_v'))

            #### Combine them into a tensor

            TF_ta = tf.einsum('ij,ik->ijk', h0t2, h0_a)
            TF_avt = tf.einsum('ijk,il->ijkl', TF_ta, h0_v)
            TF_avt = tf.expand_dims(TF_avt, [-1])
            print('TF_avt---->',TF_avt)


            #### Combined Tensor
            conv0 = tf.nn.relu(self.conv_bn(
                conv3d_revised(TF_avt, [5,5,5, TF_avt.shape[-1], 8], [1, 2, 2, 2, 1], name='conv_0')))
            conv0 = tf.layers.dropout(conv0, dropl, seed=4)
            conv1 = tf.nn.relu(
                self.conv_bn1(conv3d_revised(conv0, [5,5,5, conv0.shape[-1], 32], [1, 2, 2, 2, 1], name='conv_1')))
            conv1 = tf.layers.dropout(conv1, dropl, seed=4)
            conv2 = tf.nn.relu(self.conv_bn2(
                conv3d_revised(conv1, [5,5,5, conv1.shape[-1], 64], [1, 3, 3, 3, 1], padding='SAME', name='conv_2')))
            conv2 = tf.reshape(conv2, [-1, conv2.shape[-1]])
            flatten = tf.nn.relu(linear(conv2, 10, 'flatten'))


            pred = linear(flatten, 1, 'pred')

            return pred, flatten


    def TFConv_test(self, data_v, data_a, data_t, hidden_v, hidden_a, hidden_t, LSTM_hid, text_out):
        with tf.variable_scope("TFConv_3D") as scope:
            scope.reuse_variables()

            #### Text network for text network
            LSTM_cell_t = tf.contrib.rnn.BasicLSTMCell(LSTM_hid, name='text')
            LSTM_cell_t = tf.contrib.rnn.DropoutWrapper(LSTM_cell_t, output_keep_prob=1.0, seed=4)
            _, state_t = tf.nn.dynamic_rnn(LSTM_cell_t, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state_t.h, text_out, 'h0_t1'))
            h0t1 = tf.layers.dropout(h0t1, 0.0, seed=4)
            h0t2 = tf.nn.relu(linear(h0t1, hidden_t, 'h0_t2'))

            ### Audio Network
            data_a = tf.transpose(data_a, perm=[0, 2, 3, 1])

            conv0_a = tf.nn.relu(self.c_bn_a(conv2d_revised(data_a, [5, 3, 8, 16], [1, 1, 3, 1], name='conv_0_a'), train=False))
            conv1_a = tf.nn.relu(
                self.c_bn1_a(conv2d_revised(conv0_a, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1_a'), train=False))
            conv1_a = tf.layers.dropout(conv1_a, 0.0)
            conv2_a = tf.nn.relu(
                self.c_bn2_a(conv2d_revised(conv1_a, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2_a'), train=False))
            conv2_a = tf.layers.dropout(conv2_a, 0.0)
            conv3_a = tf.nn.relu(
                self.c_bn3_a(conv2d_revised(conv2_a, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3_a'), train=False))
            flatten_a = tf.reshape(conv3_a, [-1, conv3_a.shape[1], conv3_a.shape[-1]])

            LSTM_cell_a = tf.contrib.rnn.BasicLSTMCell(32, name='audio')
            LSTM_cell_a = tf.contrib.rnn.DropoutWrapper(LSTM_cell_a, output_keep_prob=1.0 - 0.0, seed=4)
            _, state_a = tf.nn.dynamic_rnn(LSTM_cell_a, flatten_a, dtype=tf.float32)

            h0_a = tf.nn.relu(linear(state_a.h, hidden_a, 'h0_lin_a'))

            ######  Video Network
            data_v = tf.transpose(data_v, perm=[0, 2, 3, 1])

            conv0_v = tf.nn.relu(self.c_bn_v(conv2d_revised(data_v, [5, 3, 8, 16], [1, 1, 3, 1], name='conv_0_v'), train=False))
            conv1_v = tf.nn.relu(self.c_bn1_v(conv2d_revised(conv0_v, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1_v'), train=False))
            conv1_v = tf.layers.dropout(conv1_v, 0.0)
            conv2_v = tf.nn.relu(self.c_bn2_v(conv2d_revised(conv1_v, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2_v'), train=False))
            conv2_v = tf.layers.dropout(conv2_v, 0.0)
            conv3_v = tf.nn.relu(
                self.c_bn3_v(conv2d_revised(conv2_v, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3_v'), train=False))

            flatten_v = tf.reshape(conv3_v, [-1, conv3_v.shape[1], conv3_v.shape[-1]])

            LSTM_cell_v = tf.contrib.rnn.BasicLSTMCell(32, name='video')
            LSTM_cell_v = tf.contrib.rnn.DropoutWrapper(LSTM_cell_v, output_keep_prob=1.0 - 0.0, seed=4)
            _, state_v = tf.nn.dynamic_rnn(LSTM_cell_v, flatten_v, dtype=tf.float32)

            h0_v = tf.nn.relu(linear(state_v.h, hidden_v, 'h0_lin_v'))

            #### Combine them into a tensor

            TF_ta = tf.einsum('ij,ik->ijk', h0t2, h0_a)
            TF_avt = tf.einsum('ijk,il->ijkl', TF_ta, h0_v)
            TF_avt = tf.expand_dims(TF_avt, [-1])

            #### Video, Audio and Text Subnets
            conv0 = tf.nn.relu(self.conv_bn(
                conv3d_revised(TF_avt, [5,5,5, TF_avt.shape[-1], 8], [1, 2, 2, 2, 1], name='conv_0'), train=False))
            conv0 = tf.layers.dropout(conv0, 0.0, seed=4)
            conv1 = tf.nn.relu(
                self.conv_bn1(conv3d_revised(conv0, [5,5,5, conv0.shape[-1], 32], [1, 2, 2, 2, 1], name='conv_1'), train=False))
            conv1 = tf.layers.dropout(conv1, 0.0, seed=4)
            conv2 = tf.nn.relu(self.conv_bn2(
                conv3d_revised(conv1, [5,5,5, conv1.shape[-1], 64], [1, 3,3,3, 1], padding='SAME',
                               name='conv_2'), train=False))
            conv2 = tf.reshape(conv2, [-1, conv2.shape[-1]])
            flatten = tf.nn.relu(linear(conv2, 10, 'flatten'))

            pred = linear(flatten, 1, 'pred')

            return pred, flatten

    # Code for Audio and Text
    def TFConv_train_AT(self, data_a, data_t, hidden_a, hidden_t, LSTM_hid, text_out, dropl, reuse=False):
        with tf.variable_scope("TFConv_AT") as scope:
            if reuse:
                scope.reuse_variables()

            #### Text network for text network
            LSTM_cell_t = tf.contrib.rnn.BasicLSTMCell(LSTM_hid, name='text_at')
            LSTM_cell_t = tf.contrib.rnn.DropoutWrapper(LSTM_cell_t, output_keep_prob=1.0 - dropl, seed=4)
            _, state_t = tf.nn.dynamic_rnn(LSTM_cell_t, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state_t.h, text_out, 'h0_t1_at'))
            h0t1 = tf.layers.dropout(h0t1, dropl, seed=4)
            h0t2 = tf.nn.relu(linear(h0t1, hidden_t, 'h0_t2_at'))

            ### Audio Network

            data_a = tf.transpose(data_a, perm=[0,2,3,1])
            print(' Audio shape_data --->',data_a.shape)

            conv0_a  = tf.nn.relu(self.c_bn_at_a(conv2d_revised(data_a, [5, 3, 8,  16], [1, 1, 3, 1], name='conv_0_a_at')))
            conv1_a = tf.nn.relu(self.c_bn1_at_a(conv2d_revised(conv0_a, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1_a_at')))
            conv1_a = tf.layers.dropout(conv1_a, dropl)
            conv2_a = tf.nn.relu(self.c_bn2_at_a(conv2d_revised(conv1_a, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2_a_at')))
            conv2_a = tf.layers.dropout(conv2_a, dropl)
            conv3_a = tf.nn.relu(self.c_bn3_at_a(conv2d_revised(conv2_a, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3_a_at')))
            print('conv3_a --->', conv3_a.shape)
            flatten_a = tf.reshape(conv3_a, [-1,conv3_a.shape[1],conv3_a.shape[-1]])
            print('flatten_a --->', flatten_a.shape)

            LSTM_cell_a = tf.contrib.rnn.BasicLSTMCell(32 ,name='audio_at')
            LSTM_cell_a = tf.contrib.rnn.DropoutWrapper(LSTM_cell_a, output_keep_prob=1.0 - dropl, seed=4)
            _, state_a = tf.nn.dynamic_rnn(LSTM_cell_a, flatten_a, dtype=tf.float32)

            print('state.h --->',state_a.h.shape)
            h0_a = tf.nn.relu(linear(state_a.h, hidden_a, 'h0_lin_a_at'))


            #### Combine them into a tensor
            TF_ta = tf.einsum('ij,ik->ijk', h0t2, h0_a)
            TF_at = tf.expand_dims(TF_ta, [-1])
            print('TF_at---->',TF_at)


            #### Combined Tensor
            conv0 = tf.nn.relu(self.conv_bn_at(conv2d_revised(TF_at, [5,5,TF_at.shape[-1], 8], [1,2,2,1], name='conv_0_at')))
            conv0 = tf.layers.dropout(conv0, dropl, seed=4)
            conv1 = tf.nn.relu(self.conv_bn1_at(conv2d_revised(conv0, [5,5,conv0.shape[-1], 32], [1,2,2,1], name='conv_1_at')))
            conv1 = tf.layers.dropout(conv1, dropl, seed=4)
            conv2 = tf.nn.relu(self.conv_bn2_at(conv2d_revised(conv1, [5,5,conv1.shape[-1], 64], [1,3,3,1], name='conv_2_at')))
            conv2 = tf.reshape(conv2, [-1, conv2.shape[-1]])
            flatten = tf.nn.relu(linear(conv2, 10, 'flatten_at'))

            pred = linear(flatten, 1, 'pred_at')

            return pred, flatten

    def TFConv_test_AT(self, data_a, data_t, hidden_a, hidden_t, LSTM_hid, text_out):
        with tf.variable_scope("TFConv_AT") as scope:
            scope.reuse_variables()


            #### Text network for text network
            LSTM_cell_t = tf.contrib.rnn.BasicLSTMCell(LSTM_hid, name='text_at')
            LSTM_cell_t = tf.contrib.rnn.DropoutWrapper(LSTM_cell_t, output_keep_prob=1.0 - 0.0, seed=4)
            _, state_t = tf.nn.dynamic_rnn(LSTM_cell_t, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state_t.h, text_out, 'h0_t1_at'))
            h0t1 = tf.layers.dropout(h0t1, 0.0, seed=4)
            h0t2 = tf.nn.relu(linear(h0t1, hidden_t, 'h0_t2_at'))

            ### Audio Network

            data_a = tf.transpose(data_a, perm=[0,2,3,1])
            print(' Audio shape_data --->',data_a.shape)

            conv0_a  = tf.nn.relu(self.c_bn_at_a(conv2d_revised(data_a, [5, 3, 8,  16], [1, 1, 3, 1], name='conv_0_a_at'), train=False))
            conv1_a = tf.nn.relu(self.c_bn1_at_a(conv2d_revised(conv0_a, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1_a_at'), train=False))
            conv1_a = tf.layers.dropout(conv1_a, 0.0)
            conv2_a = tf.nn.relu(self.c_bn2_at_a(conv2d_revised(conv1_a, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2_a_at'), train=False))
            conv2_a = tf.layers.dropout(conv2_a, 0.0)
            conv3_a = tf.nn.relu(self.c_bn3_at_a(conv2d_revised(conv2_a, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3_a_at'), train=False))
            flatten_a = tf.reshape(conv3_a, [-1,conv3_a.shape[1],conv3_a.shape[-1]])

            LSTM_cell_a = tf.contrib.rnn.BasicLSTMCell(32 ,name='audio_at')
            LSTM_cell_a = tf.contrib.rnn.DropoutWrapper(LSTM_cell_a, output_keep_prob=1.0 - 0.0, seed=4)
            _, state_a = tf.nn.dynamic_rnn(LSTM_cell_a, flatten_a, dtype=tf.float32)

            h0_a = tf.nn.relu(linear(state_a.h, hidden_a, 'h0_lin_a_at'))

            #### Combine them into a tensor
            TF_ta = tf.einsum('ij,ik->ijk', h0t2, h0_a)
            TF_at = tf.expand_dims(TF_ta, [-1])

            #### Combined Tensor
            conv0 = tf.nn.relu(self.conv_bn_at(conv2d_revised(TF_at, [5,5,TF_at.shape[-1], 8], [1,2,2,1], name='conv_0_at'), train=False))
            conv0 = tf.layers.dropout(conv0, 0.0, seed=4)
            conv1 = tf.nn.relu(self.conv_bn1_at(conv2d_revised(conv0, [5,5,conv0.shape[-1], 32], [1,2,2,1], name='conv_1_at'), train=False))
            conv1 = tf.layers.dropout(conv1, 0.0, seed=4)
            conv2 = tf.nn.relu(self.conv_bn2_at(conv2d_revised(conv1, [5,5,conv1.shape[-1], 64], [1,3,3,1], name='conv_2_at'), train=False))
            conv2 = tf.reshape(conv2, [-1, conv2.shape[-1]])
            flatten = tf.nn.relu(linear(conv2, 10, 'flatten_at'))

            pred = linear(flatten, 1, 'pred_at')

            return pred, flatten

    # Code for Video and Text
    def TFConv_train_VT(self, data_v, data_t, hidden_v, hidden_t, LSTM_hid, text_out, dropl, reuse=False):
        with tf.variable_scope("TFConv_VT") as scope:
            if reuse:
                scope.reuse_variables()

            #### Text network for text network
            LSTM_cell_t = tf.contrib.rnn.BasicLSTMCell(LSTM_hid, name='text_vt')
            LSTM_cell_t = tf.contrib.rnn.DropoutWrapper(LSTM_cell_t, output_keep_prob=1.0 - dropl, seed=4)
            _, state_t = tf.nn.dynamic_rnn(LSTM_cell_t, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state_t.h, text_out, 'h0_t1_vt'))
            h0t1 = tf.layers.dropout(h0t1, dropl, seed=4)
            h0t2 = tf.nn.relu(linear(h0t1, hidden_t, 'h0_t2_vt'))

            ### Audio Network

            data_v = tf.transpose(data_v, perm=[0,2,3,1])
            print(' Audio shape_data --->',data_v.shape)

            conv0_v  = tf.nn.relu(self.c_bn_vt_v(conv2d_revised(data_v, [5, 3, 8,  16], [1, 1, 3, 1], name='conv_0_v_vt')))
            conv1_v = tf.nn.relu(self.c_bn1_vt_v(conv2d_revised(conv0_v, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1_v_vt')))
            conv1_v = tf.layers.dropout(conv1_v, dropl)
            conv2_v = tf.nn.relu(self.c_bn2_vt_v(conv2d_revised(conv1_v, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2_v_vt')))
            conv2_v = tf.layers.dropout(conv2_v, dropl)
            conv3_v = tf.nn.relu(self.c_bn3_vt_v(conv2d_revised(conv2_v, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3_v_vt')))
            print('conv3_a --->', conv3_v.shape)
            flatten_v = tf.reshape(conv3_v, [-1,conv3_v.shape[1],conv3_v.shape[-1]])
            print('flatten_v --->', flatten_v.shape)

            LSTM_cell_v = tf.contrib.rnn.BasicLSTMCell(32 ,name='visual_vt')
            LSTM_cell_v = tf.contrib.rnn.DropoutWrapper(LSTM_cell_v, output_keep_prob=1.0 - dropl, seed=4)
            _, state_v = tf.nn.dynamic_rnn(LSTM_cell_v, flatten_v, dtype=tf.float32)

            print('state.h --->',state_v.h.shape)
            h0_v = tf.nn.relu(linear(state_v.h, hidden_v, 'h0_lin_v_vt'))

            #### Combine them into a tensor
            TF_tv = tf.einsum('ij,ik->ijk', h0t2, h0_v)
            TF_vt = tf.expand_dims(TF_tv, [-1])
            print('TF_vt---->',TF_vt)


            #### Combined Tensor
            conv0 = tf.nn.relu(self.conv_bn_vt(conv2d_revised(TF_vt, [3,3,TF_vt.shape[-1], 8], [1,2,2,1], name='conv_0_vt')))
            conv0 = tf.layers.dropout(conv0, dropl, seed=4)
            conv1 = tf.nn.relu(self.conv_bn1_vt(conv2d_revised(conv0, [3,3,conv0.shape[-1], 32], [1,2,2,1], name='conv_1_vt')))
            conv1 = tf.layers.dropout(conv1, dropl, seed=4)
            conv2 = tf.nn.relu(self.conv_bn2_vt(conv2d_revised(conv1, [3,3,conv1.shape[-1], 64], [1,3,3,1], name='conv_2_vt')))
            conv2 = tf.reshape(conv2, [-1, conv2.shape[-1]])
            flatten = tf.nn.relu(linear(conv2, 10, 'flatten_vt'))

            pred = linear(flatten, 1, 'pred_vt')

            return pred, flatten

    def TFConv_test_VT(self, data_v, data_t, hidden_v, hidden_t, LSTM_hid, text_out):
        with tf.variable_scope("TFConv_VT") as scope:
            scope.reuse_variables()

            #### Text network for text network
            LSTM_cell_t = tf.contrib.rnn.BasicLSTMCell(LSTM_hid, name='text_vt')
            LSTM_cell_t = tf.contrib.rnn.DropoutWrapper(LSTM_cell_t, output_keep_prob=1.0 - 0.0, seed=4)
            _, state_t = tf.nn.dynamic_rnn(LSTM_cell_t, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state_t.h, text_out, 'h0_t1_vt'))
            h0t1 = tf.layers.dropout(h0t1, 0.0, seed=4)
            h0t2 = tf.nn.relu(linear(h0t1, hidden_t, 'h0_t2_vt'))

            ### Video Network
            data_v = tf.transpose(data_v, perm=[0, 2, 3, 1])

            conv0_v = tf.nn.relu(self.c_bn_vt_v(conv2d_revised(data_v, [5, 3, 8, 16], [1, 1, 3, 1], name='conv_0_v_vt'), train=False))
            conv1_v = tf.nn.relu(self.c_bn1_vt_v(conv2d_revised(conv0_v, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1_v_vt'), train=False))
            conv1_v = tf.layers.dropout(conv1_v, 0.0)
            conv2_v = tf.nn.relu(self.c_bn2_vt_v(conv2d_revised(conv1_v, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2_v_vt'), train=False))
            conv2_v = tf.layers.dropout(conv2_v, 0.0)
            conv3_v = tf.nn.relu(self.c_bn3_vt_v(conv2d_revised(conv2_v, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3_v_vt'), train=False))
            flatten_v = tf.reshape(conv3_v, [-1, conv3_v.shape[1], conv3_v.shape[-1]])

            LSTM_cell_v = tf.contrib.rnn.BasicLSTMCell(32, name='visual_vt')
            LSTM_cell_v = tf.contrib.rnn.DropoutWrapper(LSTM_cell_v, output_keep_prob=1.0 - 0.0, seed=4)
            _, state_v = tf.nn.dynamic_rnn(LSTM_cell_v, flatten_v, dtype=tf.float32)

            h0_v = tf.nn.relu(linear(state_v.h, hidden_v, 'h0_lin_v_vt'))

            #### Combine them into a tensor
            TF_tv = tf.einsum('ij,ik->ijk', h0t2, h0_v)
            TF_vt = tf.expand_dims(TF_tv, [-1])

            #### Combined Tensor
            conv0 = tf.nn.relu(
                self.conv_bn_vt(conv2d_revised(TF_vt, [3,3, TF_vt.shape[-1], 8], [1, 2, 2, 1], name='conv_0_vt'), train=False))
            conv0 = tf.layers.dropout(conv0, 0.0, seed=4)
            conv1 = tf.nn.relu(
                self.conv_bn1_vt(conv2d_revised(conv0, [3,3, conv0.shape[-1], 32], [1, 2, 2, 1], name='conv_1_vt'), train=False))
            conv1 = tf.layers.dropout(conv1, 0.0, seed=4)
            conv2 = tf.nn.relu(
                self.conv_bn2_vt(conv2d_revised(conv1, [3,3, conv1.shape[-1], 64], [1, 3, 3, 1], name='conv_2_vt'), train=False))
            conv2 = tf.reshape(conv2, [-1, conv2.shape[-1]])
            flatten = tf.nn.relu(linear(conv2, 10, 'flatten_vt'))

            pred = linear(flatten, 1, 'pred_vt')

            return pred, flatten


    # Code for Video and Audio
    def TFConv_train_AV(self, data_v, data_a, hidden_v, hidden_a, dropl, reuse=False):
        with tf.variable_scope("TFConv_AV") as scope:
            if reuse:
                scope.reuse_variables()

            ### Audio Network
            data_a = tf.transpose(data_a, perm=[0,2,3,1])
            print(' Audio shape_data --->',data_a.shape)

            conv0_a  = tf.nn.relu(self.c_bn_av_a(conv2d_revised(data_a, [5, 3, 8,  16], [1, 1, 3, 1], name='conv_0_a_av')))
            conv1_a = tf.nn.relu(self.c_bn1_av_a(conv2d_revised(conv0_a, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1_a_av')))
            conv1_a = tf.layers.dropout(conv1_a, dropl)
            conv2_a = tf.nn.relu(self.c_bn2_av_a(conv2d_revised(conv1_a, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2_a_av')))
            conv2_a = tf.layers.dropout(conv2_a, dropl)
            conv3_a = tf.nn.relu(self.c_bn3_av_a(conv2d_revised(conv2_a, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3_a_av')))
            print('conv3_a --->', conv3_a.shape)
            flatten_a = tf.reshape(conv3_a, [-1,conv3_a.shape[1],conv3_a.shape[-1]])
            print('flatten_a --->', flatten_a.shape)

            LSTM_cell_a = tf.contrib.rnn.BasicLSTMCell(32 ,name='audio_av')
            LSTM_cell_a = tf.contrib.rnn.DropoutWrapper(LSTM_cell_a, output_keep_prob=1.0 - dropl, seed=4)
            _, state_a = tf.nn.dynamic_rnn(LSTM_cell_a, flatten_a, dtype=tf.float32)

            print('state.h --->',state_a.h.shape)
            h0_a = tf.nn.relu(linear(state_a.h, hidden_a, 'h0_lin_a_av'))


            ### Audio Network

            data_v = tf.transpose(data_v, perm=[0,2,3,1])
            print(' Audio shape_data --->',data_v.shape)

            conv0_v  = tf.nn.relu(self.c_bn_av_v(conv2d_revised(data_v, [5, 3, 8,  16], [1, 1, 3, 1], name='conv_0_v_av')))
            conv1_v = tf.nn.relu(self.c_bn1_av_v(conv2d_revised(conv0_v, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1_v_av')))
            conv1_v = tf.layers.dropout(conv1_v, dropl)
            conv2_v = tf.nn.relu(self.c_bn2_av_v(conv2d_revised(conv1_v, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2_v_av')))
            conv2_v = tf.layers.dropout(conv2_v, dropl)
            conv3_v = tf.nn.relu(self.c_bn3_av_v(conv2d_revised(conv2_v, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3_v_av')))
            print('conv3_a --->', conv3_v.shape)
            flatten_v = tf.reshape(conv3_v, [-1,conv3_v.shape[1],conv3_v.shape[-1]])
            print('flatten_v --->', flatten_v.shape)

            LSTM_cell_v = tf.contrib.rnn.BasicLSTMCell(32 ,name='visual_av')
            LSTM_cell_v = tf.contrib.rnn.DropoutWrapper(LSTM_cell_v, output_keep_prob=1.0 - dropl, seed=4)
            _, state_v = tf.nn.dynamic_rnn(LSTM_cell_v, flatten_v, dtype=tf.float32)

            print('state.h --->',state_v.h.shape)
            h0_v = tf.nn.relu(linear(state_v.h, hidden_v, 'h0_lin_v_av'))

            #### Combine them into a tensor
            TF_av = tf.einsum('ij,ik->ijk', h0_a, h0_v)
            TF_av = tf.expand_dims(TF_av, [-1])

            #### Combined Tensor
            conv0 = tf.nn.relu(self.conv_bn_av(conv2d_revised(TF_av, [3,3,TF_av.shape[-1], 8], [1,2,2,1], name='conv_0_av')))
            conv0 = tf.layers.dropout(conv0, dropl, seed=4)
            conv1 = tf.nn.relu(self.conv_bn1_av(conv2d_revised(conv0, [3,3,conv0.shape[-1], 32], [1,2,2,1], name='conv_1_av')))
            conv1 = tf.layers.dropout(conv1, dropl, seed=4)
            conv2 = tf.nn.relu(self.conv_bn2_av(conv2d_revised(conv1, [3,3,conv1.shape[-1], 64], [1,3,3,1], name='conv_2_av')))
            conv2 = tf.reshape(conv2, [-1, conv2.shape[-1]])
            flatten = tf.nn.relu(linear(conv2, 10, 'flatten_av'))

            pred = linear(flatten, 1, 'pred_av')

            return pred, flatten

    def TFConv_test_AV(self, data_v, data_a, hidden_v, hidden_a):
        with tf.variable_scope("TFConv_AV") as scope:
            scope.reuse_variables()

            ### Audio Network

            data_a = tf.transpose(data_a, perm=[0,2,3,1])
            conv0_a  = tf.nn.relu(self.c_bn_av_a(conv2d_revised(data_a, [5, 3, 8,  16], [1, 1, 3, 1], name='conv_0_a_av')))
            conv1_a = tf.nn.relu(self.c_bn1_av_a(conv2d_revised(conv0_a, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1_a_av')))
            conv2_a = tf.nn.relu(self.c_bn2_av_a(conv2d_revised(conv1_a, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2_a_av')))
            conv3_a = tf.nn.relu(self.c_bn3_av_a(conv2d_revised(conv2_a, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3_a_av')))
            flatten_a = tf.reshape(conv3_a, [-1,conv3_a.shape[1],conv3_a.shape[-1]])

            LSTM_cell_a = tf.contrib.rnn.BasicLSTMCell(32 ,name='audio_av')
            LSTM_cell_a = tf.contrib.rnn.DropoutWrapper(LSTM_cell_a, output_keep_prob=1.0, seed=4)
            _, state_a = tf.nn.dynamic_rnn(LSTM_cell_a, flatten_a, dtype=tf.float32)

            h0_a = tf.nn.relu(linear(state_a.h, hidden_a, 'h0_lin_a_av'))

            ### Video Network
            data_v = tf.transpose(data_v, perm=[0, 2, 3, 1])

            conv0_v = tf.nn.relu(self.c_bn_av_v(conv2d_revised(data_v, [5, 3, 8, 16], [1, 1, 3, 1], name='conv_0_v_av'), train=False))
            conv1_v = tf.nn.relu(self.c_bn1_av_v(conv2d_revised(conv0_v, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1_v_av'), train=False))
            conv1_v = tf.layers.dropout(conv1_v, 0.0)
            conv2_v = tf.nn.relu(self.c_bn2_av_v(conv2d_revised(conv1_v, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2_v_av'), train=False))
            conv2_v = tf.layers.dropout(conv2_v, 0.0)
            conv3_v = tf.nn.relu(self.c_bn3_av_v(conv2d_revised(conv2_v, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3_v_av'), train=False))
            flatten_v = tf.reshape(conv3_v, [-1, conv3_v.shape[1], conv3_v.shape[-1]])

            LSTM_cell_v = tf.contrib.rnn.BasicLSTMCell(32, name='visual_av')
            LSTM_cell_v = tf.contrib.rnn.DropoutWrapper(LSTM_cell_v, output_keep_prob=1.0 - 0.0, seed=4)
            _, state_v = tf.nn.dynamic_rnn(LSTM_cell_v, flatten_v, dtype=tf.float32)

            h0_v = tf.nn.relu(linear(state_v.h, hidden_v, 'h0_lin_v_av'))

            #### Combine them into a tensor
            TF_av = tf.einsum('ij,ik->ijk', h0_a, h0_v)
            TF_av = tf.expand_dims(TF_av, [-1])

            #### Combined Tensor
            conv0 = tf.nn.relu(
                self.conv_bn_av(conv2d_revised(TF_av, [3,3, TF_av.shape[-1], 8], [1, 2, 2, 1], name='conv_0_av'), train=False))
            conv0 = tf.layers.dropout(conv0, 0.0, seed=4)
            conv1 = tf.nn.relu(
                self.conv_bn1_av(conv2d_revised(conv0, [3,3, conv0.shape[-1], 32], [1, 2, 2, 1], name='conv_1_av'), train=False))
            conv1 = tf.layers.dropout(conv1, 0.0, seed=4)
            conv2 = tf.nn.relu(
                self.conv_bn2_av(conv2d_revised(conv1, [3,3, conv1.shape[-1], 64], [1, 3, 3, 1], name='conv_2_av'), train=False))
            conv2 = tf.reshape(conv2, [-1, conv2.shape[-1]])
            flatten = tf.nn.relu(linear(conv2, 10, 'flatten_av'))

            pred = linear(flatten, 1, 'pred_av')

            return pred, flatten

    # Code for Audio

    def FMA_train(self, data_a, hidden_a, dropl, reuse=False):
        with tf.variable_scope("FMA") as scope:

            if reuse:
                scope.reuse_variables()

            data = tf.transpose(data_a, perm=[0,2,3,1])
            print('shape_data --->',data.shape)

            conv0  = tf.nn.relu(self.c_bn(conv2d_revised(data, [5, 3, 8,  16], [1, 1, 3, 1], name='conv_0')))
            conv1 = tf.nn.relu(self.c_bn1(conv2d_revised(conv0, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1')))
            conv1 = tf.layers.dropout(conv1, dropl)
            conv2 = tf.nn.relu(self.c_bn2(conv2d_revised(conv1, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2')))
            conv2 = tf.layers.dropout(conv2, dropl)
            conv3 = tf.nn.relu(self.c_bn3(conv2d_revised(conv2, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3')))

            print('conv3 --->', conv3.shape)
            flatten = tf.reshape(conv3, [-1,conv3.shape[1],conv3.shape[-1]])
            print('flatten --->', flatten.shape)

            LSTM_cell = tf.contrib.rnn.BasicLSTMCell(32)
            LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, output_keep_prob=1.0 - dropl, seed=4)
            _, state = tf.nn.dynamic_rnn(LSTM_cell, flatten, dtype=tf.float32)


            print('state.h --->',state.h.shape)
            h0 = tf.nn.relu(linear(state.h, hidden_a, 'h0_lin'))
            pred = linear(h0, 1, 'h1_lin')

            print('pred--->', pred.shape)

            return pred, h0

    def FMA_test(self, data_a, hidden_a):
        with tf.variable_scope("FMA") as scope:
            scope.reuse_variables()

            data = tf.transpose(data_a, perm=[0, 2, 3, 1])


            conv0  = tf.nn.relu(self.c_bn(conv2d_revised(data, [5, 3, 8,  16], [1, 1,3, 1], name='conv_0'), train=False))
            conv1 = tf.nn.relu(self.c_bn1(conv2d_revised(conv0, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1'), train=False))
            conv1 = tf.layers.dropout(conv1, 0.0)
            conv2 = tf.nn.relu(self.c_bn2(conv2d_revised(conv1, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2'), train=False))
            conv2 = tf.layers.dropout(conv2, 0.0)
            conv3 = tf.nn.relu(self.c_bn3(conv2d_revised(conv2, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3'), train=False))

            flatten = tf.reshape(conv3, [-1,conv3.shape[1],conv3.shape[-1]])

            LSTM_cell = tf.contrib.rnn.BasicLSTMCell(32)
            LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, output_keep_prob=1.0, seed=4)
            _, state = tf.nn.dynamic_rnn(LSTM_cell, flatten, dtype=tf.float32)

            h0 = tf.nn.relu(linear(state.h, hidden_a, 'h0_lin'))
            pred = linear(h0, 1, 'h1_lin')

            return pred, h0

    # Code for Video

    def FMV_train(self, data_v, hidden_v, dropl, reuse=False):
        with tf.variable_scope("FMV") as scope:

            if reuse:
                scope.reuse_variables()

            data = tf.transpose(data_v, perm=[0,2,3,1])
            print('shape_data --->',data.shape)

            conv0  = tf.nn.relu(self.v_bn(conv2d_revised(data, [5, 3, 8,  16], [1, 1, 3, 1], name='conv_0')))
            conv1 = tf.nn.relu(self.v1_bn(conv2d_revised(conv0, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1')))
            conv1 = tf.layers.dropout(conv1, dropl)
            conv2 = tf.nn.relu(self.v2_bn(conv2d_revised(conv1, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2')))
            conv2 = tf.layers.dropout(conv2, dropl)
            conv3 = tf.nn.relu(self.v3_bn(conv2d_revised(conv2, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3')))

            print('conv3 --->', conv3.shape)
            flatten = tf.reshape(conv3, [-1,conv3.shape[1],conv3.shape[-1]])
            print('flatten --->', flatten.shape)

            LSTM_cell = tf.contrib.rnn.BasicLSTMCell(32)
            LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, output_keep_prob=1.0 - dropl, seed=4)
            _, state = tf.nn.dynamic_rnn(LSTM_cell, flatten, dtype=tf.float32)

            print('state.h --->',state.h.shape)
            h0 = tf.nn.relu(linear(state.h, hidden_v, 'h0_lin'))
            pred = linear(h0, 1, 'h1_lin')
            print('pred--->', pred.shape)

            return pred, h0

    def FMV_test(self, data_v, hidden_v):
        with tf.variable_scope("FMV") as scope:
            scope.reuse_variables()

            data = tf.transpose(data_v, perm=[0, 2, 3, 1])


            conv0  = tf.nn.relu(self.v_bn(conv2d_revised(data, [5, 3, 8,  16], [1, 1,3, 1], name='conv_0'), train=False))
            conv1 = tf.nn.relu(self.v1_bn(conv2d_revised(conv0, [5, 3, 16, 32], [1, 1, 3, 1], name='conv_1'), train=False))
            conv1 = tf.layers.dropout(conv1, 0.0)
            conv2 = tf.nn.relu(self.v2_bn(conv2d_revised(conv1, [5, 3, 32, 64], [1, 1, 3, 1], name='conv_2'), train=False))
            conv2 = tf.layers.dropout(conv2, 0.0)
            conv3 = tf.nn.relu(self.v3_bn(conv2d_revised(conv2, [5, 3, 64, 128], [1, 1, 3, 1], name='conv_3'), train=False))

            flatten = tf.reshape(conv3, [-1,conv3.shape[1],conv3.shape[-1]])

            LSTM_cell = tf.contrib.rnn.BasicLSTMCell(32)
            LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, output_keep_prob=1.0, seed=4)
            _, state = tf.nn.dynamic_rnn(LSTM_cell, flatten, dtype=tf.float32)


            h0 = tf.nn.relu(linear(state.h, hidden_v, 'h0_lin'))
            pred = linear(h0, 1, 'h1_lin')

            return pred, h0

    #### Code for Text
    def FMT_train(self, data_t, hidden_t, LSTM_hid, text_out, dropl, reuse=False):
        with tf.variable_scope("FMT") as scope:
            if reuse:
                scope.reuse_variables()

            #### LSTM for text network
            LSTM_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_hid)
            LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, output_keep_prob=1.0 - dropl, seed=4)
            _, state = tf.nn.dynamic_rnn(LSTM_cell, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state.h, text_out, 'h0_t1'))
            h0t1 = tf.layers.dropout(h0t1, dropl, seed=4)
            h0t2 = tf.nn.relu(linear(h0t1, hidden_t, 'h0_t2'))
            pred = linear(h0t2, 1, 'pred')

            return pred, h0t2

    def FMT_test(self, data_t, hidden_t, LSTM_hid, text_out, ):
        with tf.variable_scope("FMT") as scope:
            scope.reuse_variables()

            #### LSTM for text network
            LSTM_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_hid)
            LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, output_keep_prob=1.0 - 0, seed=4)
            _, state = tf.nn.dynamic_rnn(LSTM_cell, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state.h, text_out, 'h0_t1'))
            h0t1 = tf.layers.dropout(h0t1, 0.0, seed=4)
            h0t2 = tf.nn.relu(linear(h0t1, hidden_t, 'h0_t2'))
            pred = linear(h0t2, 1, 'pred')

            return pred, h0t2


    def fusion_train(self, a,v,t, at, vt, av, avt, reuse=False):
        with tf.variable_scope("fusion") as scope:
            if reuse:
                scope.reuse_variables()

            a = tf.expand_dims(a,1)
            v = tf.expand_dims(v,1)
            t = tf.expand_dims(t,1)
            at = tf.expand_dims(at,1)
            vt = tf.expand_dims(vt,1)
            av = tf.expand_dims(av,1)
            avt = tf.expand_dims(avt,1)

            input = tf.concat([a,v,t, at, vt, av, avt],axis=1)

            fused = tf.transpose(input,[0,2,1])
            fused = attention(fused,10,'attn')
            fused = tf.reshape(fused, [-1,fused.shape[1]])

            print('fusion shape--->', fused.shape)

            pred = linear(fused,1,'pred')

            return pred

    def fusion_test(self, a, v, t, at, vt, av, avt):
        with tf.variable_scope("fusion") as scope:
            scope.reuse_variables()

            a = tf.expand_dims(a, 1)
            v = tf.expand_dims(v, 1)
            t = tf.expand_dims(t, 1)
            at = tf.expand_dims(at, 1)
            vt = tf.expand_dims(vt, 1)
            av = tf.expand_dims(av, 1)
            avt = tf.expand_dims(avt, 1)

            input = tf.concat([a,v,t, at, vt, av, avt],axis=1)

            fused = tf.transpose(input, [0, 2, 1])
            fused = attention(fused, 10, 'attn')
            fused = tf.reshape(fused, [-1,fused.shape[1]])

            pred = linear(fused, 1, 'pred')

            return pred

    def load_mosei(self):

        ##### Code to load MOSEI Seq 50

        # load the audio dataset
        h5f = hdf5storage.loadmat(self.data_dir + 'train_acoustic_cohort.mat')
        audio_train = np.array(h5f['train_acoustic_cohort'])

        h5f = hdf5storage.loadmat(self.data_dir + 'test_acoustic_cohort.mat')
        audio_test = np.array(h5f['test_acoustic_cohort'])

        h5f = hdf5storage.loadmat(self.data_dir + 'valid_acoustic_cohort.mat')
        audio_val = np.array(h5f['valid_acoustic_cohort'])

        # load the visual dataset
        h5f = hdf5storage.loadmat(self.data_dir + 'train_visual_cohort.mat')
        video_train = np.array(h5f['train_visual_cohort'])

        h5f = hdf5storage.loadmat(self.data_dir + 'test_visual_cohort.mat')
        video_test = np.array(h5f['test_visual_cohort'])

        h5f = hdf5storage.loadmat(self.data_dir + 'valid_visual_cohort.mat')
        video_val = np.array(h5f['valid_visual_cohort'])

        # load the language dataset
        h5f = sio.loadmat(self.data_dir + 'train_text.mat')
        train_text = h5f['train_text']

        h5f = sio.loadmat(self.data_dir + 'test_text.mat')
        test_text = h5f['test_text']

        h5f = sio.loadmat(self.data_dir + 'valid_text.mat')
        valid_text = h5f['valid_text']

        # load the labels
        h5f = sio.loadmat(self.data_dir + 'train_labels.mat')
        y_train = h5f['train_labels']

        h5f = sio.loadmat(self.data_dir + 'test_labels.mat')
        y_test = h5f['test_labels']

        h5f = sio.loadmat(self.data_dir + 'valid_labels.mat')
        y_valid = h5f['valid_labels']


        print('min y train--->', np.min(y_train))

        print("Y train Size ---->",y_train.shape)

        train_audio  = audio_train
        train_video  = video_train
        test_audio   = audio_test
        test_video  = video_test
        valid_audio  = audio_val
        valid_video  = video_val

        y_train = np.reshape(y_train, (-1,1))
        y_test = np.reshape(y_test, (-1,1))
        y_valid = np.reshape(y_valid, (-1,1))

        print("Test Audio Size ---->",test_audio.shape)
        print("Test Visual Size ---->",test_video.shape)
        print("Test Text Size ---->",test_text.shape)
        print("Test Labels Size ---->",y_test.shape)


        return train_audio, train_text, train_video, y_train, test_audio, test_text, test_video, y_test,\
               valid_audio, valid_text, valid_video, y_valid

