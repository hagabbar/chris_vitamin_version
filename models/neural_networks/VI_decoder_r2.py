import collections

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import math as m

from . import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

class VariationalAutoencoder(object):

    def __init__(self, name, wrap_mask, nowrap_mask, m1_mask, m2_mask, sky_mask, n_input1=4, n_input2=256, n_output=3, n_channels=3, n_weights=2048, drate=0.2, n_filters=8, filter_size=8, maxpool=4, strides=1, dilations=1, batch_norm=True, twod_conv=False):
        
        self.n_input1 = n_input1                    # actually the output size
        self.n_input2 = n_input2                    # actually the output size
        self.n_output = n_output                    # the input data size
        self.n_channels = n_channels                # the number of channels/detectors
        self.n_weights = n_weights                  # the number of weights were layer
        self.n_hlayers = len(n_weights)             # the number of fully connected layers
        self.n_conv = len(n_filters)                # the number of convolutional layers
        self.n_filters = n_filters                  # the number of filters in each conv layer
        self.filter_size = filter_size              # the filter sizes in each conv layer
        self.strides = strides
        self.dilations = dilations
        self.maxpool = maxpool                      # the max pooling sizes in each conv layer
        self.name = name                            # the name of the network
        self.drate = drate                          # dropout rate
        self.wrap_mask = wrap_mask                  # mask identifying wrapped indices
        self.nowrap_mask = nowrap_mask              # mask identifying non-wrapped indices
        self.m1_mask = m1_mask                      # the mask identifying the m1 parameter
        self.m2_mask = m2_mask                      # the mask identifying the m2 parameter
        self.sky_mask = sky_mask                    # the mask identifying the sky (RA,dec) parameters
        self.nonlinear_loc_nowrap = tf.sigmoid      # activation for non-wrapped location params
        self.nonlinear_loc_wrap = tf.sigmoid        # activation for wrapped location params
        self.nonlinear_loc_m1 = tf.sigmoid          # activation for mass params
        self.nonlinear_loc_m2 = tf.sigmoid          # activation for mass params
        self.nonlinear_loc_sky = tf.identity        # activation for sky params
        self.nonlinear_scale_nowrap = tf.identity   # activation for non-wrapped scale params
        self.nonlinear_scale_wrap = tf.nn.relu      # activation for wrapped scale params
        self.nonlinear_scale_m1 = tf.nn.relu        # activation for mass params
        self.nonlinear_scale_m2 = tf.nn.relu        # activation for mass params  
        self.nonlinear_scale_sky = tf.nn.relu       # activation for sky params
        self.nonlinearity = tf.nn.relu              # activation between hidden layers
        self.batch_norm = batch_norm
        self.twod_conv = twod_conv
        if self.twod_conv:
            self.conv_out_size_t = n_input2 - 2*int(self.filter_size[0]/2)
        else:
            self.conv_out_size_t = n_input2
        for i in range(self.n_conv):
            self.conv_out_size_t = np.ceil(self.conv_out_size_t/strides[i])
            self.conv_out_size_t = np.ceil(self.conv_out_size_t/maxpool[i])
        self.conv_out_size_t = int(self.conv_out_size_t*n_filters[-1])
        if self.twod_conv:
            self.conv_out_size_t *= self.n_channels
        #self.conv_out_size_t = n_channels*int(self.conv_out_size_t*n_filters[-1])

        network_weights = self._create_weights()
        self.weights = network_weights

    def calc_reconstruction(self, z, y, training=True):
        with tf.name_scope("VI_decoder_r2"):

            # Reshape input to a 3D tensor - single channel
            if self.n_conv>0:
                if self.twod_conv:
                    conv_pool_t = tf.reshape(y, shape=[-1, self.n_input2, self.n_channels,1])
                    conv_pool_t = tf.concat([tf.reshape(conv_pool_t[:,:,-1,:],[-1,self.n_input2,1,1]),conv_pool_t,tf.reshape(conv_pool_t[:,:,0,:],[-1,self.n_input2,1,1])],axis=2)
                    conv_padding = 'VALID'
                else:
                    conv_pool_t = tf.reshape(y, shape=[-1, self.n_input2, 1, self.n_channels])
                    conv_padding = 'SAME'
                #conv_pool_t0 = tf.reshape(conv_pool_t[:,:,0], shape=[-1, self.n_input2, 1])
                #conv_pool_t1 = tf.reshape(conv_pool_t[:,:,1], shape=[-1, self.n_input2, 1])
                #conv_pool_t2 = tf.reshape(conv_pool_t[:,:,2], shape=[-1, self.n_input2, 1])

                for i in range(self.n_conv):            
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    conv_pre_t = tf.add(tf.nn.conv2d(conv_pool_t, self.weights['VI_decoder_r2'][weight_name+'t'],strides=[self.strides[i],1],dilations=[self.dilations[i],1],padding=conv_padding),self.weights['VI_decoder_r2'][bias_name+'t'])
                    conv_post_t = self.nonlinearity(conv_pre_t)
                    conv_pool_t = tf.nn.max_pool2d(conv_post_t,ksize=[self.maxpool[i],1],strides=[self.maxpool[i],1],padding='SAME')
                    conv_padding = 'SAME'
                    #conv_pre_t0 = tf.add(tf.nn.conv1d(conv_pool_t0, self.weights['VI_decoder_r2'][weight_name+'t'],stride=self.strides[i],dilations=self.dilations[i],padding='SAME'),self.weights['VI_decoder_r2'][bias_name+'t'])
                    #conv_pre_t1 = tf.add(tf.nn.conv1d(conv_pool_t1, self.weights['VI_decoder_r2'][weight_name+'t'],stride=self.strides[i],dilations=self.dilations[i],padding='SAME'),self.weights['VI_decoder_r2'][bias_name+'t'])
                    #conv_pre_t2 = tf.add(tf.nn.conv1d(conv_pool_t2, self.weights['VI_decoder_r2'][weight_name+'t'],stride=self.strides[i],dilations=self.dilations[i],padding='SAME'),self.weights['VI_decoder_r2'][bias_name+'t'])
                    #conv_post_t0 = self.nonlinearity(conv_pre_t0)
                    #conv_post_t1 = self.nonlinearity(conv_pre_t1)
                    #conv_post_t2 = self.nonlinearity(conv_pre_t2)
                    #conv_pool_t0 = tf.nn.max_pool1d(conv_post_t0,ksize=self.maxpool[i],strides=self.maxpool[i],padding='SAME')
                    #conv_pool_t1 = tf.nn.max_pool1d(conv_post_t1,ksize=self.maxpool[i],strides=self.maxpool[i],padding='SAME')
                    #conv_pool_t2 = tf.nn.max_pool1d(conv_post_t2,ksize=self.maxpool[i],strides=self.maxpool[i],padding='SAME')
     
                #conv_pool_t = tf.concat([conv_pool_t0,conv_pool_t1,conv_pool_t2],axis=-1)
                fc = tf.concat([z,tf.reshape(conv_pool_t, [-1, self.conv_out_size_t])],axis=1) 

            else:
                fc = tf.concat([z,y],axis=1)

            hidden_dropout = fc
            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden_' + str(i)
                bn_name = 'VI_bn_hidden_' + str(i)
                hidden_pre = tf.add(tf.matmul(hidden_dropout, self.weights['VI_decoder_r2'][weight_name]), self.weights['VI_decoder_r2'][bias_name])
                if self.batch_norm:
                    hidden_batchnorm = tf.layers.batch_normalization(hidden_pre,axis=-1,center=False,scale=False,
                                   beta_initializer=tf.zeros_initializer(),
                                   gamma_initializer=tf.ones_initializer(),
                                   moving_mean_initializer=tf.zeros_initializer(),
                                   moving_variance_initializer=tf.ones_initializer(),
                                   trainable=True,epsilon=1e-3,training=training)
                    hidden_post = self.nonlinearity(hidden_batchnorm)
                else:
                    hidden_post = self.nonlinearity(hidden_pre)
                hidden_dropout = tf.layers.dropout(hidden_post,rate=self.drate)
            loc_all = tf.add(tf.matmul(hidden_dropout, self.weights['VI_decoder_r2']['w_loc']), self.weights['VI_decoder_r2']['b_loc'])
            scale_all = tf.add(tf.matmul(hidden_dropout, self.weights['VI_decoder_r2']['w_scale']), self.weights['VI_decoder_r2']['b_scale'])

            # split up the output into non-wrapped and wrapped params and apply appropriate activation
            loc_nowrap = self.nonlinear_loc_nowrap(tf.boolean_mask(loc_all,self.nowrap_mask + [False],axis=1))   # add an extra null element to the mask
            scale_nowrap = -1.0*self.nonlinear_scale_nowrap(tf.boolean_mask(scale_all,self.nowrap_mask,axis=1)) 
            loc_m1 = self.nonlinear_loc_m1(tf.boolean_mask(loc_all,self.m1_mask + [False],axis=1))             # add an extra null element to the mask
            scale_m1 = -1.0*self.nonlinear_scale_m1(tf.boolean_mask(scale_all,self.m1_mask,axis=1))
            loc_m2 = self.nonlinear_loc_m2(tf.boolean_mask(loc_all,self.m2_mask + [False],axis=1))            # add an extra null element to the mask
            scale_m2 = -1.0*self.nonlinear_scale_m2(tf.boolean_mask(scale_all,self.m2_mask,axis=1))
            loc_wrap = self.nonlinear_loc_wrap(tf.boolean_mask(loc_all,self.wrap_mask + [False],axis=1))    # add an extra null element to the mask 
            scale_wrap = -1.0*self.nonlinear_scale_wrap(tf.boolean_mask(scale_all,self.wrap_mask,axis=1))
            loc_sky = self.nonlinear_loc_sky(tf.boolean_mask(loc_all,self.sky_mask + [True],axis=1))        # add an extra element to the mask for the 3rd sky parameter
            scale_sky = -1.0*self.nonlinear_scale_sky(tf.boolean_mask(scale_all,self.sky_mask,axis=1))      # send back both params but we wil only use 1
            return loc_nowrap, scale_nowrap, loc_wrap, scale_wrap, loc_m1, scale_m1, loc_m2, scale_m2, loc_sky, scale_sky   

    def _create_weights(self):
        all_weights = collections.OrderedDict()

        # Decoder
        with tf.variable_scope("VI_DEC"):
            all_weights['VI_decoder_r2'] = collections.OrderedDict()
            
            if self.n_conv>0:
                dummy_t = self.n_channels
                #dummy_t = 1
                for i in range(self.n_conv):
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    if self.twod_conv:
                        all_weights['VI_decoder_r2'][weight_name+'t'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size[i], self.n_channels*self.n_filters[i]),[self.filter_size[i], self.n_channels, 1, self.n_filters[i]]), dtype=tf.float32)
                    else:
                        all_weights['VI_decoder_r2'][weight_name+'t'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size[i], dummy_t*self.n_filters[i]),[self.filter_size[i], 1, dummy_t, self.n_filters[i]]), dtype=tf.float32)
                    all_weights['VI_decoder_r2'][bias_name+'t'] = tf.Variable(tf.zeros([self.n_filters[i]], dtype=tf.float32))
                    tf.summary.histogram(weight_name+'t', all_weights['VI_decoder_r2'][weight_name+'t'])
                    tf.summary.histogram(bias_name+'t', all_weights['VI_decoder_r2'][bias_name+'t'])
                    dummy_t = self.n_filters[i]
                fc_input_size = self.n_input1 + self.conv_out_size_t
            else:
                fc_input_size = self.n_input1 + self.n_input2*self.n_channels

            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden_' + str(i)
                all_weights['VI_decoder_r2'][weight_name] = tf.Variable(vae_utils.xavier_init(fc_input_size, self.n_weights[i]), dtype=tf.float32)
                all_weights['VI_decoder_r2'][bias_name] = tf.Variable(tf.zeros([self.n_weights[i]], dtype=tf.float32))
                tf.summary.histogram(weight_name, all_weights['VI_decoder_r2'][weight_name])
                tf.summary.histogram(bias_name, all_weights['VI_decoder_r2'][bias_name])
                fc_input_size = self.n_weights[i]

            all_weights['VI_decoder_r2']['w_loc'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output+1),dtype=tf.float32)  # +1 for extra sky param
            all_weights['VI_decoder_r2']['b_loc'] = tf.Variable(tf.zeros([self.n_output+1], dtype=tf.float32), dtype=tf.float32) # +1 for extra sky param
            tf.summary.histogram('w_loc', all_weights['VI_decoder_r2']['w_loc'])
            tf.summary.histogram('b_loc', all_weights['VI_decoder_r2']['b_loc'])
            all_weights['VI_decoder_r2']['w_scale'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output),dtype=tf.float32) # leaves 1 redundent paramster for sky
            all_weights['VI_decoder_r2']['b_scale'] = tf.Variable(tf.zeros([self.n_output], dtype=tf.float32), dtype=tf.float32)  # leaves 1 redundent paramster for sky
            tf.summary.histogram('w_scale', all_weights['VI_decoder_r2']['w_scale'])
            tf.summary.histogram('b_scale', all_weights['VI_decoder_r2']['b_scale'])
            
            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
