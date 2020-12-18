import collections

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#from .layers import batch_normalization
import numpy as np
import math as m

from . import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

class VariationalAutoencoder(object):

    def __init__(self, name, n_input=256, n_output=4, n_channels=3, n_weights=2048, n_modes=2, n_hlayers=2, drate=0.2, n_filters=8, filter_size=8, maxpool=4, n_conv=2, batch_norm=True):
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_channels = n_channels
        self.n_weights = n_weights
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_hlayers = len(n_weights)
        self.n_conv = len(n_filters)
        self.n_modes = n_modes
        self.drate = drate
        self.maxpool = maxpool
        self.batch_norm = batch_norm

        network_weights = self._create_weights()
        self.weights = network_weights

        self.nonlinearity = tf.nn.relu
        self.nonlinearity_mean = tf.clip_by_value

    def _calc_z_mean_and_sigma(self,x, training=True):
        with tf.name_scope("VI_encoder_r1"):
 
            # Reshape input to a 3D tensor - single channel
            if self.n_conv is not None:
                conv_pool = tf.reshape(x, shape=[-1, self.n_input, 1, self.n_channels])
                for i in range(self.n_conv):
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    bn_beta_name = 'bn_beta_conv_' + str(i)
                    bn_scale_name = 'bn_scale_conv_' + str(i)
                    conv_pre = tf.add(tf.nn.conv2d(conv_pool, self.weights['VI_encoder_r1'][weight_name],strides=1,padding='SAME'),self.weights['VI_encoder_r1'][bias_name])
                    #if self.batch_norm:
                    #    conv_batchnorm = tf.layers.batch_normalization(conv_pre,axis=-1,center=False,scale=False,
                    #               beta_initializer=tf.zeros_initializer(),
                    #               gamma_initializer=tf.ones_initializer(),
                    #               moving_mean_initializer=tf.zeros_initializer(),
                    #               moving_variance_initializer=tf.ones_initializer(),
                    #               trainable=True,epsilon=1e-3,training=training) 
                    #    conv_post = self.nonlinearity(conv_batchnorm)
                    #else:
                    conv_post = self.nonlinearity(conv_pre)
                    conv_pool = tf.nn.max_pool2d(conv_post,ksize=[self.maxpool[i],1],strides=[self.maxpool[i],1],padding='SAME')

                fc = tf.reshape(conv_pool, [-1, int(self.n_input*self.n_filters[-1]/(np.prod(self.maxpool)))])

            else:
                fc = tf.reshape(x,[-1,self.n_input*self.n_channels])
           
            hidden_dropout = fc
            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden_' + str(i)
                bn_name = 'VI_bn_hidden_' + str(i)
                hidden_pre = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_r1'][weight_name]), self.weights['VI_encoder_r1'][bias_name])
                #if self.batch_norm:
                #    hidden_batchnorm = tf.layers.batch_normalization(hidden_pre,axis=-1,center=False,scale=False,
                #                   beta_initializer=tf.zeros_initializer(),
                #                   gamma_initializer=tf.ones_initializer(),
                #                   moving_mean_initializer=tf.zeros_initializer(),
                #                   moving_variance_initializer=tf.ones_initializer(),   
                #                   trainable=True,epsilon=1e-3,training=training)
                #    hidden_post = self.nonlinearity(hidden_batchnorm)
                #else:
                hidden_post = self.nonlinearity(hidden_pre)
                hidden_dropout = tf.layers.dropout(hidden_post,rate=self.drate)
            loc = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_r1']['w_loc']), self.weights['VI_encoder_r1']['b_loc'])
            scale_diag = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_r1']['w_scale_diag']), self.weights['VI_encoder_r1']['b_scale_diag'])
            weight = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_r1']['w_weight']), self.weights['VI_encoder_r1']['b_weight']) 


            tf.summary.histogram('loc', loc)
            tf.summary.histogram('scale_diag', scale_diag)
            tf.summary.histogram('weight', weight)
            return tf.reshape(loc,(-1,self.n_modes,self.n_output)), tf.reshape(scale_diag,(-1,self.n_modes,self.n_output)), tf.reshape(weight,(-1,self.n_modes))    

    def _create_weights(self):
        all_weights = collections.OrderedDict()
        with tf.variable_scope("VI_ENC_r1"):            
            all_weights['VI_encoder_r1'] = collections.OrderedDict()

            if self.n_conv is not None:
                dummy = self.n_channels
                for i in range(self.n_conv):
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    #bn_beta_name = 'bn_beta_conv_' + str(i)
                    #bn_scale_name = 'bn_scale_conv_' + str(i)
                    all_weights['VI_encoder_r1'][weight_name] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size[i], dummy*self.n_filters[i]),[self.filter_size[i], 1, dummy, self.n_filters[i]]), dtype=tf.float32)
                    all_weights['VI_encoder_r1'][bias_name] = tf.Variable(tf.zeros([self.n_filters[i]], dtype=tf.float32))
                    #all_weights['VI_encoder_r1'][bn_beta_name] = tf.Variable(tf.zeros([self.n_filters[i]], dtype=tf.float32))
                    #all_weights['VI_encoder_r1'][bn_scale_name] = tf.Variable(tf.zeros([self.n_filters[i]], dtype=tf.float32))
                    tf.summary.histogram(weight_name, all_weights['VI_encoder_r1'][weight_name])
                    tf.summary.histogram(bias_name, all_weights['VI_encoder_r1'][bias_name])
                    #tf.summary.histogram(bn_beta_name, all_weights['VI_encoder_r1'][bn_beta_name])
                    #tf.summary.histogram(bn_scale_name , all_weights['VI_encoder_r1'][bn_scale_name])
                    dummy = self.n_filters[i]

                fc_input_size = int(self.n_input*self.n_filters[-1]/(np.prod(self.maxpool)))
            else:
                fc_input_size = self.n_input*self.n_channels

            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden_' + str(i)
                #bn_mean_name = 'bn_mean_hidden_' + str(i)
                #bn_var_name = 'bn_var_hidden_' + str(i)
                all_weights['VI_encoder_r1'][weight_name] = tf.Variable(vae_utils.xavier_init(fc_input_size, self.n_weights[i]), dtype=tf.float32)
                all_weights['VI_encoder_r1'][bias_name] = tf.Variable(tf.zeros([self.n_weights[i]], dtype=tf.float32))
                #all_weights['VI_encoder_r1'][bn_mean_name] = tf.Variable(tf.zeros([self.n_weights[i]], dtype=tf.float32),trainable=False)
                #all_weights['VI_encoder_r1'][bn_var_name] = tf.Variable(tf.zeros([self.n_weights[i]], dtype=tf.float32),trainable=False)
                tf.summary.histogram(weight_name, all_weights['VI_encoder_r1'][weight_name])
                tf.summary.histogram(bias_name, all_weights['VI_encoder_r1'][bias_name])
                #tf.summary.histogram(bn_mean_name, all_weights['VI_encoder_r1'][bn_mean_name])
                #tf.summary.histogram(bn_var_name, all_weights['VI_encoder_r1'][bn_var_name])
                fc_input_size = self.n_weights[i]
            all_weights['VI_encoder_r1']['w_loc'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output*self.n_modes),dtype=tf.float32)
            all_weights['VI_encoder_r1']['b_loc'] = tf.Variable(tf.zeros([self.n_output*self.n_modes], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_loc', all_weights['VI_encoder_r1']['w_loc'])
            tf.summary.histogram('b_loc', all_weights['VI_encoder_r1']['b_loc'])
            all_weights['VI_encoder_r1']['w_scale_diag'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output*self.n_modes),dtype=tf.float32)
            all_weights['VI_encoder_r1']['b_scale_diag'] = tf.Variable(tf.zeros([self.n_output*self.n_modes], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_scale', all_weights['VI_encoder_r1']['w_scale_diag'])
            tf.summary.histogram('b_scale', all_weights['VI_encoder_r1']['b_scale_diag'])
            all_weights['VI_encoder_r1']['w_weight'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_modes),dtype=tf.float32)
            all_weights['VI_encoder_r1']['b_weight'] = tf.Variable(tf.zeros([self.n_modes], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_weight', all_weights['VI_encoder_r1']['w_weight'])
            tf.summary.histogram('b_weight', all_weights['VI_encoder_r1']['b_weight'])

            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
