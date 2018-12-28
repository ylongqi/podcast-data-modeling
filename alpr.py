## The implementation of the Generative Adversarial Network (GAN) is modified from https://github.com/carpedm20/DCGAN-tensorflow
import os
import time
import math
import tensorflow as tf
import numpy as np
import pickle
import random
from ops import *
import scipy.misc

def get_spectr(path):
    
    return (np.load(path) + 2.0) / 2.0
    
def save_spectrs(spectrs, path):
    
    assert len(spectrs) <= 64, "Too many spectrograms to save. Maximum: 64."
    
    if len(spectrs) <= 32:
        size = (4, 8)
    elif len(spectrs) <= 64:
        size = (8, 8)
    
    spectrs = (spectrs + 1.)/2.
    
    h, w = spectrs.shape[1], spectrs.shape[2]
    merged_spectr = np.zeros((h * size[0], w * size[1]))
    for idx, spectr in enumerate(spectrs):
        i = idx % size[1]
        j = idx // size[1]
        merged_spectr[j * h:j * h + h, i * w:i * w + w] = spectr[:,:,0]
    merged_spectr = np.squeeze(merged_spectr)
    
    scipy.misc.imsave(path, merged_spectr)

def conv_out_size_same(size, stride):
    
    return int(math.ceil(float(size) / float(stride)))

class ALPR(object):
    
    """ALPR model
    """
    
    def __init__(self,
                 files,
                 feature_dim=4096,
                 input_height=128,
                 input_width=512,
                 batch_size=64,
                 output_height=128,
                 output_width=512,
                 z_dim=100,
                 gf_dim=64,
                 df_dim=64):
        
        """Model initialization

        Args:
            files (list): a list of spectrogram files used for training.
            feature_dim: the dimensionality of ALPR.
            input_height: the height of real spectrograms.
            input_width: the width of real spectrograms.
            batch_size: training batch size.
            output_height: the height of generated spectrograms.
            output_width: the width of generated spectrograms.
            z_dim: the dimensionality of the vector z.
            gf_dim: controlling the model size of the generator.
            df_dim: controlling the model size of the discriminator.
        """
        
        self.files = files

        self.batch_size = batch_size
        self.feature_dim = feature_dim

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        
        self.build_model()
        self.sess = tf.Session()
    
    def build_model(self):
        
        spectr_dims = [self.input_height, self.input_width, 1]
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + spectr_dims, name='real_spectr')
        self.D, self.D_logits, self.D_features = self.discriminator(self.inputs)

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.G = self.generator(self.z)
        self.g_samples = self.sampler(self.z)
        self.D_, self.D_logits_, self.D_features_, = self.discriminator(self.G)
        
        self.z_sum = tf.summary.histogram("z", self.z)
        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
            
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                              labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                              labels=tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.losses.mean_squared_error(tf.reduce_mean(self.D_features, axis=0), 
                                                   tf.reduce_mean(self.D_features_, axis=0))
        

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=80)
    
    def generator(self, z):
        
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as scope:
            
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            
            z_ = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin')

            h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4],
                          name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], 
                          name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1],
                          name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, 1],
                          name='g_h4')

            return tf.nn.tanh(h4)
    
    def discriminator(self, spectr):
        
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as scope:

            h0 = lrelu(conv2d(spectr, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(
                self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(
                self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(
                self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = lrelu(
                self.d_bn4(
                    linear(tf.reduce_mean(h3, [1, 2], keepdims=False),
                        self.feature_dim, 'd_h4_lin')))
            h5 = linear(h4, 1, 'd_h5_lin')
            
            return tf.nn.sigmoid(h5), h5, h4
    
    def sampler(self, z):
        
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as scope:
            
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            
            h0 = tf.reshape(linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(  h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4],
                        name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2],
                        name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1],
                        name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, 1], 
                          name='g_h4')

            return tf.nn.tanh(h4)
            
    def train(self,
              total_epoch=3,
              sample_num=64,
              checkpoint_dir='alpr_model/checkpoint',
              sample_dir='alpr_model/samples',
              log_dir='alpr_model/logs',
              learning_rate=0.0002, 
              beta1=0.5):
    
        """Training ALPR

        Args:
            total_epoch: the total number of training epochs.
            sample_num: the number of randomly generated spetrograms to inspect/save.
            checkpoint_dir: the path to store ALPR.
            sample_dir: the path to store randomly generated spectrograms.
            log_dir: the path to store training logs, e.g., loss.
            learning_rate: learning rate of Adam.
            beta1: beta1 of Adam.
        """
    
        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                  .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                  .minimize(self.g_loss, var_list=self.g_vars)
        
        self.sess.run(tf.global_variables_initializer())
        
        self.g_sum = tf.summary.merge([self.z_sum, self.d__sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(sample_num, self.z_dim))
        
        counter = 1
        start_time = time.time()
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        for epoch in range(total_epoch):
            
            random.shuffle(self.files)
            batch_idxs = len(self.files) // self.batch_size

            for idx in range(batch_idxs):
                    
                batch_files = self.files[idx * self.batch_size:(idx+1) * self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                batch_spectrs = [get_spectr(batch_file) for batch_file in batch_files]
                batch_spectrs = np.array(batch_spectrs).astype(np.float32)[:, :, :, None]

                _, errD, summary_str = self.sess.run([d_optim, self.d_loss, self.d_sum],
                                                     feed_dict={self.inputs: batch_spectrs,
                                                                self.z: batch_z})

                self.writer.add_summary(summary_str, counter)

                for g_i in range(2):
                    batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                    _, errG, summary_str = self.sess.run([g_optim, self.g_loss, self.g_sum], 
                                                         feed_dict={self.inputs: batch_spectrs, self.z: batch_z})
                
                self.writer.add_summary(summary_str, counter)
                
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs, time.time() - start_time, errD, errG))
                
                if np.mod(counter, 1000) == 1:
                    samples = self.sess.run(self.g_samples, feed_dict={self.z: sample_z})
                    save_spectrs(samples, './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
                    self.saver.save(self.sess, os.path.join(checkpoint_dir, 'alpr.model'), global_step=counter)
                    
                counter += 1

if __name__ == '__main__':
    
    ## Please specify a list of Spectrogram files (Numpy array) for training
    ## e.g., ['spectrograms/s_0.npy', ...]
    ## files = ...
    
    assert files is not None, "Please specify a list of Spectrogram files for training."
    model = ALPR(files)
    model.train()