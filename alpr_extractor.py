import tensorflow as tf
from ops import *

class ALPRExtractor(object):
    
    """ALPR feature extractor
    
    Extract ALPR features using a trained model.
    """
    
    
    def __init__(self):
    
        df_dim = 64
        feature_dim = 4096
        
        with tf.Graph().as_default():
            
            d_bn1 = batch_norm(name='d_bn1')
            d_bn2 = batch_norm(name='d_bn2')
            d_bn3 = batch_norm(name='d_bn3')
            d_bn4 = batch_norm(name='d_bn4')

            self._spect_tf = tf.placeholder(tf.float32, [None, 128, 512, 1])

            with tf.variable_scope("discriminator") as scope:
                h0 = lrelu(conv2d(self._spect_tf, df_dim, name='d_h0_conv'))
                h1 = lrelu(d_bn1(conv2d(h0, df_dim * 2, name='d_h1_conv'), train=False))
                h2 = lrelu(d_bn2(conv2d(h1, df_dim * 4, name='d_h2_conv'), train=False))
                h3 = lrelu(d_bn3(conv2d(h2, df_dim * 8, name='d_h3_conv'), train=False))
                self._tf_features = lrelu(d_bn4(linear(tf.reduce_mean(h3, [1, 2], keepdims=False), feature_dim, 'd_h4_lin'), train=False))

            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()
    
    def load_model(self, path):
        """Load pretrained ALPR model

        Args:
            path (str): the directory to a pretrained ALPR model.

        """

        self._saver.restore(self._sess, path)
    
    def forward(self, spectrograms):
        """Extract ALPR features for given spectrograms (i.e., conduct a forward pass)

        Args:
            spectrograms (numpy array): input spectrograms.
        
        Returns:
            ALPR features.

        """
        
        return self._sess.run(self._tf_features,
                              feed_dict={self._spect_tf: spectrograms})