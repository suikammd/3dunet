import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow  as tf
import h5py as hdf

nclass  = 2

def conv3d (input_, output_dim, f_size, is_training, scope='conv3d'):
    with tf.variable_scope (scope) as scope:
        # VGG network uses two 3*3 conv layers to effectively increase receptive field
        w1 = tf.get_variable ('w1', [f_size, f_size, f_size, input_.get_shape()[-1] , output_dim],
                             initializer=tf.truncated_normal_initializer (stddev=0.1))
        print(w1)
        conv1 = tf.nn.conv3d (input_, w1, strides=[1, 1, 1, 1, 1], padding='SAME')
        print(conv1)
        b1 = tf.get_variable('b1', [output_dim], initializer=tf.constant_initializer (0.0))
        conv1 = tf.nn.bias_add(conv1, b1)
        print(conv1)
        bn1 = tf.contrib.layers.batch_norm (conv1, is_training=is_training, scope='bn1', decay=0.9,
                                     zero_debias_moving_mean=True, variables_collections=['bn_collections'])
        r1 = tf.nn.relu(bn1)
        
        w2 = tf.get_variable('w2', [f_size, f_size, f_size, output_dim, output_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        print(w2)
        conv2 = tf.nn.conv3d(r1, w2, strides=[1, 1, 1, 1, 1], padding='SAME')
        print(conv2)
        b2 = tf.get_variable('b2', [output_dim], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.bias_add(conv2, b2)
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_training, scope='bn2', decay=0.9,
                            zero_debias_moving_mean=True, variables_collections=['bn_collections'])
        r2 = tf.nn.relu(bn2)
        return r2

def deconv3d(input_, output_shape, f_size, is_training, scope='deconv3d'):
    with tf.variable_scope(scope) as scope:
        output_dim = output_shape[-1]
        w = tf.get_variable('w', [f_size, f_size, f_size, output_dim, input_.get_shape()[-1]] ,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        deconv = tf.nn.conv3d_transpose (input_, w, output_shape, strides=[1, f_size, f_size, f_size, 1], padding='SAME')
        bn = tf.contrib.layers.batch_norm(deconv, is_training=is_training, scope='bn', decay=0.9,
                            zero_debias_moving_mean=True, variables_collections=['bn_collections'])
        r = tf.nn.relu(bn)
        return r
    
def crop_and_concat(x1, x2):
    x1_shape = x1.get_shape().as_list()
    x2_shape = x2.get_shape().as_list() 
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop  = tf.slice(x1, offsets, size)
    return tf.concat ([x1_crop, x2], 4)

def conv_relu (input_, output_dim, f_size, s_size, scope='conv_relu'):
    with tf.variable_scope(scope) as scope:
        w = tf.get_variable('w', [f_size, f_size, f_size, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv3d(input_, w, strides=[1, s_size, s_size, s_size, 1], padding='VALID')
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
        r = tf.nn.relu(conv)
        return r


def build_generator(LR, keep_prob):
  
  
  conv_size =3
  dropout=0.5
  deconv_size = 2
  pool_stride_size = 2
  pool_kernel_size = 3 # Use a larger kernel
  layers  = 3
  features_root  = 32

  # Encoding path
  connection_outputs = []
  for layer in range(layers):
      features = 2** layer * features_root
      if layer == 0:
          prev = LR
      else:
        prev = pool

      conv = conv3d(prev, features, conv_size, is_training=1, scope='encoding' + str(layer))
      connection_outputs.append(conv)
      pool = tf.nn.max_pool3d(conv, ksize=[1, pool_kernel_size, pool_kernel_size, pool_kernel_size, 1],
                                    strides=[1, pool_stride_size, pool_stride_size, pool_stride_size, 1],
                                    padding='SAME')
        
  bottom = conv3d(pool, 2**layers * features_root, conv_size, is_training=True, scope='bottom')
  bottom = tf.nn.dropout (bottom, keep_prob)

  # Decoding path
  for layer in range(layers):
      conterpart_layer = layers - 1 - layer
      features = 2**conterpart_layer * features_root
      if layer == 0:
          prev = bottom
      else:
          prev = conv_decoding
      
      shape = prev.get_shape().as_list()
      deconv_output_shape = [tf.shape(prev)[0], shape[1] * deconv_size, shape[2] * deconv_size,
                             shape[3] * deconv_size, features]
      deconv = deconv3d (prev, deconv_output_shape, deconv_size, is_training=1,
                        scope='decoding' + str(conterpart_layer))
      cc = crop_and_concat(connection_outputs[conterpart_layer], deconv)
      conv_decoding = conv3d (cc, features, conv_size, is_training=True,
                             scope='decoding' + str(conterpart_layer))
  
  with tf.variable_scope('logits') as scope:
      w = tf.get_variable('w', [1, 1, 1, conv_decoding.get_shape()[-1], nclass],
                          initializer=tf.truncated_normal_initializer(stddev=0.1))
      logits = tf.nn.conv3d(conv_decoding, w, strides=[1, 1, 1, 1, 1], padding='SAME')
      b = tf.get_variable('b', nclass, initializer=tf.constant_initializer(0.0))
      logits  = tf.nn.bias_add(logits, b)

  return logits



def train():

  batch_size=20
  
  input_T2f = './T2Flair_patch.mat'
  load_data_inputT2f = hdf.File(input_T2f)
  datainputT2f = load_data_inputT2f['T2Flair_patch']
  datainputT2f.resize([9400,32,32,32])
  
  input_label = './traininglabel.mat'
  load_data_inputlabel = hdf.File(input_label)
  datainputlabel = load_data_inputlabel['traininglabel']
  datainputlabel.resize([9400,32,32,32])

  totalnum = datainputlabel.shape[0]

  sess = tf.InteractiveSession ()

  with tf.name_scope('input'):
    Input = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 32, 2], name='Inputdata')
    labels = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 32, nclass], name='labels')
    keep_prob = tf.placeholder (tf.float32, name='dropout_ratio')
  
  logits = build_generator (Input, keep_prob)

  flat_logits = tf.reshape (logits, [-1, nclass])
  flat_labels = tf.reshape(labels, [-1, nclass])
  
  with tf.name_scope('loss_function'):
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits (logits=flat_logits,labels=flat_labels))
    loss = cross_entropy_loss
    tf.summary.scalar( 'loss', loss)

  with tf.name_scope('trainer'):
    trainer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize (loss)

  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('./suika1_rand', sess.graph)
  sess.run(tf.global_variables_initializer())

    
  def feed_dict(j):
    start = batch_size*(j-1)
    xs = np.zeros((batch_size,32,32,32,2))
    ys = np.zeros((batch_size,32,32,32,2))
    xs[0:batch_size,:,:,:,0] = datainputT2f[start:start+batch_size,:,:,:]
    xs[0:batch_size,:,:,:,1] = datainputT2f[start:start+batch_size,:,:,:]
    ys[0:batch_size,:,:,:,0] = datainputlabel[start:start+batch_size,:,:,:]
    ys[0:batch_size,:,:,:,1] = 1-datainputlabel[start:start+batch_size,:,:,:]
    return {Input: xs, labels: ys, keep_prob: 1}

  for i in range(0,30):
    for j in range(1,np.int16(totalnum/batch_size)+1):
      summary, _, lossp = sess.run([merged, trainer, loss], feed_dict=feed_dict(j))
      print('Loss at step %s batch %s: %s' % (i, j, lossp))
    train_writer.add_summary(summary,i)

  train_writer.flush()
  train_writer.close()

  saver = tf.train.Saver()
  save_path = saver.save(sess, "./suika1_rand/model_30epoch.ckpt")
  print("Model saved in file: ", save_path)

if __name__ == '__main__':
  train()
