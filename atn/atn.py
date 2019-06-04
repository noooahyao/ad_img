import tensorflow as tf


slim=tf.contrib.slim

def ATN(inputs,
        final_endpoint='Conv2d_7b_1x1',
        align_feature_maps=False,
        scope=None
        train=False):
    if output_stride != 8 and output_stride != 16:
      raise ValueError('output_stride must be 8 or 16.')

    padding = 'SAME' if align_feature_maps else 'VALID'

    end_points = {}

    def add_and_check_final(name, net):
        end_points[name] = net
        return name == final_endpoint

    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d,slim.conv2d_transpose],
                            stride=1, padding='SAME'):
          # 149 x 149 x 32
          net = slim.conv2d(inputs, 32, 3, stride=2, padding=padding,
                            scope='Conv2d_1a_3x3')
          if add_and_check_final('Conv2d_1a_3x3', net): return net, end_points

          # 147 x 147 x 32
          net = slim.conv2d(net, 32, 3, padding=padding,
                            scope='Conv2d_2a_3x3')
          if add_and_check_final('Conv2d_2a_3x3', net): return net, end_points
          # 147 x 147 x 64
          net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
          if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points
          # 73 x 73 x 64
          net = slim.max_pool2d(net, 3, stride=2, padding=padding,
                                scope='MaxPool_3a_3x3')
          if add_and_check_final('MaxPool_3a_3x3', net): return net, end_points
          # 73 x 73 x 80
          net = slim.conv2d(net, 80, 1, padding=padding,
                            scope='Conv2d_3b_1x1')
          if add_and_check_final('Conv2d_3b_1x1', net): return net, end_points
          # 71 x 71 x 192
          net = slim.conv2d(net, 192, 3, padding=padding,
                            scope='Conv2d_4a_3x3')
          if add_and_check_final('Conv2d_4a_3x3', net): return net, end_points
          # 35 x 35 x 192
          net = slim.max_pool2d(net, 3, stride=2, padding=padding,
                                scope='MaxPool_5a_3x3')
          if add_and_check_final('MaxPool_5a_3x3', net): return net, end_points
          #padding(37,37,192)
          net = tf.pad(net,paddings = [[0, 0], [1,1], [1,1], [0, 0]])    

          #deconv(4,4,512,stride=2)
          net = slim.conv2d_transpose(net,512,4,stride=2, scope='deconv_1')

          #deconv(3,3,256,stride=2)
          net = slim.conv2d_transpose(net,256,3,stride=2, scope='deconv_2')

          #deconv(4,4,128,stride=2)
          net = slim.conv2d_transpose(net,128,4,stride=2, scope='deconv_3')

          #padding(299,299,128)
          net = tf.pad(net,paddings = [[0, 0], [2,1], [2,1], [0, 0]])
          
          #conv(4,4,3)
          net = slim.conv2d_transpose(net,3,4, scope='deconv_4')
          end_points['image'] = net
          #random noise
          net = net + tf.random_normal(net.shape,6) 
          
    return end_points
      
def ATN_arg_scope(weight_decay=0.00004,
                  batch_norm_decay=0.9997,
                  batch_norm_epsilon=0.001):
  """Returns the scope with the default parameters for inception_resnet_v2.
  Args:
    weight_decay: the weight decay for weights variables.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.
  Returns:
    a arg_scope with the parameters needed for inception_resnet_v2.
  """
  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connectedi,slim.conv2d_transpose],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d,slim.conv2d_transpose], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as scope:
      return scope
      
      
      
      
      
      
      
      
      
      
