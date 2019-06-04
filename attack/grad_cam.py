# coding=UTF-8   no_attack
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as st
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
from nets import resnet_v1, inception, vgg
import argparse
import skimage
from six.moves import xrange
slim = tf.contrib.slim
from tensorflow import reduce_sum
# 声明一些攻击参数
CHECKPOINTS_DIR = './model'
model_checkpoint_map = {
    'inception_v1': os.path.join(CHECKPOINTS_DIR,'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50','model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt'),
    'resnet_adv':os.path.join(CHECKPOINTS_DIR, 'resnet_adv','resnet_4.ckpt')}

parser = argparse.ArgumentParser(description='PyTorch Attack')
parser.add_argument('--input_dir', type=str, default='../input')
parser.add_argument('--output_dir', type=str, default='../out')
parser.add_argument('--batch_size', type=int, default=1)
FLAGS = parser.parse_args()
input_dir = FLAGS.input_dir
output_dir = FLAGS.output_dir
max_epsilon = 255
max_epsilon = 2.0 * max_epsilon / 255.0
num_iter = 3
eps_iter = 2.0 * 6.5/255.0
cond_number = num_iter-1#restirct the logits function
rand_init = False
batch_size = FLAGS.batch_size
momentum = 0.8

# 在图片数据输入模型前，做一些预处理
def preprocess_for_model(images, model_type):
    if 'inception' in model_type.lower():
        #images = tf.image.resize_bilinear(images, [224,224],align_corners=False)
        # tensor-scalar operation
        images = (images / 255.0) * 2.0 - 1.0
        return images

    if 'resnet' in model_type.lower() or 'vgg' in model_type.lower():
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        #images = tf.image.resize_bilinear(images, [224,224],align_corners=False)
        tmp_0 = images[:,:,:,0] - _R_MEAN
        tmp_1 = images[:,:,:,1] - _G_MEAN
        tmp_2 = images[:,:,:,2] - _B_MEAN
        images = tf.stack([tmp_0,tmp_1,tmp_2],3)
        return images
# 加载评测图片
def load_images_with_true_label(input_dir):
    ori_images = []
    images = []
    filenames = []
    true_labels = []
    idx = 0
    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    filename2label = {dev.iloc[i]['filename'] : dev.iloc[i]['trueLabel'] for i in range(len(dev))}
    for filename in filename2label.keys():
        ori_image = imread(os.path.join(input_dir, filename), mode='RGB').astype(np.float)
        ori_image = imresize(ori_image,[224,224]).astype(np.float)
        ori_images.append(ori_image)
        #image = (image /255.0) *2.0 -1.0
        #image = skimage.util.random_noise(image, mode='gaussian', seed=None, clip=True)
        image = 2.0*(ori_image/255.0)-1.0
        images.append(image)
        filenames.append(filename)
        true_labels.append(filename2label[filename])
        idx += 1
        if idx == batch_size:
            images = np.array(images)
            ori_images = np.array(ori_images)
            yield ori_images, filenames, images, true_labels
            ori_images = []
            filenames = []
            images = []
            true_labels = []
            idx = 0
    if idx > 0:
        images = np.array(images)
        ori_images = np.array(ori_images)
        yield ori_images, filenames, images, true_labels

def save_images(ori_images, images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        image = (((images[i] + 1.0) * 0.5) * 255.0).astype(np.uint8)
        ori_image = ori_images[i].astype(np.uint8)
        #image += ori_image
        image = imresize(image, [299, 299])
        Image.fromarray(image).save(os.path.join(output_dir, filename), format='PNG')

def clip_eta(eta, ord, eps):
  """
  Helper function to clip the perturbation to epsilon norm ball.
  :param eta: A tensor with the current perturbation.
  :param ord: Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
  :param eps: Epsilon, bound of the perturbation.
  """

  # Clipping perturbation eta to self.ord norm ball
  if ord not in [np.inf, 1, 2]:
    raise ValueError('ord must be np.inf, 1, or 2.')
  reduc_ind = list(xrange(1, len(eta.get_shape())))
  avoid_zero_div = 1e-12
  if ord == np.inf:
    eta = tf.clip_by_value(eta, -eps, eps)
  else:
    if ord == 1:
      raise NotImplementedError("The expression below is not the correct way"
                                " to project onto the L1 norm ball.")
      norm = tf.maximum(avoid_zero_div,
                        reduce_sum(tf.abs(eta),
                                   reduc_ind, keepdims=True))
    elif ord == 2:
      # avoid_zero_div must go inside sqrt to avoid a divide by zero
      # in the gradient through this operation
      norm = tf.sqrt(tf.maximum(avoid_zero_div,
                                reduce_sum(tf.square(eta),
                                           reduc_ind,
                                           keepdims=True)))
    # We must *clip* to within the norm ball, not *normalize* onto the
    # surface of the ball
    factor = tf.minimum(1., eps/norm)
    eta = eta * factor
  return eta       
# 定义MI_FGSM迭代攻击的计算图
def non_target_graph(ori_x, x, y, i, x_max, x_min, grad):  #[x_input, adv_x, y, i, x_max, x_min, grad]

  #eps = 2.0 * max_epsilon / 255.0
  eps = max_epsilon
  alpha = eps_iter
  num_classes = 110
  #image = preprocess_for_model(x, 'inception_v1')
  image = x
  with slim.arg_scope(inception.inception_v1_arg_scope()):
    logits_inc_v1, end_points_inc_v1 = inception.inception_v1(
      image, num_classes=num_classes, is_training=False, scope='InceptionV1')

  # rescale pixle range from [-1, 1] to [0, 255] for resnet_v1 and vgg's input
  #image = (((image + 1.0) * 0.5) * 255.0)
  image = (image + 1.0)*0.5*255.0
  processed_imgs_res_v1_50 = preprocess_for_model(image, 'resnet_v1_50')
  with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    logits_res_v1_50, end_points_res_v1_50 = resnet_v1.resnet_v1_50(
      processed_imgs_res_v1_50, num_classes=num_classes,is_training=False,scope='resnet_v1_50')
  end_points_res_v1_50['logits'] = tf.squeeze(end_points_res_v1_50['resnet_v1_50/logits'], [1, 2])
  end_points_res_v1_50['probs'] = tf.nn.softmax(end_points_res_v1_50['logits'])

  #image = (((x + 1.0) * 0.5) * 255.0)#.astype(np.uint8)
  processed_imgs_vgg_16 = preprocess_for_model(image, 'vgg_16')
  with slim.arg_scope(vgg.vgg_arg_scope()):
    logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
      processed_imgs_vgg_16, num_classes=num_classes, is_training=False, scope='vgg_16')

  end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
  end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])

  ########################
  #logits = tf.cond(tf.less(i,cond_number),lambda: end_points_vgg_16['logits'],lambda:end_points_vgg_16['logits']) 

  logits = end_points_inc_v1['Logits']/8 + end_points_res_v1_50['logits']/6 + end_points_vgg_16['logits']/14
  # logits = tf.cond(tf.less(i,cond_number),lambda: logits,lambda:end_points_vgg_16['logits'])
  # print(end_points_res_v1_50)
  # pred = tf.argmax(end_points_inc_v1['Predictions']+end_points_res_v1_50['probs']+end_points_vgg_16['probs'], 1)
  pred = tf.argmax(end_points_vgg_16['probs'],1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                  logits,
                                                  label_smoothing=0.0,
                                                  weights=1.0)
  noise = tf.gradients(cross_entropy, x)[0]
  noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keep_dims=True)
  noise = noise / tf.reshape( 
    tf.contrib.keras.backend.std(tf.reshape(noise, [batch_size, -1]), axis=1),[batch_size, 1, 1, 1])
  noise = momentum * grad + noise
  noise = noise / tf.reshape( 
    tf.contrib.keras.backend.std(tf.reshape(noise, [batch_size, -1]), axis=1),[batch_size, 1, 1, 1])
  x = x + alpha * tf.clip_by_value(tf.round(noise), -20, 20)
  # x = x + alpha * tf.sign(noise)
  # x = x + alpha * noise / tf.norm(noise, ord=2)
  x = tf.clip_by_value(x, x_min, x_max)
  ############################PGD################
  #eta = x - ori_x
  #eta = clip_eta(eta, ord = 2, eps=eps)
  #x = ori_x + eta
  #x = tf.clip_by_value(x, x_min, x_max)
  ##############################################
  i = tf.add(i, 1)
  return ori_x, x , y, i, x_max, x_min, noise

def stop(adv_x,x, y, i, x_max, x_min, grad):
  return tf.less(i, num_iter)
def chan_noise_size(adv_images,ori_images,size):
    #size = 220   # <224
    start = int((224-size)/2)
    ori_images[:, start: start + size, start: start + size, :] = ori_images[:, start: start + size, start: start + size, :] - ori_images[:, start: start + size, start: start + size, :]
    adv_images[:, : start, :, :] = adv_images[:, : start, :, :] - adv_images[:, : start, :, :] - 1
    adv_images[:, start + size:, :, :] = adv_images[:, start + size:, :, :] - adv_images[:, start + size:, :, :] - 1
    adv_images[:, start: start + size, : start, :] = adv_images[:, start: start + size, : start, :] - adv_images[:, start: start + size, : start, :] - 1
    adv_images[:, start: start + size, start + size:, :] = adv_images[:, start: start + size, start + size:, :] - adv_images[:, start: start + size, start + size:, :] - 1
    return adv_images
def grad_cam(x_input,sess,image):
        image = (image+1.0)*0.5*255.0
        img_vgg=preprocess_for_model(x_input, 'vgg_16')
        with slim.arg_scope(vgg.vgg_arg_scope()):
          logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
        img_vgg, num_classes=110, is_training=True, scope='vgg_16',reuse=True)
        end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
        end_points_vgg_16['pool5'] = end_points_vgg_16['vgg_16/pool5']
        end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])
        predict = tf.argmax(end_points_vgg_16['probs'],1)
        logits = end_points_vgg_16['logits']
        before_fc = end_points_vgg_16['pool5']
        probs = end_points_vgg_16['probs']        
        nb_classes = 110
        conv_layer = before_fc
        one_hot = tf.one_hot(predict, 110)
        signal = tf.multiply(logits, one_hot)
        loss = tf.reduce_mean(signal)
        #loss = tf.losses.softmax_cross_entropy(one_hot,
        #                                          logits,
        #                                          label_smoothing=0.0,
        #                                          weights=1.0)
        grads = tf.gradients(loss, conv_layer)[0]
        norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
        output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={x_input:image})
        grads_val = grads_val[0]
        output = output[0] 
        weights = np.mean(grads_val,axis=(0,1))                     # [512]
        cam = np.ones(output.shape[0 : 2], dtype = np.float32)  # [7,7]

        # Taking a weighted average
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]
        # Passing through ReLU
        #cam = imresize(cam, (224,224))
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = imresize(cam, (224,224))
        
        # Converting grayscale to 3-D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3,[1,1,3])       

        img = image[0]
        img = img/np.max(img)

        # Superimposing the visualization with the image.
        new_img = img+cam3
        new_img = new_img/np.max(new_img)
        #new_img = new_img.astype(np.uint8)
        return cam3
# Momentum Iterative FGSM
def non_target_mi_fgsm_attack(input_dir, output_dir):

  # some parameter
  #eps = 2.0 * max_epsilon / 255.0
  eps = max_epsilon
  batch_shape = [batch_size, 224, 224, 3]

  #_check_or_create_dir(output_dir)

  with tf.Graph().as_default():
    # Prepare graph
    raw_inputs = tf.placeholder(tf.uint8, shape=[None, 224,224, 3])

    # preprocessing for model input,
    # note that images for all classifier will be normalized to be in [-1, 1]
    #processed_imgs = preprocess_for_model(raw_inputs, 'inception_v1')

    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)
   
    # y = tf.constant(np.zeros([batch_size]), tf.int64)
    #y = tf.placeholder(tf.int32, shape=[batch_size])
    y = tf.constant(np.zeros([batch_size]), tf.int64)
    i = tf.constant(0)
    #grad = tf.zeros(shape=batch_shape)
    if rand_init:
      grad = tf.random_uniform(tf.shape(x_input),
                              tf.cast(-0.05, x_input.dtype),
                              tf.cast(0.05, x_input.dtype),
                              dtype=x_input.dtype)    
    else:
      grad = tf.zeros(tf.shape(x_input)) 
    adv_x = x_input + grad
    adv_x = tf.clip_by_value(adv_x,x_min, x_max) 
    ori_x,x_adv, _, _, _, _, _ = tf.while_loop(stop, non_target_graph, [x_input, adv_x, y, i, x_max, x_min, grad])
    #cam_img = grad_cam(x_input)
    # Run computation
    s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
    s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
    s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      s1.restore(sess, model_checkpoint_map['inception_v1'])
      s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
      s3.restore(sess, model_checkpoint_map['vgg_16'])
      for ori_images, filenames, raw_images, true_labels in load_images_with_true_label(input_dir):
        adv_images = sess.run(x_adv, feed_dict={x_input: raw_images})
        #cam_images = sess.run(cam_img,feed_dict={x_input:raw_images})
        cam_images = grad_cam(x_input,sess,raw_images)
        #adv_images = chan_noise_size(adv_images,ori_images,220)
        skimage.io.imsave(os.path.join(output_dir, 'cam_'+filenames[0]),cam_images)
        save_images(ori_images, adv_images, filenames, output_dir)
        

if __name__=='__main__':
    non_target_mi_fgsm_attack(input_dir, output_dir)
    pass
