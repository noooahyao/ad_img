# coding=UTF-8   no_attack
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as st
from PIL import Image,ImageEnhance
from scipy.misc import imread
from scipy.misc import imresize
from tensorflow.contrib.slim.nets import resnet_v1, inception, vgg
import argparse
import skimage
from six.moves import xrange
slim = tf.contrib.slim
from tensorflow import reduce_sum
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 声明一些攻击参数
CHECKPOINTS_DIR = './model'
model_checkpoint_map = {
    'inception_v1': os.path.join(CHECKPOINTS_DIR,'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50','model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt'),
    'inception_adv':os.path.join(CHECKPOINTS_DIR,'inception_adv','inception_15.ckpt'),
    #'resnet_adv': os.path.join(CHECKPOINTS_DIR, 'resnet_adv','resnet.ckpt'),
    'vgg_adv': os.path.join(CHECKPOINTS_DIR, 'vgg_adv', 'vgg.ckpt')}

parser = argparse.ArgumentParser(description='PyTorch Attack')
parser.add_argument('--input_dir', type=str, default='../input')
parser.add_argument('--output_dir', type=str, default='../out')
parser.add_argument('--batch_size', type=int, default=11)
FLAGS = parser.parse_args()
input_dir = FLAGS.input_dir
output_dir = FLAGS.output_dir
max_epsilon = 255
max_epsilon = 2.0 * max_epsilon / 255.0
num_iter = 50
eps_iter = 2.0 * 0.5/255.0
cond_number = num_iter#restirct the logits function
rand_init = False
batch_size = FLAGS.batch_size
momentum = 1.0
crop_pic = True
crop_size = 190
sig=4
prob=1.0
image_width = 224
image_height = 224
image_resize = 299

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
        image = 2.0*(ori_image/255.0)-1.0
        #image = skimage.util.random_noise(image, mode='gaussian', seed=None, clip=True)
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
        #image = color_change(image)
        # #image = (images[i] * 255.0).astype(np.uint8)
        # # resize back to [299, 299]
        ori_image = ori_images[i].astype(np.uint8)
        if crop_pic:image += ori_image
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
def robust_resize(images):
  img = tf.image.resize_bilinear(images,[324,324])
  img = tf.image.resize_bilinear(images,[124,124])
  img = tf.image.resize_bilinear(images,[224,224])
  return img
# 定义MI_FGSM迭代攻击的计算图
def input_diversity(input_tensor):
  """
  kernel_size=10
  p_dropout=0.1
  kernel = tf.divide(tf.ones((kernel_size,kernel_size,3,3),tf.float32),tf.cast(kernel_size**2,tf.float32))
  input_shape = input_tensor.get_shape()
  rand = tf.where(tf.random_uniform(input_shape) < tf.constant(p_dropout, shape=input_shape), 
    tf.constant(1., shape=input_shape), tf.constant(0., shape=input_shape))
  image_d = tf.multiply(input_tensor,rand)
  image_s = tf.nn.conv2d(input_tensor,kernel,[1,1,1,1],'SAME')
  input_tensor = tf.add(image_d,tf.multiply(image_s,tf.subtract(tf.cast(1,tf.float32),rand)))
  """
  rnd = tf.random_uniform((), image_width, image_resize, dtype=tf.int32)
  rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  h_rem = image_resize - rnd
  w_rem = image_resize - rnd
  pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
  pad_bottom = h_rem - pad_top
  pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
  pad_right = w_rem - pad_left
  padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
  padded.set_shape((input_tensor.shape[0],image_resize,image_resize, 3))
  ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(prob), lambda: padded, lambda: input_tensor)
  ret = tf.image.resize_images(ret, [224, 224],
    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return ret
def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  interval = (2*nsig+1.)/(kernlen)
  x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
  kern1d = np.diff(st.norm.cdf(x))
  kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
  kernel = kernel_raw/kernel_raw.sum()
  return kernel

def non_target_graph(ori_x, x, y, i, x_max, x_min, grad):  #[x_input, adv_x, y, i, x_max, x_min, grad]
  #eps = 2.0 * max_epsilon / 255.0
  eps = max_epsilon
  alpha = eps_iter
  num_classes = 110
  #image = robust_resize(x)
  image = input_diversity(x)
  with slim.arg_scope(inception.inception_v1_arg_scope()):
    logits_inc_v1, end_points_inc_v1 = inception.inception_v1(
      image, num_classes=num_classes, is_training=False)
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
  
  image = (image/255.0)*2.0-1.0
  with slim.arg_scope(inception.inception_v1_arg_scope()):
    logits_inc_adv, end_points_inc_adv = inception.inception_v1(
      image, num_classes=num_classes, is_training=False, scope='inception_adv')
 
  image = (image + 1.0)*0.5*255.0
  #processed_imgs_res_v1_50 = preprocess_for_model(image, 'resnet_v1_50')
  #with slim.arg_scope(resnet_v1.resnet_arg_scope()):
  #  logits_res_adv, end_points_res_adv = resnet_v1.resnet_v1_50(
  #    processed_imgs_res_v1_50, num_classes=num_classes,is_training=False,scope='adv_resnet')
  #end_points_res_adv['logits'] = tf.squeeze(end_points_res_adv['adv_resnet/logits'], [1, 2])
  #end_points_res_adv['probs'] = tf.nn.softmax(end_points_res_v1_50['logits'])

  processed_imgs_vgg_16 = preprocess_for_model(image, 'vgg_16')
  with slim.arg_scope(vgg.vgg_arg_scope()):
    logits_vgg_adv, end_points_vgg_adv = vgg.vgg_16(
      processed_imgs_vgg_16, num_classes=num_classes, is_training=False, scope='adv_vgg')
  end_points_vgg_adv['logits'] = end_points_vgg_adv['adv_vgg/fc8']
  end_points_vgg_adv['probs'] = tf.nn.softmax(end_points_vgg_adv['logits'])

  ########################
  #logits = tf.cond(tf.less(i,cond_number),lambda: end_points_vgg_16['logits'],lambda:end_points_vgg_16['logits']) 

  logits = (0.8*end_points_inc_v1['Logits'] + 1.1*end_points_res_v1_50['logits'] \
      + 0.6*end_points_vgg_16['logits'] + end_points_inc_adv['Logits'] \
      + end_points_vgg_adv['logits']) \
     /6.5
  # logits = tf.cond(tf.less(i,cond_number),lambda: logits,lambda:end_points_vgg_16['logits'])

  prediction = (end_points_inc_v1['Predictions'] + end_points_res_v1_50['probs'] \
      + end_points_vgg_16['probs'] + end_points_inc_adv['Predictions'] \
      + end_points_vgg_adv['probs']) \
     /5.0
  pred = tf.argmax(prediction,1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                  logits,
                                                  label_smoothing=0.0,
                                                  weights=1.0)
  #another loss function
  #loss = tf.reshape(tf.nn.top_k(prediction,110)[0][:,0],[10,1])-tf.reshape(\
  #       tf.nn.top_k(prediction,110)[0][:,109],[10,1])
  #item1 = tf.reshape(tf.cast(tf.equal(pred,y),tf.float32),[batch_size,1])
  #item2 = tf.reshape(tf.cast(tf.equal(pred,y),tf.float32)-1.0,[batch_size,1])
  #loss = item1 * loss + item2 * loss
  loss = 1.0*cross_entropy# +0.2* loss
  
  noise = tf.gradients(loss, x)[0]
  kernel = gkern(7, sig).astype(np.float32)
  stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
  stack_kernel = np.expand_dims(stack_kernel, 3)
  noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')  
 
  #noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keep_dims=True)
  noise = noise / tf.reshape( 
    tf.contrib.keras.backend.std(tf.reshape(noise, [batch_size, -1]), axis=1),[batch_size, 1, 1, 1])

  #noise = clip_eta(noise, ord = 2, eps=eps)
  noise = momentum * grad + noise
  noise = noise / tf.reshape( 
    tf.contrib.keras.backend.std(tf.reshape(noise, [batch_size, -1]), axis=1),[batch_size, 1, 1, 1])
  
  perturb = tf.clip_by_value(tf.round(noise), -10, 10) 
  x = x + alpha * perturb
  # x = x + alpha * tf.sign(noise)
  x = tf.clip_by_value(x, -1.0, 1.0)
 
  #part_x,x = change_size(ori_x,x,size = crop_size)
  #x = x+part_x
  ############################PGD################
  #eta = x - ori_x
  #eta = clip_eta(eta, ord = 2, eps=eps)
  #x = ori_x + eta
  #x = tf.clip_by_value(x, x_min, x_max)
  ##############################################
  i = tf.add(i, 1)
  #random_noise = tf.random_normal(x.shape,6/255,6/255,dtype=x.dtype)
  #x = tf.cond(tf.less(i,num_iter),lambda: x,lambda:random_noise+x)
  #x = tf.clip_by_value(x, -1, 1)
  return ori_x, x , y, i, x_max, x_min, noise

def stop(adv_x,x, y, i, x_max, x_min, grad):
  return tf.less(i, num_iter)

# Momentum Iterative FGSM
def change_size(ori_images,adv_images,size=220):
  #size = 220   # <224
  start = int((224-size)/2)
  ori_images[:, start: start + size, start: start + size, :] = ori_images[:, start: start + size, start: start + size, :] - ori_images[:, start: start + size, start: start + size, :]
  adv_images[:, : start, :, :] = adv_images[:, :start, :, :] - adv_images[:, : start, :, :] - 1
  adv_images[:, start + size:, :, :] = adv_images[:, start + size:, :, :] - adv_images[:, start + size:, :, :] - 1
  adv_images[:, start: start + size, : start, :] = adv_images[:, start: start + size, : start, :] - adv_images[:, start: start + size, : start, :] - 1
  adv_images[:, start: start + size, start + size:, :] = adv_images[:, start: start + size, start + size:, :] - adv_images[:, start: start + size, start + size:, :] - 1
  return ori_images,adv_images
def color_change(images):
  for i in range(batch_size):
    img = images[i]
    img = Image.fromarray(img.astype(np.uint8))
    random_factor = np.random.randint(0, 1) / 100.  # 随机因子
    color_image = ImageEnhance.Color(img).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(0, 1) / 100.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(0, 1) / 100.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 1) / 100.  # 随机因子
    img = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  
    images[i]=np.array(img).astype(np.float)
    return images
def non_target_mi_fgsm_attack(input_dir, output_dir):

  # some parameter
  #eps = 2.0 * max_epsilon / 255.0
  eps = max_epsilon
  batch_shape = [None, 224, 224, 3]

  #_check_or_create_d/ir(output_dir)

  with tf.Graph().as_default():
    # Prepare graph
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
      grad = tf.sign(tf.random_normal(tf.shape(x_input),
                              tf.cast(0, x_input.dtype),
                              tf.cast(1*2/255.0, x_input.dtype),
                              dtype=x_input.dtype))/255.0    
    else:
      grad = tf.zeros(tf.shape(x_input)) 
    adv_x = x_input + grad
    adv_x = tf.clip_by_value(adv_x,x_min, x_max) 
    ori_x,x_adv, _, _, _, _, _ = tf.while_loop(stop, non_target_graph, [x_input, adv_x, y, i, x_max, x_min, grad])

    # Run computation
    s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
    s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
    s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))
    s4 = tf.train.Saver(slim.get_model_variables(scope='inception_adv'))
    #s5 = tf.train.Saver(slim.get_model_variables(scope='adv_resnet'))
    s6 = tf.train.Saver(slim.get_model_variables(scope='adv_vgg'))
    with tf.Session() as sess:
      s1.restore(sess, model_checkpoint_map['inception_v1'])
      s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
      s3.restore(sess, model_checkpoint_map['vgg_16'])
      s4.restore(sess, model_checkpoint_map['inception_adv'])
      #s5.restore(sess, model_checkpoint_map['resnet_adv'])
      s6.restore(sess, model_checkpoint_map['vgg_adv']) 
      for ori_images, filenames, raw_images, true_labels in load_images_with_true_label(input_dir):
        adv_images = sess.run(x_adv,feed_dict={x_input: raw_images})
        #adv_images = color_change(adv_images)
        #print(adv_images.shape,ori_images.shape)        
        if crop_pic:ori_images,adv_images = change_size(ori_images,adv_images,size = crop_size)
        save_images(ori_images, adv_images, filenames, output_dir)
        
if __name__=='__main__':
    non_target_mi_fgsm_attack(input_dir, output_dir)
    pass
