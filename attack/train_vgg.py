# coding=UTF-8
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as st
from PIL import Image
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from scipy.misc import imread
from scipy.misc import imresize
from nets import resnet_v1, inception, vgg
import argparse
import skimage
from six.moves import xrange
slim = tf.contrib.slim
from tensorflow import reduce_sum
# 声明一些攻击参数
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
CHECKPOINTS_DIR = './model'
model_checkpoint_map = {
    'inception_v1': os.path.join(CHECKPOINTS_DIR,'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50','model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt')}
parser = argparse.ArgumentParser(description='PyTorch Attack')
parser.add_argument('--input_dir', type=str, default='../input')
parser.add_argument('--output_dir', type=str, default='../out')
parser.add_argument('--batch_size', type=int, default=16)
FLAGS = parser.parse_args()
input_dir = FLAGS.input_dir
output_dir = FLAGS.output_dir
max_epsilon = 255.0
max_epsilon = 2.0 * max_epsilon / 255.0
num_iter = 3
eps_iter = 2.0 * 6.5/255.0
cond_number = num_iter #restirct the logits function
rand_init = False
batch_size = FLAGS.batch_size
momentum = 1.0
model_type='vgg'
init_learning_rate= 0.01
epoch_num=25
num_classes=110
# 在图片数据输入模型前，做一些预处理
def preprocess_for_model(images, model_type,train=False):
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
        if not train:
          images = tf.stack([tmp_0,tmp_1,tmp_2],3)
        else:
          images[:,:,:,0] = images[:,:,:,0] - _R_MEAN
          images[:,:,:,1] = images[:,:,:,1] - _G_MEAN
          images[:,:,:,2] = images[:,:,:,2] - _B_MEAN
        return images
# 加载评测图片
def change_picture(image):
    left = 36
    right = 263
    top = 36
    bottom = 263

    new_image = image.copy()
    n = 1
    for i in range(4):
        new_image[left:right, top:bottom] += image[(left - i):(right - i), top:bottom]
        new_image[left:right, top:bottom] += image[(left + i):(right + i), top:bottom]
        new_image[left:right, top:bottom] += image[left:right, (top - i):(bottom - i)]
        new_image[left:right, top:bottom] += image[left:right, (top + i):(bottom + i)]
        n += 4

    new_image[left:right, top:bottom] /= n
    #new_image = new_image.astype(numpy.uint8)
    return new_image
def load_images_with_true_label(input_dir,train=True):
    images = []
    filenames = []
    true_labels = []
    idx = 0
    train_label = []
    train_data = []
    for root, _, file_names in os.walk(input_dir):
        np.random.shuffle(file_names)
        for _, file_name in enumerate(file_names):
            train_label.append(int(file_name.split('_')[0]))
            train_data.append(os.path.join(root,file_name))
    filename2label = {train_data[i] : train_label[i] for i in range(len(train_data))}
    for filename in filename2label.keys():
        image = imread(os.path.join(input_dir, filename), mode='RGB').astype(np.float)
        #image = change_picture(image)
        if train:
          image = img_aug(image)
        image = imresize(image,[224,224]).astype(np.float)
        image = 2.0*(image/255.0)-1.0
        images.append(image)
        filenames.append(filename)
        true_labels.append(filename2label[filename])
        idx += 1
        if idx == batch_size*2:
            images = np.array(images)
            yield filenames, images, true_labels
            filenames = []
            images = []
            true_labels = []
            idx = 0
# 定义MI_FGSM迭代攻击的计算图
def non_target_graph(ori_x,x, y, i, x_max, x_min, grad):

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
      processed_imgs_vgg_16, num_classes=num_classes, is_training=False, scope='vgg_16',reuse=True)

  end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
  end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])
  ########################
  #logits = tf.cond(tf.less(i,cond_number),lambda: end_points_vgg_16['logits'],lambda:end_points_vgg_16['logits']) 
  logits = end_points_inc_v1['Logits']/8 +end_points_res_v1_50['logits']/6 + end_points_vgg_16['logits']/14
  logits = tf.cond(tf.less(i,cond_number),lambda: logits,lambda:end_points_vgg_16['logits'])
  

  #pred = tf.argmax((end_points_inc_v1['Predictions']+end_points_res_v1_50['probs']+end_points_vgg_16['probs'])/3.0,1)
  pred = tf.argmax(logits,1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                  logits,
                                                  label_smoothing=0.0,
                                                  weights=1.0)
  noise = tf.gradients(cross_entropy, x)[0]
  #noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keep_dims=True)
  noise = noise / tf.reshape( 
    tf.contrib.keras.backend.std(tf.reshape(noise, [batch_size, -1]), axis=1),[batch_size, 1, 1, 1])

  #noise = clip_eta(noise, ord = 2, eps=eps)
  noise = momentum * grad + noise
  noise = noise / tf.reshape( 
    tf.contrib.keras.backend.std(tf.reshape(noise, [batch_size, -1]), axis=1),[batch_size, 1, 1, 1])
  x = x + alpha * tf.clip_by_value(tf.round(noise), -100, 100)
  #x = x + alpha * tf.sign(noise)
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
# Momentum Iterative FGSM
def one_hot(y,class_num=110):
  y_label=np.zeros([len(y),class_num],dtype=int)
  for i in range(len(y)):
    y_label[i,y[i]]=1
  return y_label
def img_aug(img):
  img = Image.fromarray(img.astype(np.uint8))
  image_width = img.size[0]
  image_height = img.size[1]
  crop_size = np.random.randint(0.3*image_width,image_width)
  nh = np.random.randint(0, image_width - crop_size)
  nw = np.random.randint(0, image_width - crop_size)  
  random_region = (nh,nw,nh+crop_size,nw+crop_size)
  img = img.crop(random_region) 
  #random_factor = np.random.randint(0, 5) / 10.  # 随机因子
  #color_image = ImageEnhance.Color(img).enhance(random_factor)  # 调整图像的饱和度
  #random_factor = np.random.randint(0, 5) / 10.  # 随机因子
  #brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
  #random_factor = np.random.randint(0,5) / 10.  # 随机因1子
  #contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
  #random_factor = np.random.randint(0,5) / 10.  # 随机因子
  #img = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
  img = img.rotate(np.random.randint(0, 5))
  #print(np.array(img).shape)
  return np.array(img)
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
    #ori_x,x_adv, _, _, _, _, _ = tf.while_loop(stop, non_target_graph, [x_input, adv_x, y, i, x_max, x_min, grad])

    #training setting
    train_img = tf.placeholder(tf.float32,shape=[None,224,224,3])
    train_label = tf.placeholder(tf.float32,shape = [None, 110]) 
    with slim.arg_scope(vgg.vgg_arg_scope()):
      logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
        train_img, num_classes=110, is_training=True, scope='vgg_16')
    end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
    end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])
 
    predict = tf.argmax(end_points_vgg_16['probs'],1)
    logits = end_points_vgg_16['logits']    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_label,logits=logits))
    learning_rate = tf.placeholder(tf.float32,name='learning_rate')
    optimizer =  tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(cost)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,tf.argmax(train_label,1)),tf.float32))
    #saver = tf.train.Saver(tf.global_variables(scope='resnet_v1_50'))
    
    ori_x,x_adv, _, _, _, _, _ = tf.while_loop(stop, non_target_graph, [x_input, adv_x, y, i, x_max, x_min, grad])
    # Run computation
    s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
    s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
    s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer()) 
      s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
      s1.restore(sess, model_checkpoint_map['inception_v1'])
      s3.restore(sess, model_checkpoint_map['vgg_16'])
      
      for time in range(epoch_num):
        train_acc = []
        count = 0
        train_loss = []
        val_acc = []
        val_loss = []
        epoch_learning_rate = init_learning_rate
        for filenames, raw_images, true_labels in load_images_with_true_label(input_dir,train=True):
            raw_images = list(raw_images)
            adv_images = sess.run(x_adv, feed_dict={x_input: np.array(raw_images[:16])})
            raw_images[:16] = list(adv_images)
            labels = one_hot(true_labels,110)
            img = (np.array(raw_images)+1.0)*0.5*255.0
            img = preprocess_for_model(img, model_type,train=True)
            train_feed_dict = {
              train_img: img,
              train_label: labels,
              learning_rate:epoch_learning_rate}
            a,batch_loss = sess.run([train,cost],feed_dict=train_feed_dict)
            batch_acc = accuracy.eval(feed_dict = train_feed_dict)
            train_acc.append(batch_acc)
            train_loss.append(batch_loss)
            count+=1
            if count%100==0 :
                print('acc: ',np.mean(np.array(train_acc)),' loss: ',np.mean(np.array(train_loss)))
                #break
        for filenames, raw_images, true_labels in load_images_with_true_label(output_dir,train=False):
            raw_images = list(raw_images)
            adv_images = sess.run(x_adv, feed_dict={x_input: np.array(raw_images[:16])})
            raw_images[:16] = list(adv_images)
            labels = one_hot(true_labels,110)
            img = (np.array(raw_images)+1.0)*0.5*255.0
            img = preprocess_for_model(img, model_type,train=True)            
            train_feed_dict = {
              train_img : img,
              train_label: labels,
              learning_rate:epoch_learning_rate
             }
            pre,batch_acc =sess.run([predict,accuracy],feed_dict = train_feed_dict)
            val_acc.append(batch_acc)
            val_loss.append(batch_loss)
            count+=1
            #break
            #print(pre,'\n',true_labels,batch_acc)
        print('val_acc: ',np.mean(np.array(val_acc)),' loss: ',np.mean(np.array(val_loss)))
        s3.save(sess=sess,save_path='./train_model/%s_%s.ckpt'%(model_type,time))
        #s2.restore(sess,'./train_model/%s_%s.ckpt'%(model_type,time))
if __name__=='__main__':
    non_target_mi_fgsm_attack(input_dir, output_dir)
    pass
