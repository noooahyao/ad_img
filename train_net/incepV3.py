# coding=UTF-8
import os
import numpy as np
import tensorflow as tf
import scipy.stats as st
from PIL import Image
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from scipy.misc import imread
from scipy.misc import imresize
from nets import inception_v3
import argparse
import skimage
from six.moves import xrange
slim = tf.contrib.slim
from tensorflow import reduce_sum
# 声明一些攻击参数
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CHECKPOINTS_DIR = '../models'
parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--model', type=str, default='../models/inception_v3.ckpt')
parser.add_argument('--input_dir', type=str, default='../data/train')
parser.add_argument('--output_dir', type=str, default='../data/val')
parser.add_argument('--img_size', type=int, default=299)
parser.add_argument('--batch_size', type=int, default=32)
model_type='InceptionV3'
FineTune=True
FLAGS = parser.parse_args()
input_dir = FLAGS.input_dir
output_dir = FLAGS.output_dir
batch_size = FLAGS.batch_size
init_learning_rate= 0.001
epoch_num=2
num_classes=2

# 在图片数据输入模型前，做一些预处理
def preprocess_for_model(images, model_type,train=False):
    if 'inception' in model_type.lower():
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

    for num in range(len(train_data)):
        image = imread(train_data[num], mode='RGB').astype(np.float)
        #image = change_picture(image)
        if train:
          image = img_aug(image)
        image = imresize(image,[FLAGS.img_size,FLAGS.img_size]).astype(np.float)
        #image = 2.0*(image/255.0)-1.0
        images.append(image)
        true_labels.append(train_label[num])
        idx += 1
        if idx == batch_size:
            images = np.array(images)
            yield images, true_labels
            filenames = []
            images = []
            true_labels = []
            idx = 0
# 定义MI_FGSM迭代攻击的计算图

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
def one_hot(y,class_num=110):
  y_label=np.zeros([len(y),class_num],dtype=int)
  for i in range(len(y)):
    y_label[i,y[i]]=1
  return y_label
def Train(input_dir, output_dir):
  # some parameter
  batch_shape = [batch_size, FLAGS.img_size, FLAGS.img_size, 3]

  with tf.Graph().as_default():
    # Prepare graph

    train_img = tf.placeholder(tf.float32,shape=[None,FLAGS.img_size,FLAGS.img_size,3])
    train_label = tf.placeholder(tf.float32,shape = [None, 2]) 
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
      logits, end_points = inception_v3.inception_v3(
      train_img, num_classes=2,is_training=True,scope=model_type)
    
    predict = tf.argmax(end_points['Predictions'],1)
    logits = end_points['Logits']    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_label,logits=logits))
    
    learning_rate = tf.placeholder(tf.float32,name='learning_rate')
    optimizer =  tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train_step = optimizer.minimize(cost)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,tf.argmax(train_label,1)),tf.float32))
    
    # Run computation
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer()) 
      if FineTune:
          exclusion=['%s/Logits'%model_type,'%s/AuxLogits'%model_type]#去除class类里的描述
          except_logitis=slim.get_variables_to_restore(exclude=exclusion)
          #print(except_logitis)
          init_fn=slim.assign_from_checkpoint_fn(FLAGS.model,except_logitis,ignore_missing_vars=True)
          init_fn(sess)
          saver=tf.train.Saver(slim.get_model_variables(scope=model_type))
      else:
          saver=tf.train.Saver(slim.get_model_variables(scope=model_type))
          saver.restore(sess,FLAGS.model)
      for time in range(epoch_num):
        train_acc = []
        count = 0
        train_loss = []
        val_acc = []
        val_loss = []
        epoch_learning_rate = init_learning_rate
        for raw_images, true_labels in load_images_with_true_label(input_dir,train=True):
            import time
            start=time.time()
            labels = one_hot(np.array(true_labels),2)
            img = np.array((raw_images / 255.0) * 2.0 - 1.0)
            img = np.reshape(img,[batch_size,FLAGS.img_size,FLAGS.img_size,3])
            train_feed_dict = {
              train_img: img,
              train_label: labels,
              learning_rate:epoch_learning_rate}

            sess.run(train_step,feed_dict=train_feed_dict)
            batch_loss = cost.eval(feed_dict = train_feed_dict)
            batch_acc = accuracy.eval(feed_dict = train_feed_dict)
            train_acc.append(batch_acc)
            train_loss.append(batch_loss)
            count+=1
            end=time.time()
            break
            if count%100==0 :
                print('acc: ',np.mean(np.array(train_acc)),' loss: ',np.mean(np.array(train_loss)))
        print(end-start,"!!!")
        #saver.save(sess=sess,save_path='./train_model/%s_%s.ckpt'%(model_type,time))
        for raw_images, true_labels in load_images_with_true_label(output_dir,train=False):
            start=time.time()
            labels = one_hot(np.array(true_labels),2)
            img = np.array((raw_images / 255.0) * 2.0 - 1.0)
            img = np.reshape(img,[batch_size,FLAGS.img_size,FLAGS.img_size,3])
            train_feed_dict = {
              train_img : img,
              train_label: labels,
              learning_rate:epoch_learning_rate
             }
            batch_acc =sess.run([accuracy],feed_dict = train_feed_dict)
            val_acc.append(batch_acc)
            count+=1
            end=time.time()
            break
        print(end-start,"!!!")
        print('val_acc: ',np.mean(np.array(val_acc)))
        break
if __name__=='__main__':
    Train(input_dir, output_dir)
    pass
