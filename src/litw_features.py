import cv2
from keras import Model
import numpy as np
import metrics
import utils
from pyreadline import Readline
readline = Readline()
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def extract_litw_logos(filename, new_path=''):
    """
    Given Logos in The Wild dataset, extract all logos from images.

    Args:
      filename: text file, where each line defines the logos in the image
        specified. The format is:
          path-to-file1.jpg xmin,ymin,xmax,ymax,class_id[,confidence] xmin,ymin,xmax,ymax,class_id[,confidence]
          path-to-file2.jpg xmin,ymin,xmax,ymax,class_id[,confidence]
      new_path: replace /home/ubuntu/logohunter in each path (used if text file
        was created on machine with different directory structure)
    Returns:
      all_logos: list of np.arrays for each logo
      brand_map: brand id (in range 0,...,n_brands) for each extracted logo
    """
    img_list_lbl, bbox_list_lbl = metrics.read_txt_file(filename)

    if new_path != '':
        img_list_lbl = [ path.replace('/home/ubuntu/logohunter', new_path) for path in img_list_lbl]
    all_logos = []
    brand_map = []
    for idx in range(len(bbox_list_lbl)):

        im = cv2.imread(img_list_lbl[idx])[:,:,::-1]
        # im = cv2.resize(im,(128,128))
        for bb in bbox_list_lbl[idx]:
            if bb[3]-bb[1] < 10 or bb[2]-bb[1] < 10 or bb[3]>im.shape[0] or bb[2]> im.shape[0]:
                continue
            image = im[bb[1]:bb[3], bb[0]:bb[2]]
            try:
                image = cv2.resize(image,(128,128))
            except Exception as e:
                print(img_list_lbl[idx])
                print(str(e))

            all_logos.append(image)
            # all_logos.append(im[bb[1]:bb[3], bb[0]:bb[2]])
            brand_map.append(bb[-1])

    return all_logos, brand_map

def extract_litw_features(filename, model, my_preprocess):
    """
    Given Logos in The Wild dataset, extract all logos from images and extract
    features by applying truncated model (flattening W * H * n_filters features
    from last layer).

    Args:
      filename: text file, where each line defines the logos in the image
        specified. The format is:
          path-to-file1.jpg xmin,ymin,xmax,ymax,class_id[,confidence] xmin,ymin,xmax,ymax,class_id[,confidence]
          path-to-file2.jpg xmin,ymin,xmax,ymax,class_id[,confidence]
    Returns:
      features: (n_logos, n_features)-shaped np.array of features
      all_logos: list of np.arrays for each logo
      brand_map: brand id (in range 0,...,n_brands) for each extracted logo
    """

    all_logos, brand_map = extract_litw_logos(filename)
    # for i in all_logos:
    #     print(np.array(i).shape)
        # logo.append(cv2.resize(i,(128,128)))
    print("**************************************")
    print(np.array(all_logos).shape)
    features = utils.features_from_image(all_logos, model, my_preprocess)

    return features, all_logos, brand_map

if __name__ == '__main__':

    model, preprocess_input, input_shape = utils.load_extractor_model('NASNet', flavor=0)
    my_preprocess = lambda x: preprocess_input(utils.pad_image(x, input_shape))
    print('Extracting features from LogosInTheWild database (train set) - this will take a while (~5 minutes)')
    features, all_logos, brand_map = extract_litw_features('data_all_train.txt', model, my_preprocess)

    print('Processed {} logos, transformed into feature vectors'.format(len(features)))

    # # save inception features at default size 299*299
    # utils.save_features('inception_logo_features.hdf5', features, brand_map, input_shape)
    #
    # # save features for Inception with smaller input: 200 instead of 299 - last layer is 4*4 instead of 8*8
    # # Extract features at last layer as well as after last 3 inception blocks (mixed9,8,7)
    # input_shape = (200,200,3)
    # new_preprocess = lambda x: preprocess_input(utils.pad_image(x, input_shape))
    #
    # trunc_layer = [-1, 279, 248, 228]
    # for i_layer in range(4):
    #     model_out = Model(inputs=model.inputs, outputs=model.layers[trunc_layer[i_layer]].output)
    #     features = utils.features_from_image(all_logos, model_out, new_preprocess)
    #
    #     extra = '_trunc{}'.format(i_layer) if i_layer > 0 else ''
    #     utils.save_features('inception_logo_features_200{}.hdf5'.format(extra), features, brand_map, input_shape)
    #
    #
    # save features for VGG16 at 3 different input scales
    # from keras.applications.vgg16 import VGG16
    # from keras.applications.vgg16 import preprocess_input
    # model = VGG16(weights='imagenet', include_top=False)
    #
    # for n in [224,128,64]:
    #     input_shape = (n,n,3)
    #     new_preprocess = lambda x: preprocess_input(utils.pad_image(x, input_shape))
    #     features = utils.features_from_image(all_logos, model, new_preprocess)
    #     utils.save_features('vgg16_logo_features_{}.hdf5'.format(n), features, brand_map, input_shape)

    from keras.applications.nasnet import NASNetMobile
    from keras.applications.nasnet import preprocess_input
    model_out = NASNetMobile(weights='imagenet', include_top=False)
    input_shape = (224, 224, 3)

    new_preprocess = lambda x: preprocess_input(utils.pad_image(x, input_shape))
    features = utils.features_from_image(all_logos, model, new_preprocess)
    utils.save_features('NASNet_logo_features_{}.hdf5'.format(224), features, brand_map, input_shape)

    # from keras.applications.nasnet import NASNetLarge
    # from keras.applications.nasnet import preprocess_input
    # model_out = NASNetLarge(weights='imagenet', include_top=False)
    # input_shape = (331, 331, 3)
    #
    # new_preprocess = lambda x: preprocess_input(utils.pad_image(x, input_shape))
    # features = utils.features_from_image(all_logos, model, new_preprocess)
    # utils.save_features('NASNet_logo_features_{}.hdf5'.format(331), features, brand_map, input_shape)