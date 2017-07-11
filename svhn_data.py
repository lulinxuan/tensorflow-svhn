import os
import struct
import numpy as np
import scipy
import sys
import tarfile
from PIL import Image

from digit_struct import DigitStruct

from scipy.io import loadmat
from scipy import ndimage
from sklearn.cross_validation import train_test_split
from itertools import product
from six.moves.urllib.request import urlretrieve
import matplotlib.pyplot as plt
import tensorflow as tf


"""
Extract train.tar.gz, test.tar.gz, extra.tar.gz into train_data
so there is three folders: train_data/test, train_data/train, train_data/extra
running this file will generate three tfrecords files
"""


DATA_PATH = "train_data/"
CROPPED_DATA_PATH = DATA_PATH
FULL_DATA_PATH = DATA_PATH
PIXEL_DEPTH = 255
NUM_LABELS = 10

OUT_HEIGHT = 64
OUT_WIDTH = 64
NUM_CHANNELS = 3
MAX_LABELS = 5

last_percent_reported = None

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_data_file(file_name):
    file = open(file_name, 'rb')
    data = process_data_file(file)
    file.close()
    return data


def read_digit_struct(data_path):
    struct_file = os.path.join(data_path, "digitStruct.mat")
    dstruct = DigitStruct(struct_file)
    structs = dstruct.get_all_imgs_and_digit_structure()
    return structs

def convert_imgs_to_array(img_array, labels):
    rows = img_array.shape[0]
    cols = img_array.shape[1]
    chans = img_array.shape[2]
    num_imgs = img_array.shape[3]
    # Note: not the most efficent way but can monitor what is happening
    new_array = []
    for x in range(0, num_imgs):
        # TODO reuse normalize_img here
        image = img_array[:, :, :, x]
        image = Image.fromarray(image, 'RGB').resize([64, 64], Image.ANTIALIAS)
        image = np.array(image)
        color = image[0][0]
        image[:,0:16] = color
        image[:,48:64] = color

        label = labels[x]
        labels_array = np.ones([MAX_LABELS+1], dtype=np.uint8) * 10
        labels_array[0]=1
        labels_array[1]=label
        # plt.imshow(image)
        # plt.show()
        example = tf.train.Example(features=tf.train.Features(feature={
            'size': _int64_feature(64),
            'label': _bytes_feature(labels_array.tostring()),
            'image': _bytes_feature(image.tostring())
            }))
        # norm_vec = (255-image)*1.0/255.0
        # norm_vec -= np.mean(norm_vec, axis=0)
        new_array.append(example)
    return new_array


def convert_labels_to_one_hot(labels):
    labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
    return labels

def process_data_file(file):
    data = loadmat(file)
    imgs = data['X']
    labels = data['y'].flatten()
    labels[labels == 10] = 0
    img_array = convert_imgs_to_array(imgs, labels)
    return img_array


def get_data_file_name(master_set, dataset):
    if master_set == "cropped":
        if dataset == "train":
            data_file_name = "train_32x32.mat"
        elif dataset == "test":
            data_file_name = "test_32x32.mat"
        elif dataset == "extra":
            data_file_name = "extra_32x32.mat"
        else:
            raise Exception('dataset must be either train, test or extra')
    elif master_set == "full":
        if dataset == "train":
            data_file_name = "train.tar.gz"
        elif dataset == "test":
            data_file_name = "test.tar.gz"
        elif dataset == "extra":
            data_file_name = "extra.tar.gz"
    else:
        raise Exception('Master data set must be full or cropped')
    return data_file_name

def handle_tar_file(path):
    ''' Extract and return the data file '''
    # extract_data_file(file_pointer)

    structs = read_digit_struct(path)
    data_count = len(structs)

    img_data = []
    labels = []

    for i in range(data_count):
        lbls = structs[i]['label']
        file_name = os.path.join(path, structs[i]['name'])
        top = structs[i]['top']
        left = structs[i]['left']
        height = structs[i]['height']
        width = structs[i]['width']
        if(len(lbls) <= MAX_LABELS):
            # empty label/image has no number, but adding these images into the training step would decrease the accuracy...
            # so I do not use it at last
            label, empty_label = create_label_array(lbls)
            labels.append(label)
            img, empty = create_img_array(file_name, top, left, height, width, OUT_HEIGHT, OUT_WIDTH)
            img_data.append(img)
        else:
            print("Skipping {}, only images with less than {} numbers are allowed!").format(file_name, MAX_LABELS)

    new_array = []
    for x in range(0, len(img_data)):
        image = img_data[x]
        example = tf.train.Example(features=tf.train.Features(feature={
            'size': _int64_feature(64),
            'label': _bytes_feature(labels[x].tostring()),
            'image': _bytes_feature(image.tostring())
            }))
        new_array.append(example)

    return new_array


def create_svhn(dataset, master_set):
    path = DATA_PATH+dataset
    return handle_tar_file(path)

def write_tfrecord_file(data_array, data_set_name, data_path):
    writer = tf.python_io.TFRecordWriter(os.path.join(DATA_PATH, data_path+"_"+data_set_name+'_imgs.tfrecords'))
    for data in data_array:
        writer.write(data.SerializeToString())
    writer.close()

def create_label_array(el):
    """[count, digit, digit, digit, digit, digit]"""
    num_digits = len(el)  # first element of array holds the count
    labels_array = np.ones([MAX_LABELS+1], dtype=int) * 10
    labels_array[0] = num_digits

    for n in range(num_digits):
        if el[n] == 10: el[n] = 0  # reassign 0 as 10 for one-hot encoding
        labels_array[n+1] = el[n]
    

    return np.uint8(labels_array), np.uint8([0, 10, 10, 10, 10, 10])


def create_img_array(file_name, top, left, height, width, out_height, out_width):
    img = Image.open(file_name)
    
    img_top = np.amin(top)
    img_left = np.amin(left)
    img_height = np.amax(top) + np.amax(height) - img_top
    img_width = np.amax(left) + np.amax(width) - img_left

    box_left = np.floor(img_left - 0.1 * img_width)
    box_top = np.floor(img_top - 0.1 * img_height)
    box_right = np.amin([np.ceil(box_left + 1.2 * img_width), img.size[0]])
    box_bottom = np.amin([np.ceil(img_top + 1.2 * img_height), img.size[1]])
    cropped_img = img.crop((int(box_left), int(box_top), int(box_right), int(box_bottom)))#.resize([out_height, out_width], Image.ANTIALIAS)

    width = box_right - box_left
    height = box_bottom - box_top

    if width > height:
        factor = 64 / float(width)
    else:
        factor = 64 / float(height)

    cropped_img = cropped_img.resize([int(width*factor), int(height*factor)], Image.ANTIALIAS)

    new_img = np.zeros((64,64,3))
    cropped_img = np.array(cropped_img)

    new_img[int((64-int(height*factor)) / 2):int(height*factor)+int((64-int(height*factor)) / 2) , int((64-int(width*factor)) / 2):int(width*factor)+int((64-int(width*factor)) / 2) , :] = cropped_img
    new_img = np.uint8(new_img)

    #create empty image

    height = img.size[1]-1
    if img_left > img.size[0] - img_width - img_left:#left has more space than right
        width = img_left
        c = np.amin([width, height])
        empty_img = img.crop((int(img_left - c), int(0), int(img_left), int(c))).resize([out_height, out_width], Image.ANTIALIAS)
    else:
        width = img.size[0] - img_width - img_left
        c = np.amin([width, height])
        empty_img = img.crop((int(img_left + img_width), int(0), int(img_left + img_width + c), int(c))).resize([out_height, out_width], Image.ANTIALIAS)

    pix_empty = np.uint8(np.array(empty_img))

    return new_img, pix_empty



def generate_full_files():
    test_data = create_svhn('test', 'full')
    write_tfrecord_file(test_data, 'test', 'full')

    train_data = create_svhn('train', 'full')
    write_tfrecord_file(train_data, 'train', 'full')

    extra_data = create_svhn('extra', 'full')
    write_tfrecord_file(extra_data, 'extra', 'full')
    print("Full Files Done!!!")

def generate_cropped_files():
    test_data = create_svhn('test', 'cropped')
    write_tfrecord_file(test_data, 'test', 'cropped')

    train_data = create_svhn('train', 'cropped')
    write_tfrecord_file(train_data, 'train', 'cropped')

    print("Cropped Files Done!!!")


if __name__ == '__main__':
    generate_full_files()
