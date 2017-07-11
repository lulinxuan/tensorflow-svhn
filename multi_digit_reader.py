import sys
import os
import numpy as np
import tensorflow as tf
import PIL.Image as Image

import svhn

import time
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile

WEIGHTS_FILE = "ckpt_reg/"
NUM_IMAGE_PER_BATCH = 512

pred_dict = {}
box_dict = {}
step = 0

def prediction_to_string(pred_array, pure, box_arr):
    global pred_dict, step, box_dict, NUM_IMAGE_PER_BATCH
    for i in range(len(pred_array)):
        sure = True    
        pred_str = ""
        for j in range(len(pred_array[i])):
            value = pred_array[i][j]
            if pred_array[i][j] != 10:
                pred_str += str(value)
                pos = pure[i][j][value]
                if pos < 0.4:
                    sure = False
        if sure and pred_str != "":
            if pred_str in pred_dict:
                pred_dict[pred_str] += 1
            else:
                pred_dict[pred_str] = 1

            if pred_str in box_dict:
                box_dict[pred_str].append(box_arr[i+NUM_IMAGE_PER_BATCH*(step-1)])
            else:
                box_dict[pred_str] = [box_arr[i+NUM_IMAGE_PER_BATCH*(step-1)]]


def generate_image_set(img):
    image = img
    shape = np.shape(image)
    out_arr=[]
    box_arr=[]
    if shape[0] < 20:
        #height less than 20
        width = int(shape[1] * (20.0/shape[0]))
        image = image.resize((width, 20), Image.ANTIALIAS)
    
    if shape[1] < 20:
        #width less than 20
        height = int(shape[0] * (20.0/shape[1]))
        image = image.resize((20, height), Image.ANTIALIAS)

    shape = np.shape(image)
    x_pos = 0
    y_pos = 0
    size = 20

    window_move_step = 3
    window_size_step = 5
    while size <= min(shape[0], shape[1]):
        x_pos = 0
        while x_pos+size <= shape[1]:
            y_pos = 0
            while y_pos+size <= shape[0]:
                out_arr.append(np.uint8(np.array(image.crop((x_pos, y_pos, x_pos+size, y_pos+size)).resize((64, 64), Image.ANTIALIAS))))
                box_arr.append([x_pos, y_pos, x_pos+size, y_pos+size])
                y_pos += window_move_step
            x_pos += window_move_step
        size += window_size_step
    return out_arr, box_arr

def get_data_batch(arr):
    global step, NUM_IMAGE_PER_BATCH
    if step*NUM_IMAGE_PER_BATCH > len(arr):
        return None

    if len(arr) - step*NUM_IMAGE_PER_BATCH < NUM_IMAGE_PER_BATCH:
        data = arr[step* NUM_IMAGE_PER_BATCH:]
        step += 1
        return data

    data = arr[step*NUM_IMAGE_PER_BATCH : (step+1)*NUM_IMAGE_PER_BATCH]
    step += 1
    return data


def detect(img_path):
    """
    The code is ugly here. THis part is still in experiment

    1. Using sliding window, record the window position where there is a number
    2. For every pixel in the image, calculate how many windows found in step 1 have covered it (cover means the pixel is in the window frame)
    3. Find all the pixels appears more than X%
    4. Find a frame covers all the pixels in step 3
    5. Inference the frame found in step 4
    """

    sample_img = Image.open(img_path)
    arr, box_arr = generate_image_set(sample_img)
    
    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, shape=(None, 64, 64, 3))
        X = tf.map_fn(tf.image.per_image_standardization, X)
        logits1, logits2, logits3, logits4, logits5, logits6 = svhn.net_1(X, 1.0)

        pred = tf.stack([\
          tf.argmax(tf.nn.softmax(logits2), 1),\
          tf.argmax(tf.nn.softmax(logits3), 1),\
          tf.argmax(tf.nn.softmax(logits4), 1),\
          tf.argmax(tf.nn.softmax(logits5), 1),\
          tf.argmax(tf.nn.softmax(logits6), 1)], axis=1)

        pred_pure = tf.stack([\
          tf.nn.softmax(logits2),\
          tf.nn.softmax(logits3),\
          tf.nn.softmax(logits4),\
          tf.nn.softmax(logits5),\
          tf.nn.softmax(logits6)], axis=1)

        with tf.Session() as sess:
          variable_averages = tf.train.ExponentialMovingAverage(svhn.MOVING_AVERAGE_DECAY)
          variables_to_restore = variable_averages.variables_to_restore()
          saver = tf.train.Saver(variables_to_restore)
          ckpt = tf.train.get_checkpoint_state('ckpt/')

          if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model restored."
            next_batch = get_data_batch(arr)
            while next_batch:
                feed_dict = {X: next_batch}
                predictions = sess.run(pred, feed_dict=feed_dict)
                predictions_pure = sess.run(pred_pure, feed_dict=feed_dict)
                pred_str = prediction_to_string(predictions, predictions_pure, box_arr)
                next_batch = get_data_batch(arr)
            global pred_dict, box_dict

            i=0
            while len(pred_dict) > 6 and i < len(pred_dict):
                key = pred_dict.keys()[i]

                value = pred_dict[key]
                count = 0
                for other_key in pred_dict:
                    if pred_dict[other_key] > value:
                        count += 1
                if count > 6:
                    del(pred_dict[key])
                    del(box_dict[key])
                else:
                    i += 1
            min_count = 0
            pos_arr = np.zeros((np.shape(sample_img)[0], np.shape(sample_img)[1]))
            for key in box_dict:
                for box in box_dict[key]:
                    min_count += 1
                    for x in range(box[0], box[2]):
                        for y in range(box[1], box[3]):
                            pos_arr[y][x] += 1

            min_count = int(float(min_count) * 0.35)
            pix = np.array(sample_img)
            for x in range(np.shape(sample_img)[1]):
                for y in range(np.shape(sample_img)[0]):
                    if pos_arr[y][x] < min_count:
                        pos_arr[y][x] = 0


            top = np.shape(sample_img)[0]-1
            left = np.shape(sample_img)[1]-1
            right = 0
            buttom = 0
            for x in range(np.shape(sample_img)[0]):
                for y in range(np.shape(sample_img)[1]):
                    if pos_arr[x][y] >0:
                        if x > right:
                            right = x
                        if x < left:
                            left = x
                        if y > buttom:
                            buttom = y
                        if y < top:
                            top = y

            if buttom == 0:
                return

            height = buttom - top
            width = right - left
            factor = 64.0 / float(max(height, width))

            cropped = sample_img.crop((top, left, buttom, right)).resize((int(height*factor), int(width*factor)), Image.ANTIALIAS)
            new_img = np.zeros((64,64,3))
            
            height = height * factor
            width = width * factor
            new_img[int((64-int(width)) / 2):int(width)+int((64-int(width)) / 2), int((64-int(height)) / 2):int(height)+int((64-int(height)) / 2) , :] = np.array(cropped)
            new_img = np.uint8(new_img)

            feed_dict = {X: [new_img]}
            predictions = sess.run(pred, feed_dict=feed_dict)
            print(predictions[0])

            plt.imshow(new_img)
            plt.show()




if __name__ == "__main__":
    img_path = None
    if len(sys.argv) > 1:
        print("Reading Image file:", sys.argv[1])
        if os.path.isfile(sys.argv[1]):
            img_path = sys.argv[1]
        else:
            raise EnvironmentError("Image file cannot be opened.")
    else:
        raise EnvironmentError("You must pass an image file to process")

    detect(img_path)
