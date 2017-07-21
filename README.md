tensorflow-svhn
===

</br>

* Background
* File Structure
* Usage
* Accuracy

</br>

Background
---
This project is based on tensorflow 1.0, and python 2.7.

I spent quite a few days trying to improve the accuracy of this model, some cannot work, and some can. I have achieved 4.8% error rate on sequencial digits recognition. As a beginner of machine learning, it's quite good for me. But there are still quite a lot modern net structure I haven't use, so I believe it should be able to achieve even higher accuracy.

</br>

File Structure
---
* `ckpt/`&nbsp; &nbsp; &nbsp; #store checkpoint files
* `train_data/`
  * `train/`&nbsp; &nbsp; &nbsp; #extracted from train.tar.gz
  * `test/`&nbsp; &nbsp; &nbsp; &nbsp; #extracted from test.tar.gz
  * `extra/`&nbsp; &nbsp; &nbsp; #extracted from extra.tar.gz
  * `full_train_imgs.tfrecords`&nbsp; &nbsp; &nbsp; #generated using svhn_data.py
  * `full_test_imgs.tfrecords`
  * `full_extra_imgs.tfrecords`
* `digit_struct.py`&nbsp; &nbsp; &nbsp; #data structure for reading original images 
* `svhn_data.py`&nbsp; &nbsp;&nbsp; &nbsp; #convert images to tfrecord files
* `svhn.py`&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; #model, training operation, loss operation
* `svhn_input.py`&nbsp; &nbsp; &nbsp; #generate input queue for training and evaluation
* `svhn_train.py`
* `svhn_eval.py`
* `multi_digit_reader.py`

</br>

Usage
---
1. Download [train.tar.gz](http://ufldl.stanford.edu/housenumbers/train.tar.gz), [test.tar.gz](http://ufldl.stanford.edu/housenumbers/test.tar.gz), [extra.tar.gz](http://ufldl.stanford.edu/housenumbers/extra.tar.gz)
2. Extract them into `/train_data`
3. Run `svhn_data.py` to generate tfrecord files
4. The data is ready now. You can run `svhn_train.py` to train it from start, or copy everything from `ckpt-95.1%-acc/` to `ckpt/`(if there is no `ckpt/` folder, create one), and run `svhn_eval.py` to get the model accuracy
5. If you want to train the model from start, make sure there is nothing in `ckpt/`, or it will load the ckeckpoints from `ckpt/`. The checkpoint is saved in `train_data/`, if you want to continue from your last training, then just put your last checkpoint into `ckpt/`
6. Run python multi_digit_reader.py image-name.png to read a complete image. This is not accurate at all, I'm trying to come up with a better way.

</br>

Accuracy
---

|without extra images(70K training images set) </t> | 76% | </br>
|use extra images(600K training+extra images set) | 86% | </br>
|extra + 6 conv + 1 fc | 89.8% | </br>
|extra + 6 conv + 2 fc | 91.2% | </br>
|extra + 7 conv + 2 fc | 92.2% | </br>
|extra + 7 conv + 2 fc + densely connect | 92.2% | </br>
|extra + 8 conv + 2 fc | cannot train | </br>
|extra + 7 conv + 2 fc + inception block | cannot train | </br>
|extra + 7 conv + 2 fc + spatial transformer | cannot train | </br>
|extra + 7 conv + 2 fc + increase number of params | 93.3% | </br>
|extra + 7 conv + 2 fc + increase number of params + bacth normalization | 94.5% | </br>
|extra + 7 conv + 2 fc + increase number of params + bacth normalization | 94.5% | </br>
|extra + 7 conv + 2 fc + increase number of params + bacth normalization + clear some comments(???) | 95.1% | </br>
|extra + 7 conv + 2 fc + increase number of params + bacth normalization + max-avg pooling | 95.2% | </br>
|Update Nvidia driver from 375 -> 381 | 95.6% | </br>(WTF???)



