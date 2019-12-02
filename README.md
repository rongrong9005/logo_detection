# logo_detection
A good model can solve a series of wide problems. Our project aims to make a flexible model which can be easily defined and switched queries based on current tasks.

We divide this project into two stage, first finding
all possible logo in an image. Second, making comparison with
input brand.

# A. Prepare YOLO:
We use Keras as our deep learning framework. Before training
model, we should download pretrained yolo model from yolo
GitHub. Also, we need to first build python version 3.7
environment. In this project, we run the implement based on
anaconda environment, and some requirement packages:
keras-2.2.5, numpy-1.16.5, opencv-3.4.2, tensorflow-gpu-
1.14.0. The total training dataset storage is 3.56 GB and
pretrained yolo model is 242 MB. We use one graph card
GTX1070 to train.

# B. Feature extractor:
We used NASNet to get the input brand features. NASNet is
given in Keras-application, so we only need to download from
Keras without output layer. If the memory of test GPU is not
enough, because the size of NASNetLarge is 343 MB, you can
choose NASNetMobile from Keras-application which only
has 23 MB.
