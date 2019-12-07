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
has 23 MB. Thus, in our project, we use NASNetMobile. 
Otherwise, using NASNetLarge is also acceptable. You only
need to change two places: in "utils.py" file, in "load_extractor_model"
method, change import application "NASNetMobile" to "NASNetLarge.
Same as "litw_features.py" file. Notice that the input shape in NASNetLarge
is 331x331.


# Running for testing model
Here is three ways:

1. python logohunter.py  --image --input_brands ../data/test/test_brands/test_lexus.png \
                              --input_images ../data/test/lexus/ \
                              --output ../data/test/test_lexus --outtxt

2. python logohunter.py  --image --input_brands ../data/test/test_brands/test_golden_state.jpg  \
                              --input_images ../data/test/goldenstate/  \
                              --output ../data/test/test_gs --outtxt

3. python logohunter.py  --image --input_images data_test.txt  \
                              --input_brands ../data/test/test_brands/test_lexus.png  \
                              --outtxt --no_save_img


