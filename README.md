<h2>Tensorflow-Image-Segmentation-Common-Carotid-Artery-UltraSound-Images (2025/02/17)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>

This is the first experiment of Image Segmentation for <b>CCAUS (Common Carotid Artery UltraSound) Images</b>
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and Mendeley <a href="https://data.mendeley.com/datasets/d4xt63mgjm/1">
Common Carotid Artery Ultrasound Images
</a>
<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 709x749 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/images/202201121748100022VAS_slice_1128.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/masks/202201121748100022VAS_slice_1128.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test_output/202201121748100022VAS_slice_1128.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/images/202201121748100022VAS_slice_2919.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/masks/202201121748100022VAS_slice_2919.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test_output/202201121748100022VAS_slice_2919.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/images/202201121834300037VAS_slice_949.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/masks/202201121834300037VAS_slice_949.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test_output/202201121834300037VAS_slice_949.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this CCAUSSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been take from the following Mendeley website:<br>
<a href="https://data.mendeley.com/datasets/d4xt63mgjm/1">
Common Carotid Artery Ultrasound Images
</a>
<br><br>
Momot, Agata (2022), “Common Carotid Artery Ultrasound Images”,<br>
 Mendeley Data, V1, doi: 10.17632/d4xt63mgjm.1
<br>
<br>
<b>Description</b><br>
The dataset consists of aquistified ultrasound images of the common carotid artery. 
The images were taken from a Mindary UMT-500Plus ultrasound machine with an L13-3s linear probe. 
The study group consisted of 11 subjects, with each person examined at least once on the left 
and right sides. 2 subjects were examined using the vascular modality and 8 using the carotid modality. 
Time series (DICOM) were converted to png files and cropped appropriately. Each person had 100 images, 
making a total of 1100 images. The dataset also includes corresponding expert masks (corresponding file name) 
made by a technician and verified by an expert. The collection can be used for carotid artery segmentation 
and geometry measurement and evaluation. 
<br>
Image resolution: 709 x 749 x 3 <br>
Number of images: 2200 <br>
File format: PNG<br>
<br>
<b>License:</b><br>
<a href="http://creativecommons.org/licenses/by/4.0/">
Creative Commons Attribution 4.0 International License.
</a>
<br>
<br>
<h3>
<a id="2">
2 CCAUS ImageMask Dataset
</a>
</h3>
 If you would like to train this CCAUS Segmentation model by yourself,
 please download the dataset <a href="https://data.mendeley.com/datasets/d4xt63mgjm/1">
<b>Common Carotid Artery Ultrasound Images</b>
</a>
<br><br>
The folder structure of the dataset is the following.<br>
<pre>
./Common Carotid Artery Ultrasound Images
   ├─Expert mask images
   │  ├─202201121748100022VAS_slice_1069.png
   │  ├─202201121748100022VAS_slice_1080.png
      ...
   │  └─202202071359200056VAS_slice_258.png
   └─US images
       ├─202201121748100022VAS_slice_1069.png
       ├─202201121748100022VAS_slice_1080.png
       ...
       └─202202071359200056VAS_slice_258.png
</pre>

Please run the following Python script to split the original 709x749 pixels dataset into <b>test</b>, <b>train</b> and <b>valid</b>
 subsets.<br>
<li><a href="./generator/split_master.py">split_master.py</a></li>

<pre>
./dataset
└─CCAUS
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>

<br>
<b>CCAUS Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/CCAUS_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not so large 
to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained CCAUS TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/CCAUSand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Enabled Batch Normalization.<br>
Defined a small <b>base_filters=16</b> and large <b>base_kernels=(9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Dataset class</b><br>
Specified ImageMaskDataset class.
<pre>
[dataset]
datasetclass  = "ImageMaskDataset"
resize_interpolation = "cv2.INTER_CUBIC"
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]

epoch_change_infer      = True
epoch_change_infer_dir  = "./epoch_change_infer"
epoch_change_tiledinfer = False
epoch_change_tiledinfer_dir = "./epoch_change_tiledinfer"
num_infer_images       = 6
</pre>

By using this callback, on every epoch_change, the epoch change tiledinfer procedure can be called
 for 6 image in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/asset/epoch_change_tiled_infer_at_start.png" width="1024" height="auto"><br>
<br>
<br>
<b>Epoch_change_inference output at ending (90,91,92)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/asset/epoch_change_tiled_infer_at_end.png" width="1024" height="auto"><br>
<br>
<br>

In this experiment, the training process was stopped at epoch 92 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/asset/train_console_output_at_epoch_92.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/CCAUS</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for CCAUS.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/asset/evaluate_console_output_at_epoch_92.png" width="720" height="auto">
<br><br>Image-Segmentation-CCAUS

<a href="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this CCAUS/test was low, and dice_coef high as shown below.
<br>
<pre>
loss,0.0257
dice_coef,0.9583
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/CCAUS</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for CCAUS.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images (709x749 pixels)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks (709x749 pixels)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 709x749 pixels</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/images/202201121748100022VAS_slice_1165.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/masks/202201121748100022VAS_slice_1165.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test_output/202201121748100022VAS_slice_1165.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/images/202201121748100022VAS_slice_1450.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/masks/202201121748100022VAS_slice_1450.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test_output/202201121748100022VAS_slice_1450.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/images/202201121748100022VAS_slice_2689.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/masks/202201121748100022VAS_slice_2689.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test_output/202201121748100022VAS_slice_2689.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/images/202201121806070027VAS_slice_24.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/masks/202201121806070027VAS_slice_24.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test_output/202201121806070027VAS_slice_24.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/images/202201121815070032VAS_slice_1362.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/masks/202201121815070032VAS_slice_1362.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test_output/202201121815070032VAS_slice_1362.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/images/202201121819140033VAS_slice_655.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/masks/202201121819140033VAS_slice_655.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test_output/202201121819140033VAS_slice_655.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/images/202201121837530038VAS_slice_269.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test/masks/202201121837530038VAS_slice_269.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CCAUS/mini_test_output/202201121837530038VAS_slice_269.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>


<h3>
References
</h3>
<b>1. Ultrasound Common Carotid Artery Segmentation Based on Active Shape Model</b><br>
Xin Yang, Jiaoying Jin, Mengling Xu, Huihui Wu, Wanji He, Ming Yuchi, Mingyue Ding<br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3606761/">https://pmc.ncbi.nlm.nih.gov/articles/PMC3606761/</a>
<br>
<br>

<b>2. Automated Segmentation of Common Carotid Artery in Ultrasound Images</b><br>
J. H. Gagan; Harshit S. Shirsat; Grissel P. Mathias; B. Vaibhav Mallya; Jasbon Andrade; K. V. Rajagopal
<br>
Published in: IEEE Access ( Volume: 10)<br>
<a href="https://ieeexplore.ieee.org/document/9785785">https://ieeexplore.ieee.org/document/9785785</a>
<br>
<br>
<b>3. Method for Carotid Artery 3-D Ultrasound Image Segmentation Based on CSWin Transformer</b><br>
Yanping Lin, Jianhua Huang, Wangjie Xu, Cancan Cui, Wenzhe Xu, Zhaojun Li 
<br>
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0301562922006342">
https://www.sciencedirect.com/science/article/abs/pii/S0301562922006342</a>
<br>
<br>

<b>4. Common Carotid Artery Ultrasound</b><br>
<a href="https://www.cuh.nhs.uk/patient-information/ultrasound-scan-of-your-carotid-arteries/">
https://www.cuh.nhs.uk/patient-information/ultrasound-scan-of-your-carotid-arteries/</a>
<br>
<br>

