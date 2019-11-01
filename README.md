# Image-Denoising-in-extreme-low-light
#### 1) Used a U-net on the Raw images as input
#### 2) Used the weighted average of ​ edge loss, PSNR loss and 1- SSIM​ as the loss function
#### 3) For the edge loss calculation a canny map was used
#### 4) The output was RGB denoised image
#### 5) The model is inspired from the paper 'Seeing in the dark ' https://arxiv.org/abs/1805.01934
#### 6) In comparison to the above paper we changed the loss function and also used canny map to calculate   edge loss
_______________________________________________________________________________________________________________________________

# MODEL
The dataset used contained images shot in low light with the exposure time of 1o miliseconds and 30 miliseconds and the denoised ground truth images were shot under same circumstances but with an exposure time of 30 seconds thus they captured light quite well and trhey were also not noisy. We got this dataset from the github repository of the original paper.First we calculated an amplification factor as the ratio of denoised ground truth image exposure time and corresponding noisy image exposure time and we multiplied the input noisy image by it. Input size is [4,256,256] and it is raw images. These are low light noisy images. These images are passed through a U-net which first downsamples them to [512,16,16] then the network upscales the image using convolutional and transpose convolutional layers and the output of every transpose convolutional layer is concatenated with the corresponding feature maps on the encoder(downsampling network) side. The final output is thus 3 RGB channels and then the loss is calculated and the model is trained end to end.

