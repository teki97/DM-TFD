# A Data-Driven High-Resolution Time-frequency Distribution
![](https://github.com/teki97/DM-TFD/blob/master/fig/architecture.png)
We provide a pytorch implementation of the paper: A Data-Driven High-Resolution Time-frequency Distribution (DH-TFD) [1], where a data-driven modelig based time-frequency distribution model is proposed to gain high resolution and cross-term (CT) free TFDs. As shown in the above figure, the proposed model includes **N** Skipping Weighted Conv Modules. Specifically, several stacked multi-channel convolutional kernels to simulate traditional kernel functions while a skipping operator is utilized to maintain correct information transmission. In addition, bottleneck attention module (BAM) [2, 3] with groupnormalization is regarded as the weighted block to refine the coarse features extracted by convolutional layers to improve performance.  
All pre-trained networks related to this paper are provided in **master** branch.

## Preparation
- python 3.6
- pytorch 0.4.1
- cuda 9.0
- cudNN 7.6.3.30

## Training Dataset
Our training dataset includes 80000 two or three randomly spectrally-overlapped (only one intersection) linear frequencymodulated (LFM) and sinusoidal frequency-modulated (SFM) components with amplitude modulation (AM) at a fixed SNR = **10 dB**. In particular, two-component synthetic signal is composed of an AM-LFM and an AM-SFM, and three-component synthetic signal is composed of two AM-LFMs and an AM-SFM with only one intersection. All synthetic data have 256 samples. Each epoch randomly generates batchsize=16 synthetic signals for training by changing the slope of the AM-LFM signal and the smallest and normalized frequencies of the AM-SFM signal. The proportion for two and three-component signals is 50% and 50%. Test synthetic signals in this paper are not in our training dataset.  
For the specific application, you'd better make the synthetic signals in the training dataset fit to the real-life signals to get a satisfactory performance.

## Supplementary Experiments

### Discussion on the real-life signal
In this paper, we discuss the robustness of our network on synthetic data. We also have some discussion on the real-life signal corresponding to various **N** (3, 5, 7, 9, 11, 13, 15). Seven pre-trained networks are provided.
The evaluation results measured by Renyi Entropy for the real-life bat echolocation signal are shown in the following table: 
<table>
<tr>
  <td align="left">SNR</td>
  <td align="center">N=3</td>
  <td align="center">N=5</td>
  <td align="center">N=7</td>
  <td align="center">N=9</td>
  <td align="center">N=11</td>
  <td align="center">N=13</td>
  <td align="center">N=15</td>
</tr>
<tr>
  <td align="left">45 dB</td>
  <td align="center">10.67</td>
  <td align="center">10.00</td>
  <td align="center">9.64</td>
  <td align="center">9.60</td>
  <td align="center">9.60</td>
  <td align="center">9.86</td>
  <td align="center">9.74</td>
</tr>
<tr>
  <td align="left">35 dB</td>
  <td align="center">10.67</td>
  <td align="center">10.00</td>
  <td align="center">9.64</td>
  <td align="center">9.60</td>
  <td align="center">9.60</td>
  <td align="center">9.86</td>
  <td align="center">9.74</td>
</tr>
<tr>
  <td align="left">25 dB</td>
  <td align="center">10.64</td>
  <td align="center">10.00</td>
  <td align="center">9.64</td>
  <td align="center">9.60</td>
  <td align="center">9.60</td>
  <td align="center">9.87</td>
  <td align="center">9.75</td>
</tr>
<tr>
  <td align="left">15 dB</td>
  <td align="center">10.50</td>
  <td align="center">10.02</td>
  <td align="center">9.67</td>
  <td align="center">9.66</td>
  <td align="center">9.66</td>
  <td align="center">9.92</td>
  <td align="center">9.78</td>
</tr>
<tr>
  <td align="left">5 dB</td>
  <td align="center">10.48</td>
  <td align="center">10.27</td>
  <td align="center">9.96</td>
  <td align="center">10.17</td>
  <td align="center">10.11</td>
  <td align="center">10.21</td>
  <td align="center">10.10</td>
</tr>
<tr>
  <td align="left">0 dB</td>
  <td align="center">11.11</td>
  <td align="center">10.77</td>
  <td align="center">10.47</td>
  <td align="center">10.90</td>
  <td align="center">10.72</td>
  <td align="center">10.75</td>
  <td align="center">10.55</td>
</tr>
</table>
It is noted that the network with N=11 has the best performance on the real-life data, which is different from the result on the synthetic data. The reason behind this issue is that overfitting gives rise to while increasing N. Thus, for reducing parameters and obtaining great performance, we choose to set N=9.
The visualized experimental results are supplemented as follows:  

![](https://github.com/teki97/DM-TFD/blob/master/fig/supplement.png)

### Discussion on the SNR level during training
We choose to use the synthetic data with a fixed **SNR = 10 dB** to train our model. In fact, we have also trained our model using datasets with different SNR levels, e.g., SNR = 5dB, SNR = [5, 45] dB. The TFD results of the real-world signals are shown as follows:

<img src="https://github.com/teki97/DM-TFD/blob/master/fig/snr.png" width = "500" height = "300" align=center />

It is obvious that the model trained by data with SNR = 5 dB ignores the fourth component of the bat echolocation signal, of which energy is weak. The other two models succeed in obtaining the weak energy component. However, when we use the synthetic data with SNR level ranging from 5 to 45 dB, there are considerable CTs remaining. Thus, we choose data with SNR = 10 dB to train our model. 

### Discussion on the length of the test signal
It is well-known that the convolutional layers are in theory independent of signal length, and the linear layers are not. However, before the linear layers in the BAM, the average pooling is employed, i.e., only the number of channels is considered in the following linear layers (the number of channels is independent of signal length).

Though we train our model only using 256-sample synthetic signals, we gain satisfactory performance on a 400-sample bat echolocation signal without re-training. Thus, we have experiments on the different lengths of the test signals, and the results are shown as follows:

<img src="https://github.com/teki97/DM-TFD/blob/master/fig/length.png" width = "900" height = "175" align=center />

It is notable that the CTs and noise appear with the increasing length of signal, and when the test signals are nearly twice as long as the training signal, the great representations can be also gained. That is to say, only if the test signal is **more than twice** as long as the training signal, we need to re-train the model to gain better performance.

### Discussion on the training target
Supervised learning needs labeled datasets to train algorithms that to classify data accurately. The input of our model is the WVD of the signal, which is the 2-dimensional discrete time-frequency image. Thus we need to map the IF values into an image. For instance, the linear frequency-modulated signal $z(n)$ is defined as:
$$
z(n) = \exp \left[nf_{init} + \frac{\left(f_{final} - f_{init}\right)n^{2}}{2(L-1)}\right], \, n=0,1,2,\cdots, L-1,
$$
where $f_{init}$ is the initial normalized frequency, $f_{final}$ is the final   normalized frequency, $f_{init}, f_{final} \in [-0.5, 0.5]$, and $L$ is the FFT size. Therefore, the IF values can be written as:
$$
{IF}(n) = f_{init} + \frac{\left(f_{final} - f_{init}\right)n}{L-1}, 
$$
and we map the IF values into an image as follows:
$$
IF_{index}(n) = \min \big(L-1, \text{round}\left({IF}(n) \times 2L\right)\big),
$$
where $ IF_{index} $ is a corresponding index set of IF values in the image. It is inevitable to introduce errors in the training target since values are rounded (just like the quadrature error). The errors can be written as:
$$
\text{errors}(n) = \frac{IF_{index}(n)}{2L}- {IF}(n),
$$
it is obvious that $\text{errors}(n) \in [-\frac{1}{4L}, \frac{1}{4L}]$, i.e., when $L=256$ , the maximum error is about $0.000977$. Therefore the errors are too small to have an influence on the training target images. 

Due to the focus of our paper is to enhance the representation of the signal instead of IF estimation, we still choose to generate the training target with rounded IF values. 

On the other hand, the errors will be reduced with the increasing FFT size $L$. Thus, we also have provided the numerical results using the same model trained with 256-sample signal (no re-training) on test signals with increasing $L$ (assuming that the number of time bins are the same as the frequency ones). As shown in the following figure, it can be seen that the errors in the 'models' are gradually decreased. The TFD results of our proposed method are shown in the second row, and it is noted that the errors resulting from the rounded IF values are also reduced. Additionally, the interference terms and noise begin to appear for $L=768$. It implies that when the FFT size is greater than twice times that of training signals, we need to re-train the model using training data with appropriate FFT size to gain better performance.

<img src="https://github.com/teki97/DM-TFD/blob/master/fig/target.png" width = "900" height = "350" align=center />

### Discussion on loss function and evaluation criteria
The reason behind why binary cross-entropy (BCE) loss is used to train our model is that we regard the proposed model as a  classification task. Specifically, we divide the whole TF image into two parts, i.e., auto-terms (ATs) and others(cross-terms (CTs), interference terms and noise). Therefore, we choose BCE loss to train our model instead of MSE or MAE loss (which is usually used in the regression task).

The first quality measure, $\ell_{1}$ distance to model, is computed by:
$$
d\big(\rho(t, f)\big) = \frac{\sum_{t=0}^{L-1}\sum_{f=0}^{L-1}|{\rho(t, f)} - y(t, f)|}{\sum_{t=0}^{L-1}\sum_{f=0}^{L-1}| y(t, f)|},
$$
where ${\rho}(t, f)$ is the constructed TFD result and $y(t, f)$ is the ideal TFD, i.e., the model.
We train our model only using synthetic signals, and the instantaneous amplitudes (IAs)  as well as IFs of synthetic signals can be obtained. This measure used by previous works, is appropriate since it represents the distance between the generated TFD results and the target TFD model that we want to achieve.

The second quality measure, R\'enyi entropy, is defined as:
$$
R_{\alpha}\big(\rho(t, f)\big)=\frac{1}{1-\alpha} \log_{2} \displaystyle{\iint\left(\frac{\rho(t, f)}{\iint \rho(u, v) \mathrm{d} u \mathrm{d} v}\right)^{\alpha} \mathrm{d} t \mathrm{d} f}
$$
where $\alpha=3$. This measure used by previous works, is proposed to measure the amount of information within the TF plane. The parameter $\alpha$ is chosen as an odd integer value in order to cancel the cross-terms which are integrated out over the entire TF plane. Therefore, when $\alpha=3$, the smaller R\'enyi entropy is, the higher resolution the TFD result has.

### Discussion on the bottleneck structure in the BAM
The first linear layer $W_1\in \mathbb{R}^{C\times C/r_{c}}$ achieves channel reduction with reduction ratio $r_c$, then non-linearity characteristic is introduced by ReLU function, and the second linear layer $W_2\in \mathbb{R}^{C/r_{c} \times C}$ increases the number of channels. Namely, the linear layers in the BAM act as a bottleneck.

### Visualization results
From the perspective of visualization, the TFD result of the convolutional layer has little cross-terms and lower resolution than the result of BAM. Moreover, the TFD results of the BAM can see the fourth part of the signal while the output of the convolutional layer ignores. Therefore, we had the conclusion that the convolutional layers with skipping operators can offer a coarse CT reduction while the weighted block can eliminate residual CTs and the TFD resolution is further improved.

<img src="https://github.com/teki97/DM-TFD/blob/master/fig/visual.png" width = "600" height = "175" align=center />


### Comparison on the ability to estimate instantaneous frequency
Usually time-frequency representations are compared in terms of their ability to accurately estimate instantaneous frequency, thus we have added such a comparison with ADTFD, and RS, and th results are shown in the following:

<img src="https://github.com/teki97/DM-TFD/blob/master/fig/if_es.png" align=center />

For the closely-located signal, there is almost no difference among the results of three methods. On the other hand, it can be seen that the proposed DH-TFD has better performance on the spectral-overlapped signal, especially on the intersection of the signal.

### Discussion on the parameter settings
We have some experiments on the parameter settings in the proposed model, e.g., the kernel size **K1** in the skipping Conv block, the number of filters **F** in the skipping Conv block, the kernel size **K2** in the BAM, the number of reduction ratio **R1** in the channel attention of the BAM, and the number of reduction ratio **R2** in the spatial attention of the BAM.

We adopt **K1 = 5** in the skipping Conv block. Empirically, the ideal range of the kernel size ranges from 1 to 7. There are the experimental results about K1 = 3, 5, 7 in the following:

<img src="https://github.com/teki97/DM-TFD/blob/master/fig/k1.png" width = "500" height = "175" align=center />

On the one hand, there are residual CTs when K1 = 3 and K1 = 7, and the result of K1 = 5 achieves the cross-term free TFD. On the other hand, it seems that the large kernel size contributs to smooth result, e.g., though the result of K1 = 7 remains some CTs, the ATs and CTs in this TFD result look more smooth than K1 = 3.

We adopt **F = 8** in the skipping Conv block. Due to the hardware limitation, we only provide corresponding experiments with **F** ranging from 2 to 12. There are the TFD results about F = 2, 4, 6, 8, 10, 12 in the following:

<img src="https://github.com/teki97/DM-TFD/blob/master/fig/filter.png" width = "1080" height = "175" align=center />

It is interesting that more is not always better, that is, the model with F = 12 performs more poorly than the model with F = 2. The reason is that overfitting may appear when the number of filter is a large value. Thus, we choose F = 8 to achieve satisfactory performance. 

We adopt **K2 = 3** in the BAM. Taking the computation complexity into consideration, the ideal range of the kernel size ranges from 1 to 5. There are the experimental results on the synthetic signal with K2 = 1, 3, 5 in the following:

<img src="https://github.com/teki97/DM-TFD/blob/master/fig/k2.png" width = "500" height = "175" align=center />

There is almost no difference among these results. Except for the TFD result with K2 = 1 has lower resolution compared with other results and some residual CTs. In order to reduce the number of paramters, we finally choose K2 = 3.

We adopt **R1 = 4** and **R2 = 4** in the BAM. The channel number of input is set to 8, thus the ideal range of R1/R2 ranges from 1 to 8. The TFD results of the real-life signal with R1 = 1, 2, 4 and R2 = 1, 2, 4 are shown as follows:

<img src="https://github.com/teki97/DM-TFD/blob/master/fig/r1.png" width = "500" height = "175" align=center />

<img src="https://github.com/teki97/DM-TFD/blob/master/fig/r2.png" width = "500" height = "175" align=center />

For the selection of R1, it can be seen that the TFD results with R1 = 2, 4 reduce the CTs heavily while the result with R1 = 1 remains a lot of CTs. Moreover, the result of R1 = 4 has better performance on the resolution. For the selection of R2, only R2 = 4 achieves cross-term free TFD. Thus we choose R1 = 4 and R2 = 4 in our model.




## Contributing Guideline
We would like to thank the authors in these works [2-5] for sharing the source codes of these methods, which are publicly released at https://github.com/Prof-Boualem-Boashash/TFSAP-7.1-software-package, https://github.com/mokhtarmohammadi/Locally-Optimized-ADTFD and https://github.com/Jongchan/attention-module.
## Reference
[1] Jiang, Lei, et al. "Kernel Learning for High-Resolution Time-Frequency Distribution." arXiv preprint arXiv:2007.00322 (2020).  
[2] Park, Jongchan, et al. "Bam: Bottleneck attention module." arXiv preprint arXiv:1807.06514 (2018).  
[3] Park, Jongchan, et al. "A Simple and Light-Weight Attention Module for Convolutional Neural Networks." Int J Comput Vis 128, 783–798 (2020).  
[4] Boashash, Boualem, and Samir Ouelha. "Designing high-resolution time–frequency and time–scale distributions for the analysis and classification of non-stationary signals: a tutorial review with a comparison of features performance." Digital Signal Processing 77 (2018): 120-152.  
[5] Mohammadi, Mokhtar, et al. "Locally optimized adaptive directional time–frequency distributions." Circuits, Systems, and Signal Processing 37.8 (2018): 3154-3174.  
## Contact
This repo is currently maintained by Lei Jiang (teki97@whu.edu.cn).
