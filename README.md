# Data-Driven Modeling of High-Resolution Time-frequency Distribution
![](https://github.com/teki97/DM-TFD/blob/master/fig/architecture.png)
We provide a pytorch implementation of the paper: Data-Driven Modeling of High-Resolution Time-frequency Distribution [1], where a data-driven modelig based time-frequency distribution (DM-TFD) model is proposed to gain high resolution and cross-term (CT) free TFDs. As shown in the above figure, the proposed model includes **N** Skipping Weighted Conv Modules. Specifically, several stacked multi-channel learning convolutional kernels to simulate traditional kernel functions while a skipping operator is utilized to maintain correct information transmission. In addition, bottleneck attention module (BAM) [2, 3] with groupnormalization is regarded as the weighted block to refine the coarse features extracted by convolutional layers to improve performance.  
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

It is obvious that the model trained by data with SNR = 5 dB ignores the fourth component of the bat echolocation signal, of which energy is weak. The other two model succeed in obtaining the weak energy component. However, when we use the synthetic data with SNR level ranging from 5 to 45 dB, there are considerable CTs remaining. Thus, we choose data with SNR = 10 dB to train our model. 

### Discussion on the length of the test signal
Though we train our model only using 256-sample synthetic signals, we gain satisfactory performance on a 400-sample bat echolocation signal without re-training. Thus, we have experiments on the different lengths of the test signals, and the results are shown as follows:

<img src="https://github.com/teki97/DM-TFD/blob/master/fig/length.png" width = "900" height = "175" align=center />

It is notable that the interference terms and noise appear with the increasing length of signal, and when the test signal is twice longer than the training signal, the great representation can be also gained. That is to say, only if the length of the test signal is nearly **twice** as long as the training signal, we need to re-train the model to gain better performance.

### Comparison on the ability to estimate instantaneous frequency
Usually time-frequency representations are compared in terms of their ability to accurately estimate instantaneous frequency, thus we have added such a comparison with ADTFD, RS and SST, and th results are shown in the following:

<img src="https://github.com/teki97/DM-TFD/blob/master/fig/if.png" width = "800" height = "450" align=center />

For the closely-located signal, there is almost no difference among the results of three methods. On the other hand, it can be seen that the proposed DM-TFD has better performance on the spectral-overlapped signal, especially on the intersection of the signal.

### Discussion on the parameter settings
We have some experiments on the parameter settings in the proposed model, e.g., the kernel size **K1** in the skipping Conv block, the kernel size **K2** in the BAM, the number of reduction ratio **R1** in the channel attention of the BAM, and the number of reduction ratio **R2** in the spatial attention of the BAM.

We adopt **K1 = 5** in the skipping Conv block. Empirically, the ideal range of the kernel size ranges from 1 to 7. There are the experimental results about K1 = 3, 5, 7 in the following:

<img src="https://github.com/teki97/DM-TFD/blob/master/fig/k1.png" width = "500" height = "175" align=center />

On the one hand, there are residual CTs when K1 = 3 and K1 = 7, and the result of K1 = 5 achieves the cross-term free TFD. On the other hand, it seems that the large kernel size contributs to smooth result, e.g., though the result of K1 = 7 remains some CTs, the ATs and CTs in this TFD result look more smooth than K1 = 3.

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
