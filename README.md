[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/) [![Python badge](https://img.shields.io/badge/Python-3.5-<COLOR>.svg)](https://shields.io/) [![Platform badge](https://img.shields.io/badge/Platform-Windows_10_x64-<COLOR>.svg)](https://shields.io/)
# Pointcloud grad-CAM

This is the repository for **pointcloud grad-CAM (p-grad-CAM)[[8]]**, which was originally inspired by the **Gradient Class Activation Mapping (Grad-CAM)[[1]]** approach. It was implemented utlizing the **Pointnet**[[2]] network and **Tensorflow**. For testing the validity of our approach it was compared to a similar algorithm, called **Saliency Maps**[[7]]. This algorithm was slightly rewritten to fit our testing environent and was renamed **ASM**, it is also included in this repository.

The goal of this application is to yield a similar grade of transparency for pointcloud neuronal networks as grad-CAM does for 2D image convolutional neuronal networks (CNN). Usually the user is granted no deeper insight as of why a current input yields a particular results delivered by the network. This is particularly an issue if the output does not yield the expected result. In order to increase the confidence and trust in the performance of neuronal networks it is of virtue to be able to understand the reasoning behind the networks decisions.

We seek to achieve this transparency by finding the salient regions of the input pointcloud in order to be able to identify the areas of interest for the underlying neuronal network. This should yield a visual representation of the important areas that were responsible for the predicion result of the network. The pointcloud data is then visualized using the Open3D[[4]] library.

![comparison_airplane_saliency_p-grad-cam](https://user-images.githubusercontent.com/19975052/82739509-6703ee00-9d40-11ea-931c-7d13957a1051.png)

## Installation and Usage

To run the the code Python 3.5[[5]] or better and Anaconda[[6]] is required. To use this algorithm follow the next steps:

1. Install Anaconda[[6]] for Python 3.5[[5]] on your system.

2. Open the Anaconda prompt and `cd` into the cloned repo directory.

3. Create the environment with all necessary dependencies with the following command:

`conda env create -f "..\Pointcloud-grad-CAM\Anaconda\p-grad-CAM.yml"`
    
4. Activate the virtual environment by typing the following command:

`conda activate p-grad-CAM`

5. Train the network:

`python train.py`

6. Run either p-grad-CAM or ASM on it:

`python p_grad_CAM.py` or `python ASM.py`

## References

**[Identifying salient regions using point cloud gradient Class Activation Mapping][8]**  
Dennis Struhs

**[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization][1]**  
Ramprasaath R. Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi Parikh, Dhruv Batra  
[https://arxiv.org/abs/1610.02391][1]


**[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation][2]**  
Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas  
[https://arxiv.org/abs/1610.02391][2]


**[Saliency Maps: Learning deep features for discriminative localization][7]**  
Tianhang Zheng, Changyou Chen, Junsong Yuan, Kui Ren  
[http://arxiv.org/abs/1812.01687][7]

## 3rd Party dependancies

- [Pointnet][3]
- [Open3D][4]
- [Python][5]
- [Anaconda][6]

[1]: https://arxiv.org/abs/1610.02391
[2]: https://arxiv.org/pdf/1612.00593
[3]: https://github.com/charlesq34/pointnet
[4]: http://open3d.org/docs/index.html
[5]: https://www.python.org/downloads/
[6]: https://www.anaconda.com/products/individual
[7]: http://arxiv.org/abs/1812.01687
[8]: https://github.com/Fragjacker/Pointcloud-grad-CAM/files/5371140/Master_Thesis_paper___p-grad-CAM.pdf
