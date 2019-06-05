# Master Thesis - Pointcloud grad-CAM

This is the repository for the pointcloud based grad-CAM, which was originally inspired by the **Gradient Class Activation Mapping (Grad-CAM)[[1]]** approach. It was implemented by utlizing the **pointnet**[[2]] network.

The goal of this application is to yield a similar grade of transparency for pointcloud neuronal networks as the grad-CAM yields for 2D image convolutional neuronal networks (CNN). Usually the user is granted no deeper insight as of why a current input yields that particular results proposed by the network. This is particularly an issue if the output does not yield the expected result. In order to increase the confidence and trust in the performance of neuronal networks it is of virtue to be able to understand the reasoning behind the networks decisions.

We seek to achieve this transparency by drawing a heatmap of important points from the input pointcloud in order to be able to identify the areas of interest for the underlying neuronal network. This should yield a visual representation of the important areas that were responsible for the predicion result of the network. The pointcloud data is then visualized using the Open3D[[4]] library.

![Test_1](https://user-images.githubusercontent.com/19975052/58369593-e88e2980-7efc-11e9-91a1-8f2d6b372f58.PNG)

## Installation

To run the the code Python 3.X.X is required. Our code also requires the Open3D[[4]] library as well. You can install it with Anaconda (recommended) or pip with the following commands:

`pip3 install open3d`

Or

`conda install -c open3d-admin open3d`

## References

**[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization][1]**  
Ramprasaath R. Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi Parikh, Dhruv Batra  
[https://arxiv.org/abs/1610.02391][1]


**[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation][2]**  
Charles R. Qi*, Hao Su*, Kaichun Mo*, Leonidas J. Guibas  
[https://arxiv.org/abs/1610.02391][2]

## 3rd Party dependancies

- [pointnet][3]
- [open3d][4]

[1]: https://arxiv.org/abs/1610.02391
[2]: https://arxiv.org/pdf/1612.00593
[3]: https://github.com/charlesq34/pointnet
[4]: http://open3d.org/docs/index.html
