## Single Image Deblurring with Row-dependent Blur Magnitude
![visitors](https://visitor-badge.laobi.icu/badge?page_id=jixiang2016/RSS-T)  [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Ji_Single_Image_Deblurring_with_Row-dependent_Blur_Magnitude_ICCV_2023_paper.pdf) | [Supp](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Ji_Single_Image_Deblurring_ICCV_2023_supplemental.pdf) | [Dataset](https://drive.google.com/file/d/1l0GMiv2xMcVaSuIY4E7f3zPljtRq1mju/view)

Xiang Ji<sup>1</sup>, Zhixiang Wang<sup>1,2</sup>, Shin'ichi Satoh<sup>2,1</sup>, Yinqiang Zheng<sup>1</sup>
<sup>1</sup>The University of Tokyo&nbsp;&nbsp;<sup>2</sup>National Institute of Informatics&nbsp;&nbsp;


This repository provides the official PyTorch implementation of the paper.

#### TL;DR
This paper explores a novel in-between exposure mode called global reset release (GRR) shutter, which produces GS-like blur but with row-dependent blur magnitude. We take advantage of this unique characteristic of GRR to explore the latent frames within a single image and restore a clear counterpart by relying only on these latent contexts.

<img width="900" alt="image" src="docs/shutter_modes.png">
