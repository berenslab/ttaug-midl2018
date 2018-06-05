# ttaug-midl2018
Test-time Data Augmentation for Estimation of Heteroscedastic Aleatoric Uncertainty in Deep Neural Networks, MIDL 2018

We provide the codebase behind our paper titled 'Test-time Data Augmentation for Estimation of Heteroscedastic Aleatoric Uncertainty in Deep Neural Networks' and presented at the International conference on Medical Imaging with Deep Learning in 
Amsterdam on the 6th of July 2018. Additionally, a notebook to demonstrate the use of our ResNet50-like CNN is included. 

During our study, we explored and implemented several architectural ideas from the following papers:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In European Conference on Computer Vision, pages 630–645. Springer, 2016.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.  Delving deep into rectifiers:  Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pages 1026–1034, 2015.

While you can readily replicate our results, you may also explore several variations if you feel comfortable getting into to the code. 

* Dataset: We evaluate our method using the well-known collection of fundus images obtained from a previous Kaggle competition: https://www.kaggle.com/c/diabetic-retinopathy-detection . If you keep a similar directory structure as well as a file list, and process the images accordingly, you may even be able to use our reader object. 

# Compatibility
We tested our code with Tensorflow 1.4.1 and Tensorflow 1.8. 

# Help and discussions
If you have questions and/or things to discuss, feel free to contact us. If you find terrible things about our method/code, please, let us know. We also look forward to hearing what you do with the method/code. 


# Citing the paper
If you would like to cite our paper, please, consider something similar to the following entry:
```
@inproceedings{ayhan2018,
  title={Test-time Data Augmentation for Estimation of Heteroscedastic Aleatoric Uncertainty in Deep Neural Networks},
  author={Ayhan, Murat Seckin and Berens, Philipp},
  booktitle={International conference on Medical Imaging with Deep Learning},
  year={2018}
}
```
