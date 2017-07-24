Semantic Image Inpainting With Deep Generative Models
=====================================================
[[Project]](http://www.isle.illinois.edu/~yeh17/projects/semantic_inpaint/index.html)
[[arXiv]](https://arxiv.org/abs/1607.07539)

Tensorflow implementation for semantic image inpainting:

![](http://www.isle.illinois.edu/~yeh17/projects/semantic_inpaint/img/process.png)

Semantic Image Inpainting With Deep Generative Models

[Raymond A. Yeh*](http://www.isle.illinois.edu/~yeh17/),
[Chen Chen*](http://cchen156.web.engr.illinois.edu/),
[Teck Yian Lim](http://tlim11.web.engr.illinois.edu/),
[Alexander G. Schwing](http://www.alexander-schwing.de/),
[Mark Hasegawa-Johnson](http://www.ifp.illinois.edu/~hasegawa/),
[Minh N. Do](http://minhdo.ece.illinois.edu/)

In CVPR 2017

\* indicating equal contributions.

Overview
--------
Implementation of proposed cost function and backpropogation to input. 

In this code release, we load a pretrained DCGAN model, and apply our proposed
objective function for the task of image completion

Dependencies
------------
 - Tensorflow >= 1.0
 - scipy + PIL/pillow (image io)
 - pyamg (for Poisson blending)

Tested to work with both Python 2.7 and Python 3.5


Files
-----
 - src/model.py - Main implementation
 - src/inpaint.py - command line application
 - src/external - external code used. Citations in code
 - graphs/dcgan-100.pb - frozen pretrained DCGAN with 100-dimension latent space
 
Weights
-------

Git doesn't work nicely with large binary files. Please download our weights from 
[here](https://www.dropbox.com/s/3uo97fzu4jfi2ms/dcgan-100.pb?dl=0), trained on the 
[CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

Alternatively, train your own GAN using your dataset. Conversion from checkpoint to 
Tensorflow ProtoBuf format can be done with 
[this script](https://gist.github.com/moodoki/e37a85fb0258b045c005ca3db9cbc7f6)


Running
-------

```
python src/inpaint.py --model_file graphs/dcgan-100.pb \
    --maskType center --in_image testimages/face1.png \
    --nIter 1000 --blend
```

Citation
--------

~~~
@inproceedings{
    yeh2017semantic,
    title={Semantic Image Inpainting with Deep Generative Models},
    author={Yeh$^\ast$, Raymond A. and Chen$^\ast$, Chen and Lim, Teck Yian and Schwing Alexander G. and Hasegawa-Johnson, Mark and Do, Minh N.},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    year={2017},
    note = {$^\ast$ equal contribution},
}
~~~

