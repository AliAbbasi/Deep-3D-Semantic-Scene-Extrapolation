Requirements:

    Tensorflow >= 1.2.0
    python 2.7  
    
Clone all the files and directories to your machine,

To run the models on SUNCG and NYU datasets do:

    python  cnn_se.py|gan_se.py|sscnet_se.py  SUN|NYU 

For example, `python gan_se.py NYU`.

These models will create ".ply" files in their related directory (`cnn_se_directory, gan_se_directory, sscnet_se_directory`). Use any ply viewer to see the created scenes. 

Files with name included "NYU" show the scenes from NYU real scenes dataset and others show scenes from SUNCG synthetic dataset. The printed 'A1' and 'A2' values show the accuracy and the completeness measures respectively.

These models are tested on Ubuntu 16.04.3 LTS with GeForce GTX 1080 GPU which takes about 5 minutes for each model to generate the results.
    
