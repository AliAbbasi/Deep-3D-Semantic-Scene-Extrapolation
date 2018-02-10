
Requirements:
    Tensorflow >= 1.2.0
    python 2.7
    
Clone all the files and directories to your machine,

To run the CNN model do:
    python cnn_se.py
    
To run the GAN model do:
    python gan_se.py
    
These models after running will create ".ply" files in the "cnn_se_directory" and "gan_se_directory" respectively. Files with name included "NYU" show the scens from NYU real scenes dataset and others show scenes from SUNCG synthetic scenes dataset.

These models are tested on Ubuntu 16.04.3 LTS with GeForce GTX 1080 GPU which takes about 10 miniutes to generate the results.
    
