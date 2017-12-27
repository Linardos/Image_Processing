# Image_Processing
Final Project for the Course in Image Processing

usage: ColorClassification.py [-h] [-C color_margin color_margin color_margin]
                              [-k1 Open Open Open] [-k2 Close Close Close]
                              [-cells]
                              image

Object Classification based on Color

positional arguments:
  image                 path to image

optional arguments:
  
  -h, --help            show this help message and exit
  
  -C color_margin color_margin color_margin
                        Input 3 numbers, one for each RGB color. These will be
                        used for thresholding color, Green Blue and Red. By
                        increasing each value you keep pixels with higher
                        intensity to the respective color, default = [0, 0, 0]
  
  -k1 Open Open Open    Specifies the size of kernel 1 for each color, used
                        for denoising (opening), (default: [0, 0, 0])
  
  -k2 Close Close Close
                        Specifies the size of kernel 2 for each color, used
                        for denoising (closing), (default: [0, 0, 0])
  
  -cells                Set to an alternate mode for recognizing distinct
                        cells and their nuclei (default: False)
              
