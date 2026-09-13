# Augmentation steps and execution files 
* first convert the lmdb file of basic handwritten data using <mark>lmdb_to_img.py</mark> change the necessary paths to save the images and gt_file at required destination path. 
* then using <mark>augment_bpr.py</mark> create blur, pad, rotate augmented images and store it in a folder. 
This file creates all the images and stores in a same folder with different names for each so that it is easy to convert them to one lmdb file for further training. 
* use <mark>create_dataset_lmdb_lalitha.py</mark> to create a lmdb file using the images and gt_file created above. 