========================================================================================
=== Vimeo Interpolation Testing Set
========================================================================================

This is part of the Vimeo-90K dataset. It contains 3783 triplets temporal frame interpolation, and it is used for evaluation (not for training). All videos are downloaded from vimeo.com.

========================================================================================
=== Folder structure
========================================================================================

- input: This folder stores all input frames. It uses a two-level folder structure, where each folder "%05d/%04d" contains the first and the third frame in each sequence: im1.png and im3.png.

- target: This folder stores all target frames. It uses the same two-level folder structure as "input", where each folder "%05d/%04d" contains the second frame to be interpolated: im2.png.

- sep_testlist.txt: contains the list of sequences in this testing set.

======================================================
=== Citation
======================================================

If you use this dataset in your work, please cite the following work:

@article{xue17toflow,
  author = {Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and Freeman, William T},
  title = {Video Enhancement with Task-Oriented Flow},
  journal = {arXiv},
  year = {2017}
}

For questions, please contact Tianfan Xue (tianfan.xue@gmail.com), Baian Chen(baian@mit.edu), Jiajun Wu (jiajunwu@mit.edu), or Donglai Wei(donglai@csail.mit.edu).

For more information, please refers to our project website and github repo:

Project website: http://toflow.csail.mit.edu/
Github repo: https://github.com/anchen1011/toflow

======================================================
=== Disclaimer
======================================================

This dataset is for non-commercial usage only.
