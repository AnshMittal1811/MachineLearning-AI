A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval
===========================================================================

This is the official repository for the publication:

    @inproceedings{schoenberger2016vote,
        author = {Sch\"{o}nberger, Johannes Lutz and Price, True and Sattler, Torsten and Frahm, Jan-Michael and Pollefeys, Marc},
        title = {A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval},
        booktitle={Asian Conference on Computer Vision (ACCV)},
        year={2016},
    }


World5k Dataset
---------------

The URLs for all images in our dataset can be found in the `image_urls` folder.


Vote-and-Verify Code
--------------------

An implementation of both the vocabulary tree with Hamming embedding as well as
our proposed Vote-and-Verify method can be found in COLMAP
(https://github.com/colmap/colmap). COLMAP is a Structure-from-Motion and Multi-
View Stereo library. COLMAP implements a fully functional image retrieval system
(in the `src/retrieval/*` folder), that can be used with the executables:

- `src/exe/vocab_tree_builder`:
  to build a custom vocabulary tree from image features

- `src/exe/vocab_tree_retriever`:
  to perform image retrieval using a pre-built vocabulary tree

- `src/exe/vocab_tree_matcher`:
  to match images using the vocabulary tree

The number of images to re-rank during spatial verification can be specified
using the `num_verifications` option. Please refer to the code and the
documentation of COLMAP for more details and fine-grain control of the
parameters (https://colmap.github.io/).
