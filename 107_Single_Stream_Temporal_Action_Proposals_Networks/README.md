# SST: Single-Stream Temporal Action Proposals


Welcome to the official repo for [SST: Single-Stream Temporal Action Proposals](http://vision.stanford.edu/pdf/buch2017cvpr.pdf)!

**SST** is an efficient model for generating temporal action proposals in untrimmed videos. Analogous to object proposals for *images*, temporal action proposals provide the temporal bounds in *videos* where potential actions of interest may lie.

<div class="centered">
<a href="http://vision.stanford.edu/pdf/buch2017cvpr.pdf" target="_blank"_>
<img src="https://dl.dropboxusercontent.com/s/pv2mrc0ps09zqu3/sst_modelfig.png" width="590" alt="SST model overview" />
</div>
<br/>
</a>

### Resources

Quick links:
[[cvpr paper](http://vision.stanford.edu/pdf/buch2017cvpr.pdf)]
[[poster](https://drive.google.com/file/d/0B_-dKvCH2VL7WG01Wjh4TEdZSjQ/view?usp=sharing)]
[[supplementary](https://drive.google.com/file/d/0B_-dKvCH2VL7dGV1ankxWnJVQmM/view?usp=sharing)]
[[code](https://github.com/shyamal-b/sst/)]
<!-- [[video](https://drive.google.com/file/d/0B_-dKvCH2VL7dGV1ankxWnJVQmM/view?usp=sharing)] -->

**Update:** if you find this work useful, you may *also* find our newer work of interest:  [link to SS-TAD](https://github.com/shyamal-b/ss-tad/)

Please use the following bibtex to cite our work:

    @inproceedings{sst_buch_cvpr17,
      author = {Shyamal Buch and Victor Escorcia and Chuanqi Shen and Bernard Ghanem and Juan Carlos Niebles},
      title = {{SST}: Single-Stream Temporal Action Proposals},
      year = {2017},
      booktitle = {CVPR}
      }

As part of this repo, we also include *evaluation notebooks*, *SST proposals* for THUMOS'14, and *pre-trained model parameters*. Please see the `code/` and `data/` folders for more.

### Dependencies

We include a *requirements.txt* file that lists all the dependencies you need. Once you have created a [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/), simply run `pip install -r requirements.txt` from within the environment to install all the dependencies. Note that the original code was executed using Python 2.7.

<!-- (For Mac OSX users, you may need to run `pip install --ignore-installed numpy six`) -->
