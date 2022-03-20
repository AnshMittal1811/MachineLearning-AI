```
@misc{
pascal2021on,
title={On the use of linguistic similarities to improve Neural Machine Translation for African Languages},
author={Tikeng Notsawo Pascal and NANDA ASSOBJIO Brice Yvan and James Assiene},
year={2021},
url={https://openreview.net/forum?id=Q5ZxoD2LqcI}
}
```

## I. Cross-lingual language model pretraining ([XLM](https://github.com/facebookresearch/XLM)) 

XLM supports multi-GPU and multi-node training, and contains code for:
- **Language model pretraining**:
    - **Causal Language Model** (CLM)
    - **Masked Language Model** (MLM)
    - **Translation Language Model** (TLM)
- **GLUE** fine-tuning
- **XNLI** fine-tuning
- **Supervised / Unsupervised MT** training:
    - Denoising auto-encoder
    - Parallel data training
    - Online back-translation

#### Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/) (currently tested on version 0.4 and 1.0)
- [fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) (generate and apply BPE codes)
- [Moses](https://github.com/facebookresearch/XLM/tree/master/tools#tokenizers) (scripts to clean and tokenize text only - no installation required)
- [Apex](https://github.com/nvidia/apex#quick-start) (for fp16 training)

### Pretrained models  
<table class="table table-striped">
    <caption><b>Machine Translation BLEU scores. The rows correspond to the pairs of interest on which
BLEU scores are reported. The column None is a baseline : it represents the BLEU score of a
model trained on the pair without any MLM or TLM pre-training. The column Pair is a baseline :
it represents the BLEU score of a model trained on the pair with MLM and TLM pre-training. The
column Random is also a baseline : it is the BLEU score of a 3 languages multi-task model where
the language added was chosen purely at random. The column Historical refers to the BLEU score
of our 3 languages multi-task model where the language added was chosen using clusters historicaly identified. The column LM describes the BLEU score of our 3 languages, multi-task model where the
language added was chosen using the LM similarity</b></caption>
    <thead>
        <tr>
            <th scope="col">Pretraining</th>
            <th scope="col">None</th>
            <th scope="col">Pair</th>
            <th scope="col">Random</th>
            <th scope="col">Historical</th>
            <th scope="col">LM</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th scope="row">Bafia-Bulu</th>
            <td>09.19</td>
            <td>12.58</td>
            <td>23.52</td>
            <td><b>28.81</b></td>
            <td>13.03</td>
        </tr>
        <tr>
            <th scope="row">Bulu-Bafia</th>
            <td>13.50</td>
            <td>15.15</td>
            <td>24.76</td>
            <td><b>32.83</b></td>
            <td>13.91</td>
        </tr>
        <tr>
            <th scope="row">Bafia-Ewondo</th>
            <td>09.30</td>
            <td>11.28</td>
            <td>08.28</td>
            <td><b>38.90</b></td>
            <td><b>38.90</b></td>
        </tr>
        <tr>
            <th scope="row">Ewondo-Bafia</th>
            <td>13.99</td>
            <td>16.07</td>
            <td>10.26</td>
            <td><b>35.84</b></td>
            <td><b>35.84</b></td>
        </tr>
        <tr>
            <th scope="row">Bulu-Ewondo</th>
            <td>10.27</td>
            <td>12.11</td>
            <td>11.82</td>
            <td><b>39.12</b></td>
            <td>34.86</td>
        </tr>
        <tr>
            <th scope="row">Ewondo-Bulu</th>
            <td>11.62</td>
            <td>14.42</td>
            <td>12.27</td>
            <td><b>34.91</b></td>
            <td>30.98</td>
        </tr>
        <tr>
            <th scope="row">Guidar-Guiziga</th>
            <td>11.95</td>
            <td>15.05</td>
            <td>Random</td>
            <td>Historical</td>
            <td>LM</td>
        </tr>
        <tr>
            <th scope="row">Guiziga-Guidar</th>
            <td>08.05</td>
            <td>08.94</td>
            <td>Random</td>
            <td>Historical</td>
            <td>LM</td>
        </tr>
        <tr>
            <th scope="row">Guiziga-Mofa</th>
            <td>17.78</td>
            <td>21.67</td>
            <td>Random</td>
            <td>Historical</td>
            <td>LM</td>
        </tr>
        <tr>
            <th scope="row">Mofa-Guiziga</th>
            <td>12.02</td>
            <td>15.41</td>
            <td>Random</td>
            <td>Historical</td>
            <td>LM</td>
        </tr>
        <tr>
            <th scope="row">Guidar-Kapsiki</th>
            <td>14.74</td>
            <td>17.78</td>
            <td>Random</td>
            <td>Historical</td>
            <td>LM</td>
        </tr>
        <tr>
            <th scope="row">Kapsiki-Guidar</th>
            <td>08.63</td>
            <td>09.33</td>
            <td>Random</td>
            <td>Historical</td>
            <td>LM</td>
        </tr>
        <tr>
            <th scope="row">French-Bulu</th>
            <td>19.91</td>
            <td>23.47</td>
            <td>Random</td>
            <td><b>25.06</b></td>
            <td>LM</td>
        </tr>
        <tr>
            <th scope="row">Bulu-French</th>
            <td>17.49</td>
            <td>22.44</td>
            <td>Random</td>
            <td><b>23.68</b></td>
            <td>LM</td>
        </tr>
        <tr>
            <th scope="row">French-Bafia</th>
            <td>14.48</td>
            <td>15.35</td>
            <td>Random</td>
            <td><b>30.65</b></td>
            <td>LM</td>
        </tr>
        <tr>
            <th scope="row">Bafia-French</th>
            <td>08.59</td>
            <td>11.17</td>
            <td>Random</td>
            <td><b>24.49</b></td>
            <td>LM</td>
        </tr>
        <tr>
            <th scope="row">French-Ewondo</th>
            <td>11.51</td>
            <td>13.93</td>
            <td>Random</td>
            <td><b>35.50</b></td>
            <td>LM</td>
        </tr>
        <tr>
            <th scope="row">Ewondo-French</th>
            <td>10.60</td>
            <td>13.77</td>
            <td>Random</td>
            <td><b>27.34</b></td>
            <td>LM</td>
        </tr>
    </tbody>
</table>

## II. Model-Agnostic Meta-Learning ([MAML](https://arxiv.org/abs/1911.02116))  

See [maml](https://github.com/cbfinn/maml), [learn2learn](https://github.com/learnables/learn2learn)...  

See [HowToTrainYourMAMLPytorch](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch) for a replication of the paper ["How to train your MAML"](https://arxiv.org/abs/1810.09502), along with a replication of the original ["Model Agnostic Meta Learning"](https://arxiv.org/abs/1703.03400) (MAML) paper.

## III. Train your own (meta-)model

**Open the illustrative notebook in colab**[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tikquuss/meta_XLM/blob/master/notebooks/demo/tuto.ipynb)

**Note** : Most of the bash scripts used in this repository were written on the windows operating system, and can generate this [error](https://prograide.com/pregunta/5588/configure--bin--sh--m-mauvais-interpreteur) on linux platforms.  
This problem can be corrected with the following command: 
```
filename=my_file.sh 
cat $filename | tr -d '\r' > $filename.new && rm $filename && mv $filename.new $filename 
```
### 1. Preparing the data 

At this level, if you have pre-processed binary data in pth format (for example from XLM experimentation or improvised by yourself), group them in a specific folder that you will mention as a parameter by calling the script [train.py](XLM/train.py).  
If this is not the case, we assume that you have txt files available for preprocessing. Look at the following example for which we have three translation tasks: `English-French, German-English and German-French`.

We have the following files available for preprocessing: 
```
- en-fr.en.txt and en-fr.fr.txt 
- de-en.de.txt and de-en.en.txt 
- de-fr.de.txt and de-fr.fr.txt 
```
All these files must be in the same folder (`PARA_PATH`).  
You can also (only or optionally) have monolingual data available (`en.txt, de.txt and fr.txt`; in `MONO_PATH` folder).  
Parallel and monolingual data can all be in the same folder.

**Note** : Languages must be submitted in alphabetical order (`de-en and not en-de, fr-ru and not ru-fr...`). If you submit them in any order you will have problems loading data during training, because when you run the [train.py](XLM/train.py) script the parameters like the language pair are put back in alphabetical order before being processed. Don't worry about this alphabetical order restriction, XLM for MT is naturally trained to translate sentences in both directions. See [translate.py](scripts/translate.py).

[OPUS collections](http://opus.nlpl.eu/) is a good source of dataset. We illustrate in the [opus.sh](scripts/opus.sh) script how to download the data from opus and convert it to a text file.  
Changing parameters ($PARA_PATH and $SRC) in [opus.sh](scripts/opus.sh).
```
cd meta_XLM
chmod +x ./scripts/opus.sh
./scripts/opus.sh de-fr
```

Another source for `other_languages-english` data is [anki Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/). Simply download the .zip file, unzip to extract the `other_language.txt` file. This file usually contains data in the form of `sentence_en sentence_other_language other_information` on each line. See [anki.py](scripts/anki.py) and [anky.sh](scripts/anki.sh) in relation to a how to extract data from [anki](http://www.manythings.org/anki/). Example of how to download and extract `de-en` and `en-fr` pair data.
```
cd meta_XLM
output_path=/content/data/para
mkdir $output_path
chmod +x ./scripts/anki.sh
./scripts/anki.sh de,en deu-eng $output_path scripts/anki.py
./scripts/anki.sh en,fr fra-eng $output_path scripts/anki.py
```
After that you will have in `data/para` following files : `de-en.de.txt, de-en.en.txt, deu.txt, deu-eng.zip and _about.txt`  

Move to the `XLM` folder in advance.  
```
cd XLM
```
Install the following dependencies ([fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) and [Moses](https://github.com/facebookresearch/XLM/tree/master/tools#tokenizers)) if you have not already done so. 
```
git clone https://github.com/moses-smt/mosesdecoder tools/mosesdecoder
git clone https://github.com/glample/fastBPE tools/fastBPE && cd tools/fastBPE && g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
```
  
Changing parameters in [data.sh](data.sh). Between lines 94 and 100 of [data.sh](data.sh), you have two options corresponding to two scripts to execute according to the distribution of the folders containing your data. Option 2 is chosen by default, kindly uncomment the lines corresponding to your option.  
With too many BPE codes (depending on the size of the dataset) you may get this [error](https://github.com/glample/fastBPE/issues/7). Decrease the number of codes (e.g. you can dichotomously search for the appropriate/maximum number of codes that make the error disappear)

```
languages=de,en,fr
chmod +x ../data.sh 
../data.sh $languages
```

If you stop the execution when processing is being done on a file please delete this erroneous file before continuing or restarting the processing, otherwise the processing will continue with this erroneous file and potential errors will certainly occur.  

After this you will have the following (necessary) files in `$OUTPATH` (and `$OUTPATH/fine_tune` depending on the parameter `$sub_task`):  

```
- monolingual data :
    - training data   : train.fr.pth, train.en.pth and train.de.pth
    - test data       : test.fr.pth, test.en.pth and test.de.pth
    - validation data : valid.fr.pth, valid.en.pth and valid.de.pth
- parallel data :
    - training data : 
        - train.en-fr.en.pth and train.en-fr.fr.pth 
        - train.de-en.en.pth and train.de-en.de.pth
        - train.de-fr.de.pth and train.de-fr.fr.pth 
    - test data :
        - test.en-fr.en.pth and test.en-fr.fr.pth 
        - test.de-en.en.pth and test.de-en.de.pth
        - test.de-fr.de.pth and test.de-fr.fr.pth 
    - validation data
        - valid.en-fr.en.pth and valid.en-fr.fr.pth 
        - valid.de-en.en.pth and valid.de-en.de.pth
        - valid.de-fr.de.pth and valid.de-fr.fr.pth 
 - code and vocab
```
To use the biblical corpus, run [bible.sh](bible.sh) instead of [data.sh](data.sh). Here is the list of languages available (and to be specified as `$languages` value) in this case : 
- **Languages with data in the New and Old Testament** : `Francais, Anglais, Fulfulde_Adamaoua or Fulfulde_DC (formal name : Fulfulde), Bulu, KALATA_KO_SC_Gbaya or KALATA_KO_DC_Gbaya (formal name :  Gbaya), BIBALDA_TA_PELDETTA (formal name : MASSANA), Guiziga, Kapsiki_DC (formal name : Kapsiki), Tupurri`.
- **Languages with data in the New Testament only** : `Bafia, Ejagham, Ghomala, MKPAMAN_AMVOE_Ewondo (formal name : Ewondo), Ngiemboon, Dii, Vute, Limbum, Mofa, Mofu_Gudur, Doyayo, Guidar, Peere_Nt&Psalms, Samba_Leko, Du_na_sdik_na_wiini_Alaw`.  
It is specified in [bible.sh](bible.sh) that you must have in `csv_path` a folder named csvs. Here is the [drive link](https://drive.google.com/file/d/1NuSJ-NT_BsU1qopLu6avq6SzUEf6nVkk/view?usp=sharing) of its zipped version.  
Concerning training, specify the first four letters of each language (`Bafi` instead of `Bafia` for example), except `KALATA_KO_SC_Gbaya/KALATA_KO_DC_Gbaya which becomes Gbay (first letters of Gbaya), BIBALDA_TA_PELDETTA which becomes MASS (first letters of MASSANA), MKPAMAN_AMVOE_Ewondo which becomes Ewon (first letters of Ewondo), Francais and Anglais which becomes repectively fr and en`. Indeed, [bible.sh](bible.sh) uses these abbreviations to create the files and not the language names themselves.  
One last thing in the case of the biblical corpus is that when only one language is to be specified, it must be specified twice. For example: `languages=Bafia,Bafia` instead of `languages=Bafia`.

### 2. Pretrain a language (meta-)model 

Install the following dependencie ([Apex](https://github.com/nvidia/apex#quick-start)) if you have not already done so.
```
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```

Instead of passing all the parameters of train.py, put them in a json file and specify the path to this file in parameter (See [lm_template.json](configs/lm_template.json) file for more details).
```
config_file=../configs/lm_template.json
python train.py --config_file $config_file
```
If you pass a parameter by calling the script [train.py](XLM/train.py) (example: `python train.py --config_file $config_file --data_path my/data_path`), it will overwrite the one passed in `$config_file`.  
Once the training is finished you will see a file named `train.log` in the `$dump_path/$exp_name/$exp_id` folder information about the training. You will find in this same folder your checkpoints and best model.  
When `"mlm_steps":"..."`, train.py automatically uses the languages to have `"mlm_steps":"de,en,fr,de-en,de-fe,en-fr"` (give a precise value to mlm_steps if you don't want to do all MLM and TLM, example : `"mlm_steps":"en,fr,en-fr"`). This also applies to `"clm_steps":"..."` which deviates `"clm_steps":"de,en,fr"` in this case.    

Note :  
-`en` means MLM on `en`, and requires the following three files in `data_path`: `a.en.pth, a ∈ {train, test, valid} (monolingual data)`  
-`en-fr` means TLM on `en and fr`, and requires the following six files in `data_path`: `a.en-fr.b.pth, a ∈ {train, test, valid} and b ∈ {en, fr} (parallel data)`  
-`en,fr,en-fr` means MLM+TLM on `en, fr, en and fr`, and requires the following twelve files in `data_path`: `a.b.pth and a.en-fr.b.pth, a ∈ {train, test, valid} and b ∈ {en, fr}`  

To [train with multiple GPUs](https://github.com/facebookresearch/XLM#how-can-i-run-experiments-on-multiple-gpus) use:
```
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py --config_file $config_file
```

**Tips**: Even when the validation perplexity plateaus, keep training your model. The larger the batch size the better (so using multiple GPUs will improve performance). Tuning the learning rate (e.g. [0.0001, 0.0002]) should help.

In the case of <b>metalearning</b>, you just have to specify your meta-task separated by `|` in `lgs` and `objectives (clm_steps, mlm_steps, ae_steps, mt_steps, bt_steps and pc_steps)`.  
For example, if you only want to do metalearning (without doing XLM) in our case, you have to specify these parameters: `"lgs":"de-en|de-fr|en-fr"`, `"clm_steps":"...|...|..."` and/or `"mlm_steps":"...|...|..."`. These last two parameters, if specified as such, will become respectively `"clm_steps":"de,en|de,fr|en,fr"` and/or `"mlm_steps":"de,en,de-en|de,fr,de-fr|en,fr,en-fr"`.  
The passage of the three points follows the same logic as above. That is to say that if at the level of the meta-task `de-en`:  
	- we only want to do MLM (without TLM): `mlm_steps` becomes `"mlm_steps": "de,en|...|..."`  
	- we don't want to do anything: `mlm_steps` becomes `"mlm_steps":"|...|..."`.

It is also not allowed to specify a meta-task that has no objective. In our case, `"clm_steps":"...||..."` and/or `"mlm_steps":"...||..."` will generate an exception, in which case the meta-task `de-fr` (second task) has no objective.

If you want to do metalearning and XLM simultaneously : 
- `"lgs":"de-en-fr|de-en-fr|de-en-fr"` 
- Follow the same logic as described above for the other parameters.

###### Description of some essential parameters

```
## main parameters
exp_name                     # experiment name
exp_id                       # Experiment ID
dump_path                    # where to store the experiment (the model will be stored in $dump_path/$exp_name/$exp_id)

## data location / training objective
data_path                    # data location 
lgs                          # considered languages/meta-tasks
clm_steps                    # CLM objective
mlm_steps                    # MLM objective

## transformer parameters
emb_dim                      # embeddings / model dimension
n_layers                     # number of layers
n_heads                      # number of heads
dropout                      # dropout
attention_dropout            # attention dropout
gelu_activation              # GELU instead of ReLU

## optimization
batch_size                   # sequences per batch
bptt                         # sequences length
optimizer                    # optimizer
epoch_size                   # number of sentences per epoch
max_epoch                    # Maximum epoch size
validation_metrics           # validation metric (when to save the best model)
stopping_criterion           # end experiment if stopping criterion does not improve

## dataset
#### These three parameters will always be rounded to an integer number of batches, so don't be surprised if you see different values than the ones provided.
train_n_samples              # Just consider train_n_sample train data
valid_n_samples              # Just consider valid_n_sample validation data 
test_n_samples               # Just consider test_n_sample test data for
#### If you don't have enough RAM/GPU or swap memory, leave these three parameters to True, otherwise you may get an error like this when evaluating :
###### RuntimeError: copy_if failed to synchronize: cudaErrorAssert: device-side assert triggered
remove_long_sentences_train # remove long sentences in train dataset      
remove_long_sentences_valid # remove long sentences in valid dataset  
remove_long_sentences_test  # remove long sentences in test dataset  
```

###### There are other parameters that are not specified here (see [train.py](XLM/train.py))

### 3. Train a (unsupervised/supervised) MT from a pretrained meta-model 

See [mt_template.json](configs/mt_template.json) file for more details.
```
config_file=../configs/mt_template.json
python train.py --config_file $config_file
```

When the `ae_steps` and `bt_steps` objects alone are specified, this is unsupervised machine translation, and only requires monolingual data. If the parallel data is available, give `mt_step` a value based on the language pairs for which the data is available.  
All comments made above about parameter passing and <b>metalearning</b> remain valid here : if you want to exclude a meta-task in an objective, put a blank in its place. Suppose, in the case of <b>metalearning</b>, we want to exclude from `"ae_steps":"en,fr|en,de|de,fr"` the meta-task:
- `de-en` : `ae_steps`  becomes `"ae_steps":"en,fr||de,fr"` 
- `de-fr` : `ae_steps`  becomes `"ae_steps":"en,fr|de,en|"`  

###### Description of some essential parameters  
The description made above remains valid here
```
## main parameters
reload_model     # model to reload for encoder,decoder
## data location / training objective
ae_steps          # denoising auto-encoder training steps
bt_steps          # back-translation steps
mt_steps          # parallel training steps
word_shuffle      # noise for auto-encoding loss
word_dropout      # noise for auto-encoding loss
word_blank        # noise for auto-encoding loss
lambda_ae         # scheduling on the auto-encoding coefficient

## transformer parameters
encoder_only      # use a decoder for MT

## optimization
tokens_per_batch  # use batches with a fixed number of words
eval_bleu         # also evaluate the BLEU score
```
###### There are other parameters that are not specified here (see [train.py](XLM/train.py))


### 4. case of metalearning : optionally fine-tune the meta-model on a specific (sub) nmt (meta) task 

At this point, if your fine-tuning data did not come from the previous pre-processing, you can just prepare your txt data and call the script build_meta_data.sh with the (sub) task in question. Since the codes and vocabulary must be preserved, we have prepared another script ([build_fine_tune_data.sh](scripts/build_fine_tune_data.sh)) in which we directly apply BPE tokenization on dataset and binarize everything using preprocess.py based on the codes and vocabulary of the meta-model. So we have to call this script for each subtask like this :

```
languages = 
chmod +x ../ft_data.sh
../ft_data.sh $languages
```

At this stage, restart the training as in the previous section with :
- lgs="en-fr"
- reload_model = path to the folder where you stored the meta-model
- `bt_steps'':"..."`, `ae_steps'':"..."` and/or `mt_steps'':"..."` (replace the three bullet points with your specific objectives if any)  
You can use one of the two previously trained meta-models: pre-formed meta-model (MLM, TLM) or meta-MT formed from the pre-formed meta-model. 

### 5. How to evaluate a language model trained on a language L on another language L'.

###### Our

<table class='table table-striped'><caption><b>?</b></caption><thead><tr><th scope='col'>Evaluated on (cols)---------<br/>Trained on (rows)</th><th scope='col'>Bafi</th><th scope='col'>Bulu</th><th scope='col'>Ewon</th><th scope='col'>Ghom</th><th scope='col'>Limb</th><th scope='col'>Ngie</th><th scope='col'>Dii</th><th scope='col'>Doya</th><th scope='col'>Peer</th><th scope='col'>Samb</th><th scope='col'>Guid</th><th scope='col'>Guiz</th><th scope='col'>Kaps</th><th scope='col'>Mofa</th><th scope='col'>Mofu</th><th scope='col'>Du_n</th><th scope='col'>Ejag</th><th scope='col'>Fulf</th><th scope='col'>Gbay</th><th scope='col'>MASS</th><th scope='col'>Tupu</th><th scope='col'>Vute</th></tr></thead><tbody><tr><th scope='row'>Bafi</th><td>15.155782/46.113990</td><td>3522.435230/12.694301</td><td>10532.574414/3.108808</td><td>3414.970521/10.103627</td><td>3662.233924/10.880829</td><td>4476.028980/2.072539</td><td>4594.588311/10.362694</td><td>3840.575574/13.989637</td><td><b>3111.148085/13.212435</b></td><td>4210.511141/8.031088</td><td>6607.939683/2.590674</td><td>7506.246899/3.108808</td><td>11121.594025/3.367876</td><td>3122.591005/13.212435</td><td>3183.283705/10.621762</td><td>5504.065998/8.549223</td><td>4127.620979/3.108808</td><td>9107.779213/6.994819</td><td>7440.762805/3.886010</td><td>4916.778213/12.176166</td><td>8239.932584/4.922280</td><td>3192.590598/10.362694</td></tr><tr><th scope='row'>Bulu</th><td><b>577.711688/9.585492</b></td><td>18.602898/43.264249</td><td>795.094593/17.357513</td><td>589.636415/13.471503</td><td>1482.709434/8.549223</td><td>1113.122905/12.435233</td><td>994.030274/11.658031</td><td>820.063393/10.103627</td><td>828.162228/11.658031</td><td>1519.449874/3.367876</td><td>1183.604483/9.326425</td><td>671.542857/13.989637</td><td>1427.515245/5.440415</td><td>657.031222/13.212435</td><td>1018.342338/6.217617</td><td>602.305603/10.880829</td><td>1066.765090/6.994819</td><td>1349.669421/6.476684</td><td>605.298410/13.989637</td><td>1615.328636/5.699482</td><td>2493.141092/8.290155</td><td>699.009937/13.730570</td></tr><tr><th scope='row'>Ewon</th><td>2930.433348/13.730570</td><td><b>784.556467/12.435233</b></td><td>439.343693/11.139896</td><td>8576.270483/3.886010</td><td>1408.305834/12.176166</td><td>6329.517824/5.181347</td><td>4374.527024/8.031088</td><td>5703.222147/4.922280</td><td>3226.438808/13.471503</td><td>5147.417352/9.585492</td><td>7383.547206/3.886010</td><td>2049.974847/13.730570</td><td>3458.765537/12.176166</td><td>1428.351000/11.139896</td><td>4890.406327/1.813472</td><td>2050.215975/11.917098</td><td>4693.132443/2.331606</td><td>3796.911033/9.844560</td><td>4985.892435/7.253886</td><td>3737.211837/11.658031</td><td>8497.461052/1.036269</td><td>8105.614715/2.590674</td></tr><tr><th scope='row'>Ghom</th><td>10826.769423/12.176166</td><td>7919.745037/10.621762</td><td>13681.624683/6.735751</td><td>112.759549/22.538860</td><td>8550.764036/13.212435</td><td>21351.213307/11.658031</td><td><b>5724.234345/11.917098</b></td><td>7638.186054/10.621762</td><td>8992.791640/6.735751</td><td>9870.502751/5.440415</td><td>8671.271306/14.248705</td><td>7952.305962/9.844560</td><td>17073.248866/7.253886</td><td>17507.383398/3.626943</td><td>6253.188979/12.435233</td><td>6616.060359/9.585492</td><td>31826.000072/3.108808</td><td>11636.816092/11.398964</td><td>6129.150512/14.507772</td><td>9667.854370/11.139896</td><td>14276.187678/8.031088</td><td>7152.109226/12.953368</td></tr><tr><th scope='row'>Limb</th><td>2348.605310/7.772021</td><td>5910.088736/10.103627</td><td>11640.836610/2.331606</td><td>2234.982947/8.031088</td><td>16.486114/48.186528</td><td>5240.029343/10.880829</td><td>3485.743598/11.139896</td><td><b>1744.289850/10.880829</b></td><td>2357.786346/11.658031</td><td>2829.453145/10.362694</td><td>6097.658965/6.735751</td><td>2806.032546/9.326425</td><td>2530.422427/11.139896</td><td>2234.037369/14.507772</td><td>3106.412553/9.067358</td><td>5675.990382/8.549223</td><td>4323.215519/10.880829</td><td>5303.094881/7.512953</td><td>3222.476499/10.362694</td><td>2619.771393/12.435233</td><td>6315.916126/12.435233</td><td>1965.282639/9.326425</td></tr><tr><th scope='row'>Ngie</th><td>2494.668579/10.621762</td><td>1683.610004/7.772021</td><td><b>645.074490/13.212435</b></td><td>2747.857945/10.621762</td><td>865.229192/8.031088</td><td>53.604331/32.642487</td><td>3487.877577/5.440415</td><td>2973.100164/9.844560</td><td>1694.041692/9.844560</td><td>2285.872589/8.808290</td><td>3555.658122/3.626943</td><td>2240.803918/4.663212</td><td>8214.745127/2.849741</td><td>2162.964776/8.290155</td><td>4130.931993/5.699482</td><td>1251.907556/9.585492</td><td>1406.624816/6.735751</td><td>1134.593481/8.031088</td><td>3484.481404/9.844560</td><td>1587.951832/9.326425</td><td>1786.015603/9.326425</td><td>2117.031454/10.103627</td></tr><tr><th scope='row'>Dii</th><td>5369.974508/5.181347</td><td>3526.951377/11.917098</td><td>4466.736657/2.590674</td><td>3468.181916/8.808290</td><td>1524.457754/10.880829</td><td><b>856.533233/10.362694</b></td><td>16.031832/47.150259</td><td>3570.945172/11.658031</td><td>1933.128270/11.139896</td><td>3086.805425/7.253886</td><td>5545.945984/3.626943</td><td>1592.451661/11.139896</td><td>7351.154713/2.331606</td><td>1430.511351/14.248705</td><td>4198.900876/4.145078</td><td>2587.338616/8.290155</td><td>3315.158358/2.590674</td><td>2903.721453/8.808290</td><td>4416.753252/3.886010</td><td>3044.769713/5.440415</td><td>3276.637193/10.362694</td><td>3551.309415/8.808290</td></tr><tr><th scope='row'>Doya</th><td>2413.178389/7.253886</td><td>2925.237118/9.326425</td><td>3035.126064/9.844560</td><td>6431.020717/4.404145</td><td>2888.802299/10.362694</td><td>4296.348738/9.585492</td><td>1963.357861/9.067358</td><td>225.399738/14.507772</td><td>2647.241446/4.663212</td><td>3559.797389/1.036269</td><td>3224.327707/8.549223</td><td>1628.560179/16.062176</td><td>7036.636934/2.072539</td><td>2378.384535/7.772021</td><td>2526.667089/10.103627</td><td>2560.562728/10.362694</td><td>3486.425933/7.253886</td><td>4898.016349/6.217617</td><td><b>1336.163366/12.176166</b></td><td>5378.777228/0.518135</td><td>2334.347220/9.585492</td><td>4210.426671/6.476684</td></tr><tr><th scope='row'>Peer</th><td>5417.812131/7.253886</td><td>3718.857566/8.290155</td><td>3921.429577/10.103627</td><td>8042.333854/2.590674</td><td>4744.329113/12.435233</td><td>2378.606152/7.772021</td><td>4297.265443/7.253886</td><td>7835.525318/3.108808</td><td>27.612503/46.113990</td><td>8547.481994/3.367876</td><td>7819.217930/4.922280</td><td><b>2009.553562/13.730570</b></td><td>7929.664487/2.590674</td><td>5227.466016/3.108808</td><td>2828.595071/10.103627</td><td>3109.933571/11.398964</td><td>3449.171674/7.512953</td><td>7517.809582/5.181347</td><td>3593.460649/9.326425</td><td>6490.444215/5.181347</td><td>8583.548031/6.994819</td><td>3640.649700/9.585492</td></tr><tr><th scope='row'>Samb</th><td>1921.203126/10.621762</td><td>2876.156252/8.808290</td><td>5222.268404/2.331606</td><td>2258.419159/8.808290</td><td>2940.603464/9.844560</td><td><b>757.885957/10.362694</b></td><td>2852.564926/3.886010</td><td>3568.046199/9.585492</td><td>3198.132105/11.658031</td><td>14.473909/45.336788</td><td>2135.946491/9.326425</td><td>1882.791510/12.435233</td><td>1380.449126/12.694301</td><td>2739.728389/6.217617</td><td>1114.151589/13.989637</td><td>2588.952886/10.362694</td><td>2408.673909/9.844560</td><td>1012.804391/13.471503</td><td>4310.704371/6.217617</td><td>2429.426652/3.108808</td><td>1681.603952/7.772021</td><td>2305.207465/4.404145</td></tr><tr><th scope='row'>Guid</th><td>11105.869490/11.917098</td><td>11350.393050/8.549223</td><td>24157.732815/2.331606</td><td>28800.139343/5.440415</td><td>9497.473893/11.139896</td><td>11941.642599/11.658031</td><td>26891.060403/2.072539</td><td>35288.834478/3.367876</td><td>11458.390164/9.326425</td><td>8581.012321/12.953368</td><td>669.152371/22.020725</td><td><b>8237.415053/12.953368</b></td><td>24641.309182/3.626943</td><td>12256.261503/6.735751</td><td>8329.239657/15.025907</td><td>18733.469719/2.590674</td><td>13013.633062/11.398964</td><td>22151.485850/4.922280</td><td>15139.079118/12.176166</td><td>12649.997596/11.139896</td><td>13526.708187/9.844560</td><td>14521.723680/13.471503</td></tr><tr><th scope='row'>Guiz</th><td>1900.984819/11.917098</td><td>3422.299591/5.440415</td><td>2920.779863/13.212435</td><td>2657.232975/3.886010</td><td>7763.772745/6.217617</td><td>2516.088934/11.398964</td><td>1556.474440/12.953368</td><td><b>1450.939238/12.694301</b></td><td>1852.263760/12.435233</td><td>3503.139397/5.440415</td><td>1957.981930/7.772021</td><td>5.612643/60.362694</td><td>2030.975178/10.621762</td><td>3100.456750/9.585492</td><td>3816.057439/9.067358</td><td>2527.372931/10.103627</td><td>2017.135324/9.585492</td><td>1771.010720/12.953368</td><td>2467.262902/9.067358</td><td>6465.542228/6.735751</td><td>4936.521836/5.181347</td><td>3251.372451/4.663212</td></tr><tr><th scope='row'>Kaps</th><td>4787.151015/7.772021</td><td>4026.495938/9.067358</td><td>2591.212157/13.730570</td><td>3963.789278/11.139896</td><td>4835.168698/9.844560</td><td>3738.018788/5.958549</td><td>3472.599548/9.067358</td><td>2846.824328/9.067358</td><td>3964.442923/6.217617</td><td>8248.174848/4.663212</td><td>3178.776910/9.326425</td><td>4521.187784/6.476684</td><td>6.392693/63.730570</td><td>4535.673748/6.476684</td><td>2285.708359/13.730570</td><td>5222.426332/5.699482</td><td>4409.982716/5.440415</td><td><b>2124.534904/10.362694</b></td><td>4863.209844/10.362694</td><td>4875.780156/3.886010</td><td>4278.744225/12.176166</td><td>4661.710772/9.067358</td></tr><tr><th scope='row'>Mofa</th><td>5555.267163/7.772021</td><td>5328.793555/11.658031</td><td>6064.913246/13.730570</td><td>8844.481560/5.181347</td><td>14355.051790/6.217617</td><td>10773.098216/8.290155</td><td>5702.554716/11.398964</td><td>11819.967712/5.958549</td><td>5810.652609/12.435233</td><td>10899.166334/6.476684</td><td>9606.038800/5.699482</td><td><b>4528.077873/13.471503</b></td><td>10261.988658/9.844560</td><td>38.718690/38.341969</td><td>7191.371927/8.290155</td><td>4847.594375/14.248705</td><td>8110.295270/9.844560</td><td>14375.814958/5.699482</td><td>10070.806870/3.626943</td><td>10826.318474/8.290155</td><td>10187.374717/7.772021</td><td>16953.170797/3.626943</td></tr><tr><th scope='row'>Mofu</th><td>2175.168540/11.658031</td><td>3005.393159/10.621762</td><td>2773.793897/7.253886</td><td>2257.313709/6.476684</td><td>1807.203325/13.471503</td><td>2481.194623/2.331606</td><td>1626.688315/12.435233</td><td>1473.207901/13.212435</td><td>3206.638463/8.290155</td><td><b>1358.112972/12.435233</b></td><td>2550.513183/10.880829</td><td>1867.275865/12.694301</td><td>2847.897967/4.145078</td><td>1645.699003/13.471503</td><td>50.399227/32.642487</td><td>3831.820284/3.108808</td><td>1679.421861/9.844560</td><td>1957.944241/13.989637</td><td>1655.398024/13.212435</td><td>3439.753108/6.735751</td><td>4164.392749/9.844560</td><td>2176.478824/10.103627</td></tr><tr><th scope='row'>Du_n</th><td>3358.977688/12.694301</td><td>8269.025689/5.958549</td><td>6784.926221/4.922280</td><td>4034.987828/10.362694</td><td>8317.977821/5.440415</td><td>4469.988388/9.326425</td><td>4581.242219/9.585492</td><td>4046.289387/10.880829</td><td>4587.843666/10.880829</td><td>4061.430238/12.435233</td><td>4116.231632/8.031088</td><td>4043.687467/11.658031</td><td>8587.884922/5.699482</td><td><b>2518.760103/13.989637</b></td><td>9252.838415/6.217617</td><td>38.646292/34.196891</td><td>2823.000209/11.658031</td><td>7688.259347/5.699482</td><td>4184.395191/9.844560</td><td>6460.323149/9.844560</td><td>12418.880207/5.699482</td><td>4394.753911/10.362694</td></tr><tr><th scope='row'>Ejag</th><td>878.221181/8.290155</td><td>2977.854246/10.362694</td><td>1122.454274/13.212435</td><td>4066.806240/3.626943</td><td>4401.408293/12.694301</td><td>1324.839235/11.139896</td><td>2760.972117/9.585492</td><td>802.718089/8.808290</td><td>1935.328428/6.735751</td><td>2456.134064/8.549223</td><td>948.726346/11.658031</td><td>1464.326862/6.994819</td><td>1999.633312/6.476684</td><td>2483.815842/4.663212</td><td>790.752998/11.917098</td><td>1436.471564/10.362694</td><td>27.125567/39.896373</td><td>2701.314483/8.549223</td><td><b>739.895562/13.989637</b></td><td>1119.207373/9.844560</td><td>2061.967307/3.367876</td><td>3116.635849/4.663212</td></tr><tr><th scope='row'>Fulf</th><td>3122.754082/11.139896</td><td>3172.412810/8.290155</td><td>2632.034499/10.103627</td><td>1803.237123/14.507772</td><td>3015.507576/12.953368</td><td>4697.430105/10.621762</td><td>2221.398811/11.917098</td><td>3338.511704/7.772021</td><td>5857.163684/4.663212</td><td>2631.329961/12.694301</td><td><b>1756.767457/14.248705</b></td><td>3965.216351/8.031088</td><td>2961.580251/10.362694</td><td>1850.532804/14.248705</td><td>2431.677037/8.808290</td><td>2688.040706/8.549223</td><td>6237.846441/3.108808</td><td>9.819160/53.108808</td><td>1794.314668/12.435233</td><td>2633.154009/4.922280</td><td>5899.732260/9.585492</td><td>6035.594459/5.440415</td></tr><tr><th scope='row'>Gbay</th><td>3537.010215/8.808290</td><td>2213.336729/9.326425</td><td>958.976958/14.766839</td><td>2170.105117/2.849741</td><td>2381.840897/8.549223</td><td>1092.011356/11.398964</td><td>989.079405/15.284974</td><td>2110.708219/12.953368</td><td>1212.493865/13.989637</td><td>1342.159428/12.953368</td><td><b>784.478130/16.321244</b></td><td>1404.757907/15.284974</td><td>1949.759014/13.730570</td><td>1165.979838/12.694301</td><td>1940.255308/5.699482</td><td>1073.951745/13.730570</td><td>2180.263932/7.253886</td><td>2639.229412/8.031088</td><td>4.503568/64.766839</td><td>2711.475687/5.440415</td><td>2879.142805/11.139896</td><td>2777.515280/3.626943</td></tr><tr><th scope='row'>MASS</th><td>2052.763675/6.476684</td><td>2123.090411/11.139896</td><td>1150.690864/11.398964</td><td><b>404.857470/19.170984</b></td><td>4114.380214/2.849741</td><td>1177.460159/10.880829</td><td>1553.261634/11.917098</td><td>767.332823/13.212435</td><td>1558.036793/6.217617</td><td>673.483311/13.730570</td><td>1308.799442/6.735751</td><td>2525.700131/5.440415</td><td>1157.282835/14.248705</td><td>1665.795367/8.031088</td><td>969.622799/11.139896</td><td>2236.251124/10.621762</td><td>1768.310288/9.585492</td><td>1530.460913/10.621762</td><td>703.513823/14.766839</td><td>9.311520/52.072539</td><td>3781.478640/5.440415</td><td>783.170102/16.580311</td></tr><tr><th scope='row'>Tupu</th><td>499.010245/24.611399</td><td>2789.182977/9.844560</td><td>1176.557896/16.062176</td><td>335.366353/21.243523</td><td>3759.854817/4.922280</td><td>1473.248900/8.290155</td><td>1637.969909/15.284974</td><td>444.487258/23.056995</td><td>729.184899/19.430052</td><td>326.348924/24.611399</td><td>530.140976/24.611399</td><td>834.757176/20.207254</td><td>1014.747872/11.398964</td><td>1361.103340/11.398964</td><td>447.754239/17.875648</td><td>1313.622745/15.803109</td><td>2020.767969/9.326425</td><td>1234.031067/13.730570</td><td><b>242.696296/29.533679</b></td><td>1209.709716/14.766839</td><td>5.328121/62.953368</td><td>678.820813/13.730570</td></tr><tr><th scope='row'>Vute</th><td>5247.001730/8.290155</td><td>2972.688386/11.398964</td><td>3141.040872/9.067358</td><td>4304.014532/12.435233</td><td>2981.350915/10.880829</td><td>7944.078280/2.331606</td><td>3013.186151/13.730570</td><td>2532.120943/12.176166</td><td>4688.069751/9.844560</td><td>8022.399859/3.886010</td><td>5315.095277/3.626943</td><td><b>2075.166168/12.694301</b></td><td>3794.597938/12.176166</td><td>2879.870276/13.212435</td><td>4364.837110/3.367876</td><td>3858.872867/8.549223</td><td>2749.070864/10.880829</td><td>9917.265191/3.367876</td><td>8091.176547/3.108808</td><td>5939.386425/4.404145</td><td>7670.501815/2.849741</td><td>43.658700/33.419689</td></tr></tbody></table>

###### Prerequisite
If you want to evaluate the LM on a language `lang`, you must first have a file named `lang.txt` in the `$src_path` directory of [eval_data.sh](eval_data.sh).  
For examplel if you want to use the biblical corpus, you can run [scripts/bible.py](scripts/bible.py) :
```
# folder containing the csvs folder
csv_path=
# folder in which the objective folders will be created (mono or para)
output_dir=
# monolingual one ("mono") or parallel one ("para")
data_type=mono
# list of languages to be considered in alphabetical order and separated by a comma
# case of one language
languages=lang,lang  
# case of many languages
languages=lang1,lang2,...   
old_only : use only old testament
#  use only new testament
new_only=True

python ../scripts/bible.py --csv_path $csv_path --output_dir $output_dir --data_type $data_type --languages $languages --new_only $new_only
```
See other parameters in [scripts/bible.py](scripts/bible.py)

###### Data pre-processing
Modify parameters in [eval_data.sh](eval_data.sh)
```
# languages to be evaluated
languages=lang1,lang2,... 
chmod +x ../eval_data.sh 
../eval_data.sh $languages
```

###### Evaluation 

We take the language to evaluate (say `Bulu`), replace the files `test.Bulu.pth` (which was created with the `VOCAB` and `CODE` of `Bafi`, the evaluating language) with `test.Bafi.pth` (since `Bafi` evaluates, the `train.py` script requires that the dataset has the (part of the) name of the `lgs`). Then we just run the evaluation, the results (acc and ppl) we get is the result of LM Bafia on the Bulu language.

```
# evaluating language
tgt_pair=
# folder containing the data to be evaluated (must match $tgt_path in eval_data.sh)
src_path=
# You have to change two parameters in the configuration file used to train the LM which evaluates ("data_path":"$src_path" and "eval_only": "True")
# You must also specify the "reload_model" parameter, otherwise the last checkpoint found will be loaded for evaluation.
config_file=../configs/lm_template.json 
# languages to be evaluated
eval_lang= 
chmod +x ../scripts/evaluate.sh
../scripts/evaluate.sh $eval_lang
```
When the evaluation is finished you will see a file named `eval.log` in the `$dump_path/$exp_name/$exp_id` folder containing the evaluation results.    
**Note** :The description given below is only valid when the LM evaluator has been trained on only one language (and therefore without TLM). But let's consider the case where the basic LM has been trained on `en-fr` and we want to evaluate it on `de` or `de-ru`. `$tgt_pair` deviates from `en-fr`, but `language` varies depending on whether the evaluation is going to be done on one language or two:  
- In the case of `de` : `lang=de-de`  
- in the case of `de-ru`: `lang=de-ru`.

## IV. References

Please cite [[1]](https://openreview.net/forum?id=Q5ZxoD2LqcI) and/or  [[2]](https://arxiv.org/abs/1901.07291) and/or [[3]](https://arxiv.org/abs/1911.02116) if you found the resources in this repository useful.

### On the use of linguistic similarities to improve Neural Machine Translation for African Languages

[1] Tikeng Notsawo Pascal, NANDA ASSOBJIO Brice Yvan and James Assiene
```
@misc{
pascal2021on,
title={On the use of linguistic similarities to improve Neural Machine Translation for African Languages},
author={Tikeng Notsawo Pascal and NANDA ASSOBJIO Brice Yvan and James Assiene},
year={2021},
url={https://openreview.net/forum?id=Q5ZxoD2LqcI}
}
```

### Cross-lingual Language Model Pretraining

[2] G. Lample *, A. Conneau * [*Cross-lingual Language Model Pretraining*](https://arxiv.org/abs/1901.07291) and [facebookresearch/XLM](https://github.com/facebookresearch/XLM)

\* Equal contribution. Order has been determined with a coin flip.

```
@article{lample2019cross,
  title={Cross-lingual Language Model Pretraining},
  author={Lample, Guillaume and Conneau, Alexis},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

### Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

[3] Chelsea Finn, Pieter Abbeel, Sergey Levine [*Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*](https://arxiv.org/abs/1911.02116) and [cbfinn/maml](https://github.com/cbfinn/maml)

```
@article{Chelsea et al.,
  title={Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks},
  author={Chelsea Finn, Pieter Abbeel, Sergey Levine},
  journal={Proceedings of the 34th International Conference on Machine Learning, Sydney, Australia, PMLR 70, 2017},
  year={2017}
}
```

## License

See the [LICENSE](LICENSE) file for more details.





