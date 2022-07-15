Running DECIMER

How to run from your shell:

   
    $ conda create --name DECIMER_I2S python=3.7
    $ conda activate DECIMER_I2S
    $ conda install pip
    $ pip install tensorflow==2.3.0 pillow deepsmiles sklearn
    $ git clone https://github.com/Kohulan/DECIMER-Image-to-SMILES decimer
    $ cd decimer
    $ wget https://storage.googleapis.com/decimer_weights/Trained_Models.zip
    $ unzip Trained_Models.zip
    $ cd Network

Models can be downloaded from [Trained Model](https://storage.googleapis.com/decimer_weights/Trained_Models.zip) and should be placed under Trained_Models directory

    
    $ python Predictor.py --input ../Sample_Images/P_59687969.png
    
