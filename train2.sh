#!/bin/bash
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32

# Only train the v13 model for the MUL data

echo ">>> CLEAN UP"
echo rm -rf /data/working
rm -rf /data/working && mkdir -p /data/working

source activate py35 && for train_path in $@; do
    echo ">>> PREPROCESSING STEP ---------------------------"
    echo python v5_im2.py preproc_train $train_path
    python v5_im2.py preproc_train $train_path
    echo python v12_im2.py preproc_train $train_path
    python v12_im2.py preproc_train $train_path

    ### v13 --------------
    # Training for v13 model
    echo ">>>>>>>>>> v13.py"
    python v13_2.py validate $train_path
    # Parametr optimization for v13 model
    echo ">>>>>>>>>> v13.py"
    python v13_2.py evalfscore $train_path

done
