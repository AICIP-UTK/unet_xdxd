#!/bin/bash
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32

ARGS=$@
shift $(($# - 1))
OUT=$1

# clean up
mkdir -p /data/output /data/working
#rm -f /data/working/images/v5/test_AOI_*_im.h5
#rm -f /data/working/images/v5/test_AOI_*_mul.h5
#rm -f /data/working/images/v12/test_AOI_*_mul.h5

source activate py35 && for test_path in $ARGS; do
    # Skip last arg
    [[ $test_path = $OUT ]] && break

    echo ">>> TEST PREPROCESSING STEP"
    echo ">>>" python v5_im2.py preproc_test $test_path
    python v5_im2.py preproc_test $test_path
    #echo ">>>" python v12_im2.py preproc_test $test_path
    #python v12_im2.py preproc_test $test_path

    echo ">>> INFERENCE STEP"
    # echo ">>>" python v13_2.py testproc $test_path
    # python v13_2.py testproc $test_path
    echo ">>>" python v13_2.py test $test_path
    python v13_3.py test $test_path
done

# Merge infenrece results
#echo ">>> MERGE INFERENCE RESULTS"
#echo ">>>" python merge.py $ARGS
#python merge.py $ARGS
