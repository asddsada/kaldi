#!/bin/bash

. cmd.sh


stage=7
train_stage=-10
use_gpu=true
set -e
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


# assume use_gpu=true since it would be way too slow otherwise.

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi
parallel_opts="-l gpu=4"
num_threads=16
minibatch_size=128
#####minibatch_size=512
dir=exp/nnet2_online/nnet_ms_a
mkdir -p exp/nnet2_online

echo start
# Stages 1 through 5 are done in run_nnet2_common.sh,
# so it can be shared with other similar scripts.
local/online/run_nnet2_common.sh --stage $stage

echo PASSSS
if [ $stage -le 6 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
    utils/create_split_dir.pl /export/b0{6,7,8,9}/$(USER)/kaldi-data/egs/fisher_english/s5/$dir/egs/storage $dir/egs/storage
  fi

  # Because we have a lot of data here and we don't want the training to take
  # too long, we reduce the number of epochs from the defaults (15 + 5) to (3 +
  # 1).  The option "--io-opts '--max-jobs-run 12'" is to have more than the default number
  # (5) of jobs dumping the egs to disk; this is OK since we're splitting our
  # data across four filesystems for speed.

#--splice-indexes "layer0/-2:-1:0:1:2 layer1/-1:2 layer3/-3:3 layer4/-7:2" \
  steps/nnet2/train_multisplice_accel2.sh --stage $train_stage \
    --feat-type raw \
    --splice-indexes "layer0/-2:-1:0:1:2 layer1/-1:2" \
    --num-epochs 16 \
    --num-hidden-layers 2 \
    --num-jobs-initial 3 --num-jobs-final 3 \
    --online-ivector-dir exp/nnet2_online/ivectors_train \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --mix-up 12000 \
    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
    --cmd "$decode_cmd" \
    --egs-dir "$common_egs_dir" \
    --pnorm-input-dim 2500 \
    --pnorm-output-dim 250 \
    --cleanup false \
    data/train_hires data/lang exp/tri5a $dir  || exit 1;

fi


COUNTER=1
while [  $COUNTER -le 22 ]; do

  echo XXXX
  if [ $stage -le 7 ]; then
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf --iter $COUNTER \
  data/lang exp/nnet2_online/extractor "$dir" ${dir}_online || exit 1;
  fi
  
  echo "iter $COUNTER"
  #if [ $stage -le 8 ]; then
  # do the actual online decoding with iVectors, carrying info forward from
  # previous utterances of the same speaker.
  #   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 12 --iter $COUNTER \
  #      exp/tri5a/graph data/test ${dir}_online/decode_test_${COUNTER} || exit 1;
  #fi
  #echo ZZZZ
  #if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  #   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 12 --iter $COUNTER \
  #     --per-utt true \
  #      exp/tri5a/graph data/test ${dir}_online/decode_test_utt_${COUNTER} || exit 1;
  #fi
  #echo TTTTTTTTTT
  if [ $stage -le 10 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information, but looks to the end
  # of the utterance while computing the iVector.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 12 --iter $COUNTER \
  --per-utt true --online false \
  exp/tri5a/graph data/test ${dir}_online/decode_test_utt_offline_${COUNTER} || exit 1;
  fi

  let COUNTER=COUNTER+1 
done

exit 0;

