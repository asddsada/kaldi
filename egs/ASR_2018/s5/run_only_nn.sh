#!/bin/bash

# It's best to run the commands in this one by one.

. cmd.sh || exit 1
. path.sh || exit 1
mfccdir=`pwd`/mfcc
set -e

njobs=16

# this will help find issues with the lexicon.
# steps/cleanup/debug_lexicon.sh --nj 300 --cmd "$train_cmd" data/train_100k data/lang exp/tri5a data/local/dict/lexicon.txt exp/debug_lexicon_100k


# The step below won't run by default; it demonstrates a data-cleaning method.
# It doesn't seem to help in this setup; maybe the data was clean enough already.
# local/run_data_cleaning.sh

## The following is the best current neural net recipe.
local/online/run_nnet2_multisplice.sh

# ## The following is an older recipe without multi-splice.
# # local/online/run_nnet2.sh


# ## The following is another older nnet recipe, on top of fMLLR features.
# # local/run_nnet2.sh
#

