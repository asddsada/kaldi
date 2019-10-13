#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using xvectors.
# It is closely based on "X-vectors: Robust DNN Embeddings for Speaker
# Recognition" by Snyder et al.  In the future, we will add score-normalization
# and a more effective form of PLDA domain adaptation.
#
# Pretrained models are available for this recipe.  See
# http://kaldi-asr.org/models.html and
# https://david-ryan-snyder.github.io/2017/10/04/model_sre16_v2.html
# for details.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
export=`pwd`/../asr_s5/export/LibriSpeech/

# SRE16 trials
nnet_dir=exp/xvector_nnet_1a

stage=0
if [ $stage -le 0 ]; then
  # Path to some, but not all of the training corpora
  data_root=/export/corpora/LDC

  # Prepare telephone and microphone speech from Mixer6.
  local/make_mx6.sh $data_root/LDC2013S03 data/

  # Prepare SWBD corpora.
  local/make_swbd_cellular1.pl $data_root/LDC2001S13 \
    data/swbd_cellular1_train
  local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
    data/swbd_cellular2_train
  local/make_swbd2_phase1.pl $data_root/LDC98S75 \
    data/swbd2_phase1_train
  local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
    data/swbd2_phase2_train
  local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
    data/swbd2_phase3_train

  # Combine all SWB corpora into one dataset.
  utils/combine_data.sh data/tel \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

  ##### Prepare unlabeled Cantonese and Tagalog development data. This dataset
  # was distributed to SRE participants.
  #local/make_sre16_unlabeled.pl /export/corpora5/SRE/LDC2016E46_SRE16_Call_My_Net_Training_Data data
  
  ##make train_major
  
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $export/$part data/$(echo $part | sed s/-/_/g)
  done
  mv data/train_clean_100 data/train
  for test in dev_clean test_clean dev_other test_other; do
    local/make_trails.pl data/$test
  done
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
 for name in tel dev_clean test_clean dev_other test_other train train_major; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
  utils/combine_data.sh --extra-files "utt2num_frames" data/tel_train data/tel data/train
  utils/fix_data_dir.sh data/tel_train
fi

# In this section, we augment the SWBD and SRE data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/tel_train/utt2num_frames > data/tel_train/reco2dur

  if [ ! -d "export/corpora/RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
    mv RIRS_NOISES "export/corpora/"
  fi
  
  if [ ! -d "export/corpora/musan" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/17/musan.tar.gz 
    tar -xvzf musan.tar.gz 
    mv musan "export/corpora/"
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/tel_train data/tel_train_reverb
  cp data/tel_train/vad.scp data/tel_train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/tel_train_reverb data/tel_train_reverb.new
  rm -rf data/tel_traine_reverb
  mv data/tel_train_reverb.new data/tel_train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh --sampling-rate 16000 /export/corpora/musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/tel_train data/tel_train_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/tel_train data/tel_train_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/tel_train data/tel_train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/tel_train_aug data/tel_train_reverb data/tel_train_noise data/tel_train_music data/tel_train_babble

  # Take a random subset of the augmentations (128k is somewhat larger than twice
  # the size of the SWBD+SRE list)
  utils/subset_data_dir.sh data/tel_train_aug 128000 data/tel_train_aug_128k
  utils/fix_data_dir.sh data/tel_train_aug_128k

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/tel_train_aug_128k exp/make_mfcc $mfccdir

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/tel_train_combined data/tel_train_aug_128k data/tel_train

  # Filter out the clean + augmented portion of the SRE list.  This will be used to
  # train the PLDA model later in the script.
  utils/copy_data_dir.sh data/tel_train_combined data/train_combined
  utils/filter_scp.pl data/train/spk2utt data/tel_train_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > data/train_combined/utt2spk
  utils/fix_data_dir.sh data/train_combined

fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 3 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    data/tel_train_combined data/tel_train_combined_no_sil exp/tel_train_combined_no_sil
  utils/fix_data_dir.sh data/tel_train_combined_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=500
  mv data/tel_train_combined_no_sil/utt2num_frames data/tel_train_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/tel_train_combined_no_sil/utt2num_frames.bak > data/tel_train_combined_no_sil/utt2num_frames
  utils/filter_scp.pl data/tel_train_combined_no_sil/utt2num_frames data/tel_train_combined_no_sil/utt2spk > data/tel_train_combined_no_sil/utt2spk.new
  mv data/tel_train_combined_no_sil/utt2spk.new data/tel_train_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/tel_train_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/tel_train_combined_no_sil/spk2utt > data/tel_train_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/tel_train_combined_no_sil/spk2num | utils/filter_scp.pl - data/tel_train_combined_no_sil/spk2utt > data/tel_train_combined_no_sil/spk2utt.new
  mv data/tel_train_combined_no_sil/spk2utt.new data/tel_train_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/tel_train_combined_no_sil/spk2utt > data/tel_train_combined_no_sil/utt2spk

  utils/filter_scp.pl data/tel_train_combined_no_sil/utt2spk data/tel_train_combined_no_sil/utt2num_frames > data/tel_train_combined_no_sil/utt2num_frames.new
  mv data/tel_train_combined_no_sil/utt2num_frames.new data/tel_train_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/tel_train_combined_no_sil
fi

local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 \
  --data data/tel_train_combined_no_sil --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs

if [ $stage -le 7 ]; then
  # The SRE16 major is an unlabeled dataset consisting of Cantonese and
  # and Tagalog.  This is useful for things like centering, whitening and
  # score normalization.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
    $nnet_dir data/train_major \
    exp/xvectors_train_major

  # Extract xvectors for SRE data (includes Mixer 6). We'll use this for
  # things like LDA or PLDA.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 40 \
    $nnet_dir data/train_combined \
    exp/xvectors_train_combined

  # The SRE16 test data
  for test in dev_clean test_clean dev_other test_other; do
      sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
        $nnet_dir data/${test} \
        exp/xvectors_${test}
  done
fi

if [ $stage -le 8 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd exp/xvectors_train_major/log/compute_mean.log \
    ivector-mean scp:exp/xvectors_train_major/xvector.scp \
    exp/xvectors_train_major/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $train_cmd exp/xvectors_train_combined/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_train_combined/xvector.scp ark:- |" \
    ark:data/train_combined/utt2spk exp/xvectors_train_combined/transform.mat || exit 1;

  # Train an out-of-domain PLDA model.
  $train_cmd exp/xvectors_train_combined/log/plda.log \
    ivector-compute-plda ark:data/train_combined/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_train_combined/xvector.scp ark:- | transform-vec exp/xvectors_train_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/xvectors_train_combined/plda || exit 1;

  # Here we adapt the out-of-domain PLDA model to SRE16 major, a pile
  # of unlabeled in-domain data.  In the future, we will include a clustering
  # based approach for domain adaptation, which tends to work better.
  $train_cmd exp/xvectors_train_major/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    exp/xvectors_train_combined/plda \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_train_major/xvector.scp ark:- | transform-vec exp/xvectors_train_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    exp/xvectors_train_major/plda_adapt || exit 1;
fi

if [ $stage -le 9 ]; then
  # Get results using the out-of-domain PLDA model.
  for test in dev_clean test_clean dev_other test_other; do
      $train_cmd exp/scores/log/${test}_scoring.log \
        ivector-plda-scoring --normalize-length=true \
#        --num-utts=ark:exp/xvectors_sre16_eval_enroll/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 exp/xvectors_train_combined/plda - |" \
#        "ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:exp/xvectors_sre16_eval_enroll/xvector.scp | "\
        "ark:ivector-subtract-global-mean exp/xvectors_train_major/mean.vec ark:- ark:- | transform-vec exp/xvectors_train_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean exp/xvectors_train_major/mean.vec scp:exp/xvectors_${test}/xvector.scp ark:- | transform-vec exp/xvectors_train_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat 'data/${test}/trials' | cut -d\  --fields=1,2 |" exp/scores/${test}_scores || exit 1;

      pooled_eer=$(paste data/${test}/trials exp/scores/${test}_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
      echo "Using Out-of-Domain PLDA, EER: ${test} Pooled ${pooled_eer}%"
  done
  # EER: Pooled 11.73%, Tagalog 15.96%, Cantonese 7.52%
  # For reference, here's the ivector system from ../v1:
  # EER: Pooled 13.65%, Tagalog 17.73%, Cantonese 9.61%
fi

if [ $stage -le 10 ]; then
  # Get results using the adapted PLDA model.
  for test in dev_clean test_clean dev_other test_other; do
      $train_cmd exp/scores/log/${test}_scoring_adapt.log \
        ivector-plda-scoring --normalize-length=true \
        #--num-utts=ark:exp/xvectors_sre16_eval_enroll/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 exp/xvectors_train_major/plda_adapt - |" \
        #"ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:exp/xvectors_sre16_eval_enroll/xvector.scp | "\
        "ark:ivector-subtract-global-mean exp/xvectors_train_major/mean.vec ark:- ark:- | transform-vec exp/xvectors_train_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean exp/xvectors_train_major/mean.vec scp:exp/xvectors_${test}/xvector.scp ark:- | transform-vec exp/xvectors_train_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat 'data/${test}/trials' | cut -d\  --fields=1,2 |" exp/scores/${test}_scores_adapt || exit 1;

      pooled_eer=$(paste data/${test}/trials exp/scores/${test}_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
      echo "Using Adapted PLDA, EER: ${test} Pooled ${pooled_eer}%"
  done 
  # EER: Pooled 8.57%, Tagalog 12.29%, Cantonese 4.89%
  # For reference, here's the ivector system from ../v1:
  # EER: Pooled 12.98%, Tagalog 17.8%, Cantonese 8.35%
fi
