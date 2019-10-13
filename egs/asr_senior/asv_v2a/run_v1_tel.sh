#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using ivectors.
# In the future, we will add score-normalization and a more effective form of
# PLDA domain adaptation.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
export=`pwd`/../asr_s5/export/LibriSpeech/

# SRE16 trials
sre16_trials=data/sre16_eval_test/trials

stage=0
if [ $stage -le 0 ]; then
  # Path to some, but not all of the training corpora
  data_root=/export/corpora/LDC

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
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in dev_clean test_clean dev_other test_other train tel train_major; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

if [ $stage -le 2 ]; then
  # Train the UBM.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
    --nj 40 --num-threads 8  --subsample 1 \
    data/train_major 2048 \
    exp/diag_ubm

  sid/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
    --nj 40 --remove-low-count-gaussians false --subsample 1 \
    data/train_major \
    exp/diag_ubm exp/full_ubm
fi

if [ $stage -le 3 ]; then
  # Train the i-vector extractor.
  utils/combine_data.sh data/tel_train data/tel data/train
  sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 35G" \
    --ivector-dim 600 \
    --num-iters 5 \
    exp/full_ubm/final.ubm data/tel_train \
    exp/extractor
fi

# In this section, we augment the SRE data with reverberation,
# noise, music, and babble, and combined it with the clean SRE
# data.  The combined list will be used to train the PLDA model.
if [ $stage -le 4 ]; then
  utils/data/get_utt2num_frames.sh --nj 40 --cmd "$train_cmd" data/train
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur

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

  # Take a random subset of the augmentations (64k is roughly the size of the SRE dataset)
  utils/subset_data_dir.sh data/train_aug 64000 data/train_aug_64k
  utils/fix_data_dir.sh data/train_aug_64k

  # Make MFCCs for the augmented data.  Note that we want we should alreay have the vad.scp
  # from the clean version at this point, which is identical to the clean version!
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/train_aug_64k exp/make_mfcc $mfccdir

  # Combine the clean and augmented SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/train_combined data/train_aug_64k data/train
fi

if [ $stage -le 5 ]; then
  # Extract i-vectors for SRE data (includes Mixer 6). We'll use this for
  # things like LDA or PLDA.
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
    exp/extractor data/train_combined \
    exp/ivectors_train_combined

  # The SRE16 major is an unlabeled dataset consisting of Cantonese and
  # and Tagalog.  This is useful for things like centering, whitening and
  # score normalization.
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
    exp/extractor data/train_major \
    exp/ivectors_train_major

  # The SRE16 test data
  for test in dev_clean test_clean dev_other test_other; do
      sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
        exp/extractor data/${test} \
        exp/ivectors_${test}
  done
fi

if [ $stage -le 6 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_train_major/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_train_major/ivector.scp \
    exp/ivectors_train_major/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd exp/ivectors_train_combined/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train_combined/ivector.scp ark:- |" \
    ark:data/train_combined/utt2spk exp/ivectors_train_combined/transform.mat || exit 1;

  #  Train the PLDA model.
  $train_cmd exp/ivectors_train_combined/log/plda.log \
    ivector-compute-plda ark:data/train_combined/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train_combined/ivector.scp ark:- | transform-vec exp/ivectors_train_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/ivectors_train_combined/plda || exit 1;

  # Here we adapt the out-of-domain PLDA model to SRE16 major, a pile
  # of unlabeled in-domain data.  In the future, we will include a clustering
  # based approach for domain adaptation.
  $train_cmd exp/ivectors_train_major/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    exp/ivectors_train_combined/plda \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train_major/ivector.scp ark:- | transform-vec exp/ivectors_train_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    exp/ivectors_train_major/plda_adapt || exit 1;
fi

if [ $stage -le 7 ]; then
  # Get results using the out-of-domain PLDA model
  for test in dev_clean test_clean dev_other test_other; do
      $train_cmd exp/scores/log/${test}_scoring.log \
        ivector-plda-scoring --normalize-length=true \
        #--num-utts=ark:exp/ivectors_sre16_eval_enroll/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 exp/ivectors_train_combined/plda - |" \
        #"ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:exp/ivectors_sre16_eval_enroll/ivector.scp | "\
        "ark:ivector-subtract-global-mean exp/ivectors_train_major/mean.vec ark:- ark:- | transform-vec exp/ivectors_train_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean exp/ivectors_train_major/mean.vec scp:exp/ivectors_${test}/ivector.scp ark:- | transform-vec exp/ivectors_train_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat 'data/${test}/trials' | cut -d\  --fields=1,2 |" exp/scores/${test}_scores || exit 1;
      pooled_eer=$(paste data/${test}/trials exp/scores/${test}_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
      echo "Using Out-of-Domain PLDA, EER: ${test} Pooled ${pooled_eer}%, Tagalog ${tgl_eer}%, Cantonese ${yue_eer}%"
  done
  # EER: Pooled 13.65%, Tagalog 17.73%, Cantonese 9.612%
fi

if [ $stage -le 8 ]; then
  # Get results using an adapted PLDA model. In the future we'll replace
  # this (or add to this) with a clustering based approach to PLDA adaptation.
  for test in dev_clean test_clean dev_other test_other; do
      $train_cmd exp/scores/log/${test}_scoring_adapt.log \
        ivector-plda-scoring --normalize-length=true \
        #--num-utts=ark:exp/ivectors_sre16_eval_enroll/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 exp/ivectors_train_major/plda_adapt - |" \
        #"ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:exp/ivectors_sre16_eval_enroll/ivector.scp | "\
        "ark:ivector-subtract-global-mean exp/ivectors_train_major/mean.vec ark:- ark:- | transform-vec exp/ivectors_train_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean exp/ivectors_train_major/mean.vec scp:exp/ivectors_${test}/ivector.scp ark:- | transform-vec exp/ivectors_train_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat 'data/${test}/trials' | cut -d\  --fields=1,2 |" exp/scores/${test}_scores_adapt || exit 1;

      pooled_eer=$(paste data/${test}/trials exp/scores/${test}_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
      echo "Using Adapted PLDA, EER: Pooled ${pooled_eer}%, Tagalog ${tgl_eer}%, Cantonese ${yue_eer}%"
  done
  # EER: Pooled 12.98%, Tagalog 17.8%, Cantonese 8.35%
fi