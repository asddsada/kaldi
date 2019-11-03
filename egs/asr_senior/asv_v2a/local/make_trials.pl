#!/usr/bin/perl

# base on sre16 local/make_sre16_eval.pl

# Usage: make_trials.pl test data/test/trials

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-data-dir>/<test-dir>/ <path-to-output>\n";
  print STDERR "e.g. $0 test data/test/trials\n";
  exit(1);
}

($test_dir, $out_dir) = @ARGV;

#system "cp $test_dir/utt2spk $test_dir/utt2spk_tmp;";
system "echo $0: Creating trials file in $test_dir";

open(UTT2SPK, "<", "data/$test_dir/utt2spk") or die "Could not open data/$test_dir/utt2spk";
open(TRIALS, ">", "$out_dir") or die "Could not open $out_dir";

while (<UTT2SPK>) {
  chomp;
  my ($utt_id2, $spk2) = split;
  open(SPK_VECTOR, "<", "exp/xvectors_$test_dir/spk_xvector.scp") or die "Could not open exp/xvectors_$test_dir/spk_xvector.scp";
  while (<SPK_VECTOR>) {
      chomp;
      my ($spk1,$tmp) = split;      
      my $target = "nontarget";
      if ($spk1 eq $spk2) {
        $target = "target";
      }
      print TRIALS "$spk1 $utt_id2 $target\n";
  }
  close(SPK_VECTOR) or die;
}  

close(UTT2SPK) or die;
close(TRIALS) or die;

#system "rm $test_dir/utt2spk_tmp;";
system "sort -u $out_dir > $out_dir_tmp;";
system "rm $out_dir;";
system "mv $out_dir_tmp $out_dir;";
system "echo $0: Successfully create trials file in $out_dir";
