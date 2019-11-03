#!/usr/bin/perl
#
# Usage: make_trials.pl data/test/

if (@ARGV != 1) {
  print STDERR "Usage: $0 <path-to-data-dir>/<test-dir>/\n";
  print STDERR "e.g. $0 data/test/\n";
  exit(1);
}

($test_dir) = @ARGV;

#system "cp $test_dir/utt2spk $test_dir/utt2spk_tmp;";
system "echo $0: Creating trials file in $test_dir";

open(UTT2SPK, "<", "$test_dir/utt2spk") or die "Could not open $test_dir/utt2spk";
open(TRIALS, ">", "$test_dir/trials") or die "Could not open $test_dir/utt2spk";

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
system "sort -u $test_dir/trials > $test_dir/trials_tmp;";
system "rm $test_dir/trials;";
system "mv $test_dir/trials_tmp $test_dir/trials;";
system "echo $0: Successfully create trials file in $test_dir;";
