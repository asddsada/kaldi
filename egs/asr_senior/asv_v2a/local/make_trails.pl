#!/usr/bin/perl
#
# Usage: make_trails.pl data/test/

if (@ARGV != 1) {
  print STDERR "Usage: $0 <path-to-data-dir>/<test-dir>/\n";
  print STDERR "e.g. $0 data/test/\n";
  exit(1);
}

($test_dir) = @ARGV;

system "cp $test_dir/utt2spk $test_dir/utt2spk_tmp;";
system "echo $0: Creating trails file in $test_dir";

open(UTT2SPK, "<", "$test_dir/utt2spk") or die "Could not open $test_dir/utt2spk";
open(TRAILS, ">", "$test_dir/trails") or die "Could not open $test_dir/utt2spk";

while (<UTT2SPK>) {
  chomp;
  my ($utt_id1, $spk1) = split;
  open(TEMP, "<", "$test_dir/utt2spk_tmp") or die "Could not open $test_dir/utt2spk.tmp";
  while (<TEMP>) {
      chomp;
      my ($utt_id2, $spk2) = split;      
      my $target = "nontarget";
      if ($spk1 eq $spk2) {
        $target = "target";
      }
      print TRAILS "$utt_id1 $utt_id2 $target\n";
  }
  close(TEMP) or die;
}  

close(UTT2SPK) or die;
close(TRAILS) or die;

system "rm $test_dir/utt2spk_tmp;";
system "sort -u $test_dir/trails > $test_dir/trails_tmp;";
system "rm $test_dir/trails;";
system "mv $test_dir/trails_tmp $test_dir/trails;";
system "echo $0: Successfully create trails file in $test_dir;";