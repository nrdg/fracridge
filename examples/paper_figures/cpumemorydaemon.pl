#!/usr/bin/perl -w

# call like this:
#   perl cpumemorydaemon.pl <pid>
#
# every so often, check memory usage.

use strict;

if (scalar(@ARGV) < 1) {
  die "bad args, so doing nothing\n";
}

my $pid = $ARGV[0];
my $timesleep = 0.01;

my $result1;
my $result2;
my @matches;
my $word = '\S+?\s+?';
while (1) {
  $result1 = `date +%s.%N`;
  chomp($result1);

#   $result2 = `ps u | grep $pid | grep MATLAB`;
#   @matches = ($result2 =~ m|^.+?(\d+?)\s+?([\d\.]+?)\s+?([\d\.]+?)\s+?.+$|s);
#
# ps -o vsz= 4422

  $result2 = `top -b -n 1 -p $pid`;# | grep $pid`;
#  @matches = ($result2 =~ m|^\s*?$word$word$word$word$word$word$word$word([\d\.]+?)\s+?([\d\.]+?)\s+?.+$|s);
  @matches = ($result2 =~ m|^\s*?$pid\s+?$word(.+?)\s.+$|m);

#  $result2 = `ps --no-headers -p $pid -o rss`;
#  @matches = ($result2 =~ m|^([\d\.]+?)$|s);
  
  if ($matches[0] eq "") {
    $matches[0] = "NaN";
  }
#  if ($matches[1] eq "") {
#    $matches[1] = "NaN";
#  }
#  print "$result1 $matches[0] $matches[1]\n";

  print "$result1 $matches[0]\n";
  sleep($timesleep);
}
