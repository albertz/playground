use Digest::MD5 md5;
use Data::Dumper::Perltidy;

my $data = "x";
my $digest = md5($data);
print Dumper $digest;

print map { sprintf '%02x', ord($_) } split(//, $digest);
print "\n";

print Dumper +(map(ord, split(//, $digest)))[0..5];
print Dumper +(map(ord, split(//, md5($data))))[0..5];

#@digest = map(ord, split(//, $digest));
#print Dumper @digest;
#print Dumper @digest[0..5];

#print $digest;
