#!/usr/bin/python -u

import better_exchook
better_exchook.install()

import socket, struct
from pprint import pprint
import sys, urlparse
import time

url = "udp://@239.254.42.1:3000"
if len(sys.argv) > 1: url = sys.argv[1]
url = urlparse.urlparse(url)
if url.hostname[0:1] == "@":
	# remove the "@" at the beginning
	url = list(url)
	url[1] = url[1][1:]
	# reparse
	url = urlparse.urlunparse(url)
	url = urlparse.urlparse(url)	
assert url.scheme == "udp"
addr = url.hostname
port = url.port

# from http://wiki.python.org/moin/UdpCommunication
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ttl = 1
s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)

while True:
	dataLen, = struct.unpack("L", sys.stdin.read(8))
	timeDelay, = struct.unpack("f", sys.stdin.read(4))
	data = sys.stdin.read(dataLen)
	s.sendto(data, (addr, port))
	time.sleep(timeDelay)
	