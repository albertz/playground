#!/usr/bin/python -u

import socket, struct
from pprint import pprint
import sys, urlparse
import time

url = "udp://@239.254.18.1:3000"
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


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Multicast code from: http://wiki.python.org/moin/UdpCommunication

# Set some options to make it multicast-friendly
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
	sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
except AttributeError:
	pass # Some systems don't support SO_REUSEPORT
sock.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_TTL, 20)
sock.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_LOOP, 1)


sock.bind( ("", port) ) # standard SAP

# Set some more multicast options
intf = socket.gethostbyname(socket.gethostname())
sock.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_IF, socket.inet_aton(intf))
sock.setsockopt(socket.SOL_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(addr) + socket.inet_aton(intf))

lastTime = time.time()
dataCount = 0
try:
	while True:
		data, addr = sock.recvfrom(1024*10)
		sys.stdout.write(struct.pack("L", len(data)))
		curTime = time.time()
		sys.stdout.write(struct.pack("f", curTime - lastTime))
		lastTime = curTime
		sys.stdout.write(data)
		dataCount += len(data)
		
except KeyboardInterrupt:
	print >>sys.stderr
	print >>sys.stderr, "---"
	print >>sys.stderr, "dumped", dataCount/1024, "kB"
	