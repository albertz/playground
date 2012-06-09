#!/usr/bin/python

# Session Announcement Protocol client

import better_exchook
better_exchook.install()

import socket, struct
from pprint import pprint

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

sock.bind( ("", 9875) ) # standard SAP

# Set some more multicast options
intf = socket.gethostbyname(socket.gethostname())
addr = "224.2.127.254"
sock.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_IF, socket.inet_aton(intf))
sock.setsockopt(socket.SOL_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(addr) + socket.inet_aton(intf))


class SDP:
	def __init__(self, data):
		self._data = data
		self.a = {}
		for l in data.splitlines():
			if len(l) <= 2: continue
			if l[1] != "=": continue
			attr = l[0]
			value = l[2:]
			if attr == "a":
				try:
					aattr,value = value.split(":",1)
					self.a[aattr] = value
				except:
					self.a[value] = True
			else:
				setattr(self, attr, value)
	def get(self, attr, default = None):
		return getattr(self, attr, default)
	def __repr__(self):
		return "SDP('" + self._data + "')"

class SAPPacket:
	def __init__(self, data, addr):
		header1 = ord(data[0])
		header1 = bin(header1)[2:].rjust(8, '0')
		header1 = map(int, header1)
		self.Version = header1[0] * 4 + header1[1] * 2 + header1[2] * 1
		self.IsIPv6 = bool(header1[3])
		self.R = header1[4]
		self.T = header1[5]
		self.IsEncrypted = bool(header1[6])
		self.IsCompressed = bool(header1[7])
		self.AuthLen = ord(data[1])
		self.MsgIdHash = struct.unpack("H", data[2:4])
		data = data[4:]
		if self.IsIPv6:
			self.OrigSource = data[:4*4]
			data = data[4*4:]
		else:
			self.OrigSource = data[:4]
			data = data[4:]
		# ignore AuthData
		data = data[self.AuthLen * 4:]
		if self.IsEncrypted:
			# ignore timestamp for now...
			data = data[4:]
		# search 0-byte, set payload type
		self.PayloadType = None
		for i in xrange(0, len(data)):
			if ord(data[i]) == 0:
				self.PayloadType = data[0:i]
				data = data[i+1:]
				break
		self.PayloadData = data
		self.Payload = SDP(data)
		
		try:
			mediaAddr = self.Payload.c.split()[-1].split("/")[0]
		except:
			mediaAddr = None
		try:
			mediaPort = self.Payload.m.split()[1]
		except:
			mediaPort = None
		if mediaAddr and mediaPort:
			self.Media = {}
			self.Media["Sender"] = self.Payload.s
			self.Media["Playgroup"] = self.Payload.a.get("x-plgroup", "")
			self.Media["Addr"] = "udp://@" + mediaAddr + ":" + mediaPort

playgroups = {}
count = 0

try:
	while True:
		data, addr = sock.recvfrom(1024)
		#print "received:", data, addr
		p = SAPPacket(data,addr)
		if hasattr(p, "Media"):
			senders = playgroups.get(p.Media["Playgroup"], {})		
			playgroups[p.Media["Playgroup"]] = senders
			if p.Media["Sender"] not in senders:
				count += 1
				print count, p.Media
			elif senders[p.Media["Sender"]] != p.Media["Addr"]:
				print count, p.Media
			senders[p.Media["Sender"]] = p.Media["Addr"]

except KeyboardInterrupt:
	print
	print "---"
	pprint(playgroups)
	
