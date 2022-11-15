import torch

class TextProcess:
	def __init__(self):
		char_map_str = """
		<SPACE> 0
		a 1
		b 2
		c 3
		d 4
		e 5
		f 6
		g 7
		h 8
		i 9
		j 10
		k 11
		l 12
		m 13
		n 14
		o 15
		p 16
		q 17
		r 18
		s 19
		t 20
		u 21
		v 22
		w 23
		x 24
		y 25
		z 26
		"""
		self.char_map = {}
		self.index_map = {}
		for line in char_map_str.strip().split('\n'):
			ch, index = line.split()
			self.char_map[ch] = int(index)
			self.index_map[int(index)] = ch
		self.index_map[0] = ' '
		
	def text_to_int_seq(self, text):
		"""Convert text to an integer sequence"""
		int_seq = []
		for c in text:
			if c == ' ':
				ch = self.char_map['<SPACE>']
			else:
				ch = self.char_map[c]
			int_seq.append(ch)
		return int_seq
	
	def int_to_text_seq(self, labels):
		"""Convert a sequence of integer labels to text"""
		text = []
		for i in labels:
			text.append(self.index_map[i])
		return ''.join(text).replace('<SPACE>', ' ')