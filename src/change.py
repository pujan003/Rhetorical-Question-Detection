from sys import argv
with open(argv[1],'r') as f: 
	for line in f:
		a = line.strip().split()
		b = [a[0]]
		c = a[1:]
		for i in range(0,len(c)):
			m,n = c[i].split(':')
			if(int(m) < 9):
				n = '1.0000000000000000'
			c[i] = m+":"+n
		b.extend(c)
		a = ' '.join(b)
		print a 
