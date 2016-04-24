fo = open('train_for_brown.txt','w')

def clean(x):
	for ch in ['%','&','-1','-2']:
		if ch in x:
			x=x.replace(ch," ")
	return x


with open("train.txt",'r') as f: 
	for line in f:
		dialog = ''.join(line.strip().split(',')[1:])
		dialog = " ".join(clean(dialog).split())
		dialog = dialog.replace("?", " ?")
		dialog = dialog.replace("."," .")
		fo.write(dialog + '\n')

fo.close()