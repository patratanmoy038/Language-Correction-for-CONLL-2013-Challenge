def prep_train(vocab, max_seq_length, source_file, target_file):

	# start tag has index 0
	assert(vocab['<S>']==0)

	f1 = open(source_file)
	f2 = open(target_file)
	g1 = f1.read().splitlines()
	g2 = f2.read().splitlines()
	f1.close()
	f2.close()

	dat_x = []
	dat_y = []
	# assuming input is pre-tokenized and space separated
	for x,y in zip(g1,g2):
		# ignore if too long
		if len(x.split(' '))>max_seq_length or len(y.split(' '))>max_seq_length:
			continue

		cur_x = []
		cur_y = []
		for token in x.split(' '):
			if token in vocab:
				cur_x.append(vocab[token])
			else:
				cur_x.append(vocab['UNK'])

		for token in y.split(' '):
			if token in vocab:
				cur_y.append(vocab[token])
			else:
				cur_y.append(vocab['UNK'])

		dat_x.append(cur_x + [vocab['</S>']]*(max_seq_length-len(cur_x)))
		dat_y.append(cur_y + [vocab['</S>']]*(max_seq_length-len(cur_y)))

	return dat_x, dat_y


