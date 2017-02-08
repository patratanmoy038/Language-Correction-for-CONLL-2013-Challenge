import tensorflow as tf
import numpy as np
import cPickle as cp
import tempfile
import pdb
import sys

import prep_data

sess = tf.InteractiveSession()

vocab_path = 'vocab.pkl'
embedding_path = 'embeddings.pkl'
train_source_file = 'text_files/source_train.txt'
train_target_file = 'text_files/target_train.txt'
model_path = '/scratch/chemical/dual/ch7130186/Assignment_2/models/'
model_name = 'mem_300_GRU_emb_autoenc'

with open(vocab_path) as f:
	vocab = cp.load(f)
with open(embedding_path) as f:
	w2v_embeddings = cp.load(f)

# specifications
max_seq_length = 40
batch_size = 128

vocab_size = len(vocab)
embedding_dim = w2v_embeddings.shape[1]

memory_dim = 300

print("vocab_dim     : " + str((vocab_size, embedding_dim)))
print("embedding dim : " + str((w2v_embeddings.shape)))

# structures
pred = tf.placeholder(tf.bool)
enc_inp = [tf.placeholder(tf.int32, shape=(None,), name="inp%i" % t) for t in range(max_seq_length)]
labels = [tf.placeholder(tf.int32, shape=(None,), name="labels%i"%t) for t in range(max_seq_length)]
weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]
dec_inp = [tf.zeros_like(enc_inp[0], dtype=np.int32, name="START")] + labels[:-1]	# ensure <S> token is 0th in vocab
prev_mem = tf.zeros((batch_size, memory_dim))

# model
cell = tf.nn.rnn_cell.GRUCell(memory_dim)
dec_outputs, dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_dim, feed_previous=pred)

# loss
loss = tf.nn.seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)
tf.scalar_summary("loss", loss)
summary_op = tf.merge_all_summaries()

# optimizer
learning_rate = 0.001
momentum = 0.9
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

logdir = tempfile.mkdtemp()
print logdir
summary_writer = tf.train.SummaryWriter(logdir, sess.graph_def)

sess.run(tf.initialize_all_variables())

# set embeddings
for emb in [v for v in tf.trainable_variables() if "embedding:" in v.name]:
	sess.run(emb.assign(w2v_embeddings))

# get train data
# X_train dimension is num_examples * max_seq_length
print("Loading Data")
X_train, Y_train = prep_data.prep_train(vocab, max_seq_length, train_source_file, train_target_file)
print('Total training examples : ' + str(len(X_train)))

def train_batch(batch_num, batch_size):
    X = X_train[batch_num*batch_size:(batch_num+1)*batch_size]
    Y = Y_train[batch_num*batch_size:(batch_num+1)*batch_size]
    
    # Dimshuffle to max_seq_len * batch_size
    X = np.array(X).T
    Y = np.array(Y).T

    feed_dict = {enc_inp[t]: X[t] for t in range(max_seq_length)}
    feed_dict.update({labels[t]: X[t] for t in range(max_seq_length)})
    feed_dict.update({pred:False})

    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    return loss_t, summary

training_epochs = 100
saver = tf.train.Saver()

print("Starting training!")
for epoch in range(training_epochs):
	avg_loss = 0.
	total_batch = int(len(X_train)/batch_size)
	# Loop over all batches
	for i in range(total_batch):
		loss_t, summary = train_batch(i, batch_size)	
		summary_writer.add_summary(summary, t)
		avg_loss += loss_t/total_batch

	# Display after each epoch
	print("Epoch : " + '%03d' % (epoch+1) + " loss = " + "{:.9f}".format(avg_loss))
	sys.stdout.flush()
	save_path = saver.save(sess, model_path + model_name + '_' + str(epoch) + '_' + "{:.9f}".format(avg_loss) + '.ckpt')

print("Optimization done!")
summary_writer.flush()

# save model




# TEST
# X_batch = [np.random.choice(vocab_size, size=(max_seq_length,), replace=False)
#            for _ in range(10)]
# X_batch = np.array(X_batch).T
# 
# feed_dict = {enc_inp[t]: X_batch[t] for t in range(max_seq_length)}
# feed_dict.update({pred:True})
# dec_outputs_batch = sess.run(dec_outputs, feed_dict)
# 
# pdb.set_trace()
