import sys
import os
import math
import time

import numpy as np
import tensorflow as tf

from utils import batch_norm, sequence_mask, highway, CE

#Define the class for the Language Model 
class LM(object):
    def __init__(self, options, session, inputs, training=True):
        self._options = options
        self._session = session
        self._training = training
        if self._options.reps[0]:
            self._examples = inputs.pop(0)
        if self._options.reps[1]:
            self._examplesChar = inputs.pop(0)
        if self._options.reps[2]:
            self._labels = inputs.pop(0)
        if self._options.reps[3]:
            self._labelsChar = inputs.pop(0)
        self.build_graph()

    # TODO: Add possibility of horizontal concatenation ? Possibly inside of CE - same number of parameters, just reshape the filters ?
    # TODO: process version feedforward
    def process_seq(self):
        # Getting input embeddings from inputs 
        if self._options.reps[0]:
            self._examples = tf.cast(tf.verify_tensor_all_finite(tf.cast(self._examples, 'float32'), 'Nan'), 'int64')

            self._wordemb = tf.get_variable(
                name='wordemb',
                shape=[self._options.vocab_size, self._options.w_emb_dim],
                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.vocab_size))))
            
            w_embeddings = tf.nn.embedding_lookup(self._wordemb, tf.reshape(self._examples, [-1]))
            w_input_emb = tf.reshape(w_embeddings, [self._options.batch_size, self._options.max_seq_length, self._options.w_emb_dim])

            mask = sequence_mask(self._examples)
        if self._options.reps[1]:
            self._charemb = tf.get_variable(
                name='charemb',
                shape=[self._options.char_vocab_size, self._options.c_emb_dim],
                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.char_vocab_size))))

            c_embeddings = tf.nn.embedding_lookup(self._charemb, tf.reshape(self._examplesChar, [-1]))
            self._convfilters = []
            for w, d in zip(self._options.window_sizes, self._options.filter_dims):
                self._convfilters.append(
                    tf.get_variable(
                        name='filter%d' % w,
                        shape=[w, self._options.c_emb_dim, d],
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(w * self._options.c_emb_dim)))
                    )
                )                
            c_input_emb = CE(tf.reshape(c_embeddings, [self._options.batch_size, self._options.max_seq_length, self._options.max_word_length, self._options.c_emb_dim]),
                             self._convfilters)
            
            if not self._options.reps[0]:
                mask = sequence_mask(self._examplesChar, char=True)
        if self._options.reps[0] and not self._options.reps[1]:
            input_emb = w_input_emb
        elif self._options.reps[1] and not self._options.reps[0]:
            input_emb = c_input_emb
        elif self._options.reps[0] and self._options.reps[1]:
            input_emb = tf.concat(2, [w_input_emb, c_input_emb])

        input_emb = tf.verify_tensor_all_finite(input_emb, 'Nan')

        # Batch normalization
        if self._options.batch_norm:
            self.batch_normalizer = batch_norm()
            input_emb = self.batch_normalizer(input_emb, self._training)

        # Highway Layer
        if self._options.highway_layers > 0:
            self._highway_w = []
            self._highway_wg = []
            self._highway_b = []
            self._highway_bg = []
            for i in range(self._options.highway_layers):                
                self._highway_w.append(
                    tf.get_variable(
                        name='highway_w%d' % i,
                        shape=[self._options.emb_dim] * 2,
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.w_emb_dim)))))
                self._highway_b.append(
                    tf.get_variable(
                        name='highway_b%d' % i,
                        shape=[self._options.emb_dim],
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.w_emb_dim)))))
                self._highway_wg.append(
                    tf.get_variable(
                        name='highway_wg%d' % i,
                        shape=[self._options.emb_dim] * 2,
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.w_emb_dim)))))
                self._highway_bg.append(
                    tf.get_variable(
                        name='highway_bg%d' % i,
                        shape=[self._options.emb_dim],
                        initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.w_emb_dim)))))
            input_emb = tf.reshape(highway(tf.reshape(input_emb,
                                                      [-1, self._options.emb_dim]),
                                           self._highway_w,
                                           self._highway_b,
                                           self._highway_wg,
                                           self._highway_bg),
                                   [self._options.batch_size, self._options.max_seq_length, self._options.emb_dim])
            
        # LSTM
        self.cell = tf.nn.rnn_cell.LSTMCell(self._options.hidden_dim, state_is_tuple = False, activation=tf.nn.relu)
        if self._training and self._options.dropout < 1.0:
            self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self._options.dropout )
        if self._options.hidden_layers > 1:
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * self._options.hidden_layers)
        hidden, _ = tf.nn.dynamic_rnn(self.cell,
                                      input_emb,
                                      sequence_length= tf.reduce_sum(mask, 1),
                                      dtype='float32')

        hidden = tf.verify_tensor_all_finite(hidden, 'Nan')
        return mask, hidden

    def process_output_seq(self, allVoc=True, w_labels=None, c_labels=None):    

        if self._options.reps[2]:
            self._output_wordemb = tf.get_variable(
                name="output_wordemb",
                shape= [self._options.hidden_dim, self._options.vocab_size],
                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.hidden_dim)) ))
            if allVoc:
                return self._output_wordemb
            else:
                w_output_embeddings = tf.nn.embedding_lookup(self._wordemb, w_labels) 
            
        if self._options.reps[3]:            
            if allVoc:
                print "Error: Can't use a non-sampling objective with character-based output representation - it would imply processing the whole vocabulary for each batch"
                sys.exit()

            if self._options.reps[1] and self._options.reuse_character_layer:
                c_output_embeddings = tf.nn.embedding_lookup(self._charemb, c_labels)
                c_output_emb = CE(tf.reshape(c_output_embeddings, [self._options.batch_size, self._options.max_seq_length, self._options.max_word_length, self._options.c_emb_dim]),
                             self._convfilters)
            else:
                self._output_charemb = tf.get_variable(
                    name='output_charemb',
                    shape=[self._options.char_vocab_size, self._options.c_emb_dim],
                    initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(self._options.char_vocab_size))))
                
                c_output_embeddings = tf.nn.embedding_lookup(self._output_charemb, c_labels) 
                self._output_convfilters = []
                for w, d in zip(self._options.window_sizes, self._options.filter_dims):
                    self._output_convfilters.append(
                        tf.get_variable(
                            name='output_filter%d' % w,
                            shape=[w, self._options.c_emb_dim, d],
                            initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(w * self._options.c_emb_dim)))
                        )
                    )
                c_output_emb = CE(tf.reshape(c_output_embeddings, [self._options.batch_size, self._options.max_seq_length, self._options.max_word_length, self._options.c_emb_dim]),
                                  self._output_convfilters)
            
        if self._options.reps[2] and not self._options.reps[3]:
            output_emb = w_output_embeddings
        elif self._options.reps[3] and not self._options.reps[2]:
            output_emb = c_output_emb
        elif self._options.reps[2] and self._options.reps[3]:
            output_emb = tf.concat(2, [w_output_embeddings, c_output_emb])

        return output_emb

    #TODO: Deal with length 0 sequences ! 
    def tryLoss_seq(self, hidden, mask, output):
        _hidden = tf.reshape(hidden, [-1, self._options.hidden_dim])
        _logits = tf.matmul(_hidden, output)
        _logits = tf.verify_tensor_all_finite(_logits, 'Nan')
        """
        #Without masking:
        logits = tf.reshape(_logits, [self._options.batch_size,self._options.max_seq_length, self._options.vocab_size])
        cross_entropy_seq = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self._labels)
        """
        #With masking :      
        _labels = tf.reshape(self._labels, [-1])
        _cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(_logits, _labels)
        _mask = tf.reshape(mask, [-1])
        _cross_entropy = tf.mul(_cross_entropy, tf.cast(_mask, dtype='float32'))
        length = tf.maximum(tf.reduce_sum(mask, 1), tf.ones(self._options.batch_size, dtype = 'int64'))
        cross_entropy_seq = tf.reduce_sum(tf.reshape(_cross_entropy,[self._options.batch_size, self._options.max_seq_length]), 1) / tf.cast(length, dtype='float32')

        loss = tf.reduce_mean(cross_entropy_seq)
        return loss

    # Pour sampled loss: ajouter juste pour les mots concernes la character representation, et des zeros ailleurs ? 


    def optimize(self, loss):
          self._lr = self._options.learning_rate
          optimizer = tf.train.AdamOptimizer(self._lr)
          """
          gvs = optimizer.compute_gradients(loss)          
          gvs = [(tf.verify_tensor_all_finite(grad, 'NaN with '+var.name), var) for grad, var in gvs]
          capped_gvs = [(tf.clip_by_value(grad,-0.1,0.1), var) for grad, var in gvs]
          train = optimizer.apply_gradients(capped_gvs)
          """
          train = optimizer.minimize(loss)
          self._train = train

    def build_graph(self):
        self._mask, self._hidden = self.process_seq()
        self._output_emb = self.process_output_seq()
        loss = self.tryLoss_seq(self._hidden, self._mask, self._output_emb)
        if self._training:
            self.optimize(loss)
        self._loss = loss

    def call(self):
        start_time = time.time()
        average_loss = 0
        if self._training:
            print('In training')
            n_steps = self._options.n_training_steps // self._options.training_sub
            op = self._train
            display = n_steps // self._options.display_step
            call = "Training:"
        else:
            n_steps = self._options.n_testing_steps
            op = tf.no_op()
            display = n_steps-1
            call = "Testing:"

        for step in xrange(n_steps):
            _, loss = self._session.run([op, self._loss])
            average_loss+= loss

        # Record monitored values                                                                                                                                                                           
            if self._training:
                if step % (display) == 0:
                    print(" %s Cross-entropy at batch %i : %.3f ; Computation speed : %.3f sec/batch" % ( call, step+1,
                                                                                                          loss, 
                                                                                                          (time.time() - start_time) / (step + 1) ))
            else:
                if step % (display) == 0 and step > 0:
                    print(" %s Perplexity, score and norm at batch %i : %.3f; Computation speed : %.3f sec/batch" % ( call, step+1,
                                                                                                                      average_loss/(step+1),
                                                                                                                      (time.time() - start_time) / (step + 1) ))
