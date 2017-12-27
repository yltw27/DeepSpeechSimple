import tensorflow as tf
import scipy.io.wavfile as wav
import os
import numpy as np
from python_speech_features import mfcc
from datetime import datetime

ckpt_folder = ''
meta = ".meta"

n_input = 26
n_context = 5
n_hidden_1 = n_hidden_2 = n_cell_dim = n_hidden_5 = 128
n_hidden_3 = n_cell_dim*2
n_hidden_6 = 29
default_stddev = 0.046875
relu_clip = 20.0


def variable_on_worker_level(name, shape, initializer):
    var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var
    

def BiRNN(batch_x, seq_length, dropout):
    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    batch_x_shape = tf.shape(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x, [-1, n_input + 2*n_input*n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.

    # 1st layer
    b1 = variable_on_worker_level('b1', [n_hidden_1], tf.random_normal_initializer(stddev=default_stddev))
    h1 = variable_on_worker_level('h1', [n_input + 2*n_input*n_context, n_hidden_1], tf.contrib.layers.xavier_initializer(uniform=False))
    layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
    layer_1 = tf.nn.dropout(layer_1, (1.0))

    # 2nd layer
    b2 = variable_on_worker_level('b2', [n_hidden_2], tf.random_normal_initializer(stddev=default_stddev))
    h2 = variable_on_worker_level('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(default_stddev))
    layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), relu_clip)
    layer_2 = tf.nn.dropout(layer_2, (1.0))

    # 3rd layer
    b3 = variable_on_worker_level('b3', [n_hidden_3], tf.random_normal_initializer(stddev=default_stddev))
    h3 = variable_on_worker_level('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=default_stddev))
    layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), relu_clip)
    layer_3 = tf.nn.dropout(layer_3, (1.0))

    # Now we create the forward and backward LSTM units.
    # Both of which have inputs of length `n_cell_dim` and bias `1.0` for the forget gate of the LSTM.

    # Forward direction cell: (if else required for TF 1.0 and 1.1 compat)
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True,   
                   reuse=tf.get_variable_scope().reuse)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                input_keep_prob=1.0,
                                                output_keep_prob=1.0,
                                                seed=4567)
    # Backward direction cell: (if else required for TF 1.0 and 1.1 compat)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True, 
                   reuse=tf.get_variable_scope().reuse)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                input_keep_prob=1.0,
                                                output_keep_prob=1.0,
                                                seed=4567)

    # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
    # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
    layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])

    # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                             cell_bw=lstm_bw_cell,
                                                             inputs=layer_3,
                                                             dtype=tf.float32,
                                                             time_major=True,
                                                             sequence_length=seq_length)

    # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
    # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
    outputs = tf.concat(outputs, 2)
    outputs = tf.reshape(outputs, [-1, 2*n_cell_dim])

    # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    b5 = variable_on_worker_level('b5', [n_hidden_5], tf.random_normal_initializer(stddev=default_stddev))
    h5 = variable_on_worker_level('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=default_stddev))
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
    layer_5 = tf.nn.dropout(layer_5, (1.0))

    # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
    # creating `n_classes` dimensional vectors, the logits.
    b6 = variable_on_worker_level('b6', [n_hidden_6], tf.random_normal_initializer(stddev=default_stddev))
    h6 = variable_on_worker_level('h6', [n_hidden_5, n_hidden_6], tf.contrib.layers.xavier_initializer(uniform=False))
    layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6], name="logits")

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6
    

def audiofile_to_input_vector(audio, fs, numcep, numcontext):
    # Get mfcc coefficients
    features = mfcc(audio, samplerate=fs, numcep=numcep)

    # We only keep every second feature (BiRNN stride = 2)
    features = features[::2]

    # One stride per time step in the input
    num_strides = len(features)

    # Add empty initial and final contexts
    empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
    features = np.concatenate((empty_context, features, empty_context))

    # Create a view into the array with overlapping strides of size
    # numcontext (past) + 1 (present) + numcontext (future)
    window_size = 2*numcontext+1
    train_inputs = np.lib.stride_tricks.as_strided(
            features,
            (num_strides, window_size, numcep),
            (features.strides[0], features.strides[0], features.strides[1]),
            writeable=False)

        # Flatten the second and third dimensions
    train_inputs = np.reshape(train_inputs, [num_strides, -1])

    # Whiten inputs (TODO: Should we whiten?)
    # Copy the strided array so that we can write to it safely
    train_inputs = np.copy(train_inputs)
    train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)

    # Return results
    return train_inputs
    

def decode_with_lm(inputs, sequence_length, beam_width=100,
                   top_paths=1, merge_repeated=True):
    custom_op_module = tf.load_op_library('native_client/libctc_decoder_with_kenlm.so')
    decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = (custom_op_module.ctc_beam_search_decoder_with_lm(
                                                                    inputs, sequence_length, beam_width=beam_width,
                                                                    model_path='data/lm/lm.binary', 
                                                                    trie_path='data/lm/trie', 
                                                                    alphabet_path='data/alphabet.txt',
                                                                    lm_weight=2.15, 
                                                                    word_count_weight=-0.10,
                                                                    valid_word_count_weight=1.10,
                                                                    top_paths=top_paths, 
                                                                    merge_repeated=merge_repeated))

    return [tf.SparseTensor(ix, val, shape) for (ix, val, shape) in zip(decoded_ixs, decoded_vals, decoded_shapes)]
      
      
def create_inference_graph(batch_size=None):
    # Input tensor will be of shape [batch_size, n_steps, n_input + 2*n_input*n_context]
    input_tensor = tf.placeholder(tf.float32, [batch_size, None, n_input + 2*n_input*n_context], name='input_node')
    seq_length = tf.placeholder(tf.int32, [batch_size], name='input_lengths')

    # Calculate the logits of the batch using BiRNN
    logits = BiRNN(input_tensor, None, [0.0 for i in range(6)])

    # Beam search decode the batch
    decoder = tf.nn.ctc_beam_search_decoder

    decoded, _ = decoder(logits, seq_length, merge_repeated=True, beam_width=100)
    y = tf.sparse_to_dense(tf.to_int32(decoded[0].indices), tf.to_int32(decoded[0].dense_shape), tf.to_int32(decoded[0].values), name='output_node')

    return (
        {
            'input': input_tensor,
            'input_lengths': seq_length,
        },
        {
            'outputs': y,
        }
    )
    
    
def do_single_file_inference(input_file_path):
    fs, audio = wav.read(input_file_path)
    mfcc = audiofile_to_input_vector(audio=audio, fs=fs, numcep=26, numcontext=n_context)

    output = session.run(outputs['outputs'], feed_dict = {
        inputs['input']: [mfcc],
        inputs['input_lengths']: [len(mfcc)],
        })

    result = ''
    for w in output[0]:  #[0]:
        idx = w + 96
        if idx == 96:
            idx = 32
        result += chr(idx)
    print('+96 result: {}'.format(result))  
    
        
if __name__ == '__main__':
    a = datetime.now()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        inputs, outputs = create_inference_graph(batch_size=1)

        # Create a saver using variables from the above newly created graph
        saver = tf.train.import_meta_graph(ckpt_folder + meta, clear_devices=True)

        # Restore variables from training checkpoint
        checkpoint = tf.train.get_checkpoint_state(ckpt_folder)
        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)        

        print('\nDuration of restoring the graph: {}'.format(datetime.now()-a))
        
        test_num = int(input('Please enter test number: '))
        for i in range(test_num):
            os.system('arecord -d 5 -r 16000 -f S16_LE ./test.wav')
            start = datetime.now()
            do_single_file_inference('./test.wav')
            print('Inference time: {}\n{}'.format(datetime.now() - start, '-' * 50))
    
