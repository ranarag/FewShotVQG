import torch.nn as nn

from .base_rnn import BaseRNN
import torch

class EncoderRNN(BaseRNN):
    """Applies a multi-layer RNN to an input sequence.

    Inputs: inputs, input_lengths
        - **inputs**: List of sequences, whose length is the batch size
            and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): List that contains
            the lengths of sequences in the mini-batch, it must be
            provided when using variable length RNN (default: `None`).

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): Tensor containing the
            encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): Tensor
            containing the features in the hidden state `h`

    Examples::
         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)
    """

    def __init__(self, vocab_size, max_len, hidden_size, ques_encoder=False,
                 input_dropout_p=0.2, dropout_p=0.2, n_layers=1,
                 bidirectional=False, rnn_cell='lstm', variable_lengths=False, embedding=None):
        """Constructor for EncoderRNN.

        Args:
            vocab_size (int): Size of the vocabulary.
            max_len (int): A maximum allowed length for the sequence to be
                processed.
            hidden_size (int): The number of features in the hidden state `h`.
            input_dropout_p (float, optional): Dropout probability for the input
                sequence (default: 0).
            dropout_p (float, optional): Dropout probability for the output
                sequence (default: 0).
            n_layers (int, optional): Number of recurrent layers (default: 1).
            bidirectional (bool, optional): if True, becomes a bidirectional
                encoder (defulat False).
            rnn_cell (str, optional): Type of RNN cell (default: gru).
            variable_lengths (bool, optional): If use variable length
                RNN (default: False).
        """
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)
        self.variable_lengths = False
#         print(vocab_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding, requires_grad=False)


        self.hidden_dim = hidden_size

        self.vocab_size = vocab_size
        self.lstm = nn.LSTMCell(768, 768, 1)
#                                  batch_first=True, bidirectional=bidirectional,
#                                  dropout=dropout_p)
#         self.rnn = nn.LSTMCell(hidden_size, hidden_size)
        self.use_tf = True
        self.init_weights()
#         batch_size = 30
#         self.deep_output = nn.Linear(hidden_size, hidden_size)
    def init_weights(self):
        """Initialize weights.
        """
        # self.embedding.weight.data.uniform_(-0.1, 0.1)
        nn.init.kaiming_uniform_(self.embedding.weight,nonlinearity='relu')

    def init_hidden(self, batch_size):
        
        # initialize the hidden state and the cell state to zeros
        return (torch.zeros(batch_size, self.hidden_dim).cuda(),
                torch.zeros(batch_size, self.hidden_dim).cuda())
        
        
    def forward_rnn(self, captions):

        max_timespan = 4
        batch_size = captions.size(0)
    
        h, c = self.init_hidden(batch_size)
        prev_words = torch.zeros(batch_size, 1).long().cuda()
#         print(captions.size())
#         exit()
#         embedding = torch.zeros(batch_size, 768)
#         embedding = self.embedding(captions)
#         print(self.embedding)
#         print(self.embedding(0))
#         if self.use_tf:
        embedding = self.embedding(captions) 
#     if self.training else self.embedding(prev_words)
#         else:
#             embedding = self.embedding(prev_words)
#         print("OKAY")
#         print(embedding.size())
#         exit()
#         print(batch_size, max_timespan, self.hidden_dim)
        preds = torch.zeros(batch_size, max_timespan, self.hidden_dim).cuda()

        for t in range(max_timespan):
 

            lstm_input = embedding[:, t, :]
#             else:
#                 embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
#                 lstm_input = embedding

            h, c = self.lstm(lstm_input, (h,c))
#             ho = h[0]
#             output = self.deep_output(h)

            preds[:, t, :] = h
#             if not self.training or not self.use_tf:
#                 embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1))
        return preds, h

    def forward(self, input_var, input_lengths=None, h0=None):
        """Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): Tensor containing the features of
                the input sequence.
            input_lengths (list of int, optional): A list that contains
                the lengths of sequences in the mini-batch.
            h0 : Tensor containing initial hidden state.

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): Variable containing
                the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size):
                Variable containing the features in the hidden state h
        """
#         embedded = self.embedding(input_var.long())
#         embedded = self.input_dropout(embedded)
        
#         if self.variable_lengths:
#             embedded = nn.utils.rnn.pack_padded_sequence(
#                     embedded, input_lengths, batch_first=True, enforce_sorted=False)
#         max_timespan = max([len(caption) for caption in input_var]) - 1
        output, hidden = self.forward_rnn(input_var)
#         print("FORWARD DONE")
#         if self.variable_lengths:
#             output, _ = nn.utils.rnn.pad_packed_sequence(
#                     output, batch_first=True)
        return output, hidden
