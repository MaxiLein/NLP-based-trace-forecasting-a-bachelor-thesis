import numpy as np
import tensorflow as tf

# Reproducibility
tf.random.set_seed(42)


@tf.keras.saving.register_keras_serializable()
class InputTokeniser(tf.keras.layers.Layer):
    
    def __init__(self, prop_dim=1_000):
        super().__init__()
        # Input: (B, T)
        # With T's elements of format [identifier, [event-props], [trace-props]]
        # prop_dim = 1_000
        
        self.event_tokeniser = tf.keras.Sequential([
            tf.keras.layers.Dense(prop_dim * 4),
            tf.keras.layers.Dense(prop_dim * 0.5, activation='relu'),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1)),
        ])
        self.trace_tokeniser = tf.keras.Sequential([
            tf.keras.layers.Dense(prop_dim * 4),
            tf.keras.layers.Dense(prop_dim * 0.5, activation='relu'),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1)),
        ])
        
    def call(self, x):
        # x = [(identifier, event_props, trace_props)] = batches of tuples of sequences 
        # each tuple is one sequence
        # Tuple = (
            #   identifier: tf.Tensor: shape=(sequence_length,), dtype=int32, 
            #   event_props: tf.Tensor: shape=(sequence_length, prop_dim), dtype=float32, 
            #   trace_props: tf.Tensor: shape=(sequence_length, prop_dim), dtype=float32
            # )
        identifier, event_props, trace_props = zip(*x)
        event_props =  tf.cast(tf.convert_to_tensor(event_props), dtype=tf.float32) # (B, T, prop_dim)
        trace_props =  tf.cast(tf.convert_to_tensor(trace_props), dtype=tf.float32) # (B, T, prop_dim)
        
        identifier_token = tf.cast(tf.convert_to_tensor(identifier), dtype=tf.float32) # (B, T)
        event_token = self.event_tokeniser(event_props) # (B, T)
        trace_token = self.trace_tokeniser(trace_props) # (B, T)
        
        # Stack the tokens
        stacked = tf.stack([identifier_token, event_token, trace_token], axis=-1)
        B, T, _ = stacked.shape
        # Reshape to one sequence
        tokens = tf.reshape(stacked, [B, 3*T]) # (B, T, 3) -> (B, 3*T)

        return tokens
        
@tf.keras.saving.register_keras_serializable()
class SequenceEncoding(tf.keras.layers.Layer):
    
    def __init__(self, batch_size=40, vocab_size=100, embedding_dim=1_000, sequence_length=100):
        super().__init__()
        B = batch_size
        T = sequence_length
        C = embedding_dim
        self.token_i_emb = tf.keras.layers.Embedding(vocab_size, C) # (B, T) -> (B, T, C)
        self.token_e_emb = tf.keras.layers.Dense(T*C, use_bias=False) # (B, T) -> (B, T*C)
        self.token_t_emb = tf.keras.layers.Dense(T*C, use_bias=False) # (B, T) -> (B, T*C)
        
    def call(self, x):
        i = x[:,0::3] # (B, 3*T) -> (B, T)
        e = x[:,1::3] # (B, 3*T) -> (B, T)
        t = x[:,2::3] # (B, 3*T) -> (B, T)
        
        i = tf.cast(i, dtype=tf.int32) # (B, T) -> (B, T)
        i = self.token_i_emb(i) # (B, T) -> (B, T, C)
        B, T, C = i.shape
        
        e = self.token_e_emb(e) # (B, T) -> (B, T*C)
        t = self.token_t_emb(t) # (B, T) -> (B, T*C)
        # Reshape properties
        e = tf.reshape(e, (B,T,C))
        t = tf.reshape(t, (B,T,C))
        
        # Combine the embeddings back together, keeping the batch dimension and their old order
        stacked = tf.stack([i, e, t], axis=2) # (B, T, 3, C)
        # Reshape to interleave the tensors along the T dimension
        x = tf.reshape(stacked, (B, 3*T, C)) # (B, T, 3, C) -> (B, 3*T, C)
    
        # Add positional encoding
        x += SequenceEncoding.positional_encoding(3*T, C) #(B, 3*T, C) + (3*T, C) -> (B, T, C)
        return x
    
    @staticmethod
    def positional_encoding(sequence_length: int, dimension: int) -> tf.Tensor:
        """
        Generate positional encodings for a sequence.

        #### Args:
            sequence_length (int): The sequence_length of the sequence.
            dimension (int): The (Channel / Feature) dimensionality of the model's embeddings. Should be an even number.

        #### Returns:
            tf.Tensor: A positional encoding matrix suitable for use in a Transformer model.

        #### Example:
            >>> positional_encoding_matrix = positional_encoding(14, 64)
            tf.Tensor: shape=(2, 2), dtype=float32, numpy=array
            ([[0., 1.],
            [0.84147096, 0.5403023 ]],
            dtype=float32)
        """
        # Original paper uses sine and cosine functions of different frequencies
        # PE(pos,2i) = sin( pos/10000^ (2i/dmodel) )
        # PE(pos,2i+1) = cos( pos/10000^ (2i/dmodel) )
        # pos: position, i: dimension, dmodel: dimension of model

        # Check whether dimension is even
        assert dimension % 2 == 0, \
            f"Embedding-Dimension must be an even number, but was {dimension}."

        dimension = dimension // 2

        # Create an array of positions from 0 to sequence_length-1.
        positions = np.arange(sequence_length)[:, np.newaxis]  # (seq, 1)

        # Create an array of dimension values scaled by 1/dimension.
        dimensions = np.arange(dimension)[np.newaxis, :] / dimension  # (1, dimension)

        # Calculate the angle rates for the sine and cosine components.
        angle_rates = 1 / (10000 ** dimensions)  # (1, dimension)

        # Calculate the angle values for each position and dimension.
        angle_rads = positions * angle_rates  # (sequence_length, dimension)

        # Combine the sine and cosine components to create the positional encoding matrix.
        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1
        )

        # Cast the positional encoding to tf.float32.
        return tf.cast(pos_encoding, dtype=tf.float32)
     
@tf.keras.saving.register_keras_serializable()
class FeedForward(tf.keras.layers.Layer):
    
    def __init__(self, embedding_dim=1_000, ffwd_dim=None, dropout=0.15):
        super().__init__()
        # B = batch_size
        # T = sequence_length
        C = embedding_dim
        
        # ffwd_dim = 4 * C as in the paper (default)
        ffwd_dim = ffwd_dim or 4 * C
        
        self.ffwd = tf.keras.Sequential([
            tf.keras.layers.Dense(ffwd_dim, activation='relu'), # (B, T, C) -> (B, T, ffwd_dim)
            tf.keras.layers.Dense(C), # (B, T, ffwd_dim) -> (B, T, C)
            tf.keras.layers.Dropout(dropout), # (B, T, C) -> (B, T, C)
        ])
    
    def call(self, x):
        return self.ffwd(x) # (B, T, C) -> (B, T, C)
         
@tf.keras.saving.register_keras_serializable()
class DecoderLayer(tf.keras.layers.Layer):
    
    def __init__(self, embedding_dim, n_head, ffwd_dim=None, dropout=0.15):
        super().__init__()
        # B = batch_size
        # T = sequence_length
        C = embedding_dim
        
        # Split the embedding dimension into n_head parts
        # Since the result of the heads is concatenated, the resulting dimension is n_head * head_size = embedding_dim
        # embedding_dim -> n_head * head_size -> embedding_dim
        head_size = embedding_dim // n_head # head_size = key_dim
        
        self.norm_x = tf.keras.layers.LayerNormalization()
        self.mhsa = tf.keras.layers.MultiHeadAttention(
            num_heads=n_head,
            key_dim=head_size,
            dropout=dropout,
            use_bias=False,
        )
        self.add_x_mhsa = tf.keras.layers.Add()
        self.add_norm_mhsa = tf.keras.layers.Add()
        self.norm_mhsa = tf.keras.layers.LayerNormalization()
        self.ffwd = FeedForward(embedding_dim=embedding_dim, ffwd_dim=ffwd_dim, dropout=dropout)
        self.add_x_ffwd = tf.keras.layers.Add()
        self.add_norm_ffwd = tf.keras.layers.Add()
        self.norm_ffwd = tf.keras.layers.LayerNormalization()
        self.norm_residual = tf.keras.layers.LayerNormalization()
        self.add_x_residual = tf.keras.layers.Add()
        
    def call(self, x):
        
        norm_x = self.norm_x(x) # (B, T, C) -> (B, T, C)
        attention = self.mhsa(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            use_causal_mask=True) # (B, T, C) -> (B, T, C)
        resi_1 = self.add_x_mhsa([x, attention]) # (B, T, C) + (B, T, C) -> (B, T, C)
        resi_2 = self.add_norm_mhsa([attention, norm_x]) # (B, T, C) + (B, T, C) -> (B, T, C)
        norm_mhsa = self.norm_mhsa(resi_2) # (B, T, C) -> (B, T, C)
        forward = self.ffwd(norm_mhsa)
        resi_3 = self.add_x_ffwd([resi_1, forward]) # (B, T, C) + (B, T, C) -> (B, T, C)
        resi_4 = self.add_norm_ffwd([forward, norm_mhsa]) # (B, T, C) + (B, T, C) -> (B, T, C)
        norm_ffwd = self.norm_ffwd(resi_4) # (B, T, C) -> (B, T, C)
        norm_residual = self.norm_residual(resi_3) # (B, T, C) -> (B, T, C)
        output = self.add_x_residual([norm_residual, norm_ffwd]) # (B, T, C) + (B, T, C) -> (B, T, C)
        
        return output

@tf.keras.saving.register_keras_serializable()
class OutputProjection(tf.keras.layers.Layer):
    def __init__(self, prop_dim=1_000, vocab_size=100):
        super().__init__()
        # B = batch_size
        # T = sequence_length
        # C = embedding_dim
        # Input: (B, 3*T, C)
        # 3*T = because of the 3 tokens per event
        ##########################################
        # self.norm = tf.keras.layers.LayerNormalization() # (B, 3*T, C) -> (B, 3*T, C)
        # Propabilities for the next token
        self.identifier = tf.keras.layers.Dense(vocab_size) # (B, T, C) -> (B, T, vocab_size) 
        self.event_props_projection = tf.keras.layers.Dense(prop_dim) # (B, T, C) -> (B, T, prop_dim)
        self.trace_props_projection = tf.keras.layers.Dense(prop_dim) # (B, T, C) -> (B, T, prop_dim)
        
    def call(self, x):
        # x = (B, 3*T, C)
        ##########################################
        # x = self.norm(x) # (B, 3*T, C) -> (B, 3*T, C)
        # Split the input into the three tokens
        identity = x[:,::3,:] # (B, 3*T, C) -> (B, T, C)
        event_props = x[:,1::3,:] # (B, 3*T, C) -> (B, T, C)
        trace_props = x[:,2::3,:] # (B, 3*T, C) -> (B, T, C)
        
        # Get the probabilities for the next token
        identity = self.identifier(identity) # (B, T, C) -> (B, T, vocab_size)
        
        event_props = self.event_props_projection(event_props) # (B, T, C) -> (B, T, prop_dim)
        
        trace_props = self.trace_props_projection(trace_props) # (B, T, C) -> (B, T, prop_dim)
        
        # Reshape predictions to input-like format:
        # output = [(identifier, event_props, trace_props)] = batches of tuples of sequences 
        # each tuple is one sequence
        # Tuple = (
            #   identifier: tf.Tensor: shape=(T,vocab_size), dtype=float32, 
            #   event_props: tf.Tensor: shape=(T, prop_dim), dtype=float32, 
            #   trace_props: tf.Tensor: shape=(T, prop_dim), dtype=float32
            # )
        
        # Unstack the tensors to get a list of tensors for each batch
        identity_unstacked = tf.unstack(identity, axis=0)
        event_props_unstacked = tf.unstack(event_props, axis=0)
        trace_props_unstacked = tf.unstack(trace_props, axis=0)

        # Combine the unstacked tensors into a list of tuples
        combined_list = [(id_batch, ev_batch, tr_batch) for id_batch, ev_batch, tr_batch in zip(identity_unstacked, event_props_unstacked, trace_props_unstacked)]
        
        return combined_list
        
        # Other format (decided against it):
        # Unstack tensors along the batch dimension
        identity = tf.unstack(identity, axis=0) # (B, T, vocab_size) -> (T, vocab_size)
        event_props = tf.unstack(event_props, axis=0)
        trace_props = tf.unstack(trace_props, axis=0)
        
        predicted_sequence = [OutputProjection.process_batch(i, e, t) for i, e, t in zip(identity, event_props, trace_props)]
        # [[(identifier, event_props, trace_props)]] = batches of sequences of tuples
        
        return predicted_sequence
    
    @staticmethod    
    def process_batch(*ts):
        # Unstack tensors along time dimension
        ts = [tf.unstack(t, axis=0) for t in ts]
        
        # Stack the unstacked tensors along the batch dimension
        return list(zip(*ts))
                 
@tf.keras.saving.register_keras_serializable()
class Transformer(tf.keras.Model):
    
    def __init__(self,vocab_size ,embedding_dim=1_000, n_head=6, n_layers=6, batch_size=40, dropout=0.15, prop_dim=1_000, sequence_length=100, ffwd_dim=None):
        super().__init__()
        # B = batch_size
        T = sequence_length
        C = embedding_dim
        
        self.input_tokeniser = InputTokeniser(prop_dim=prop_dim) # (B, T) -> (B, 3*T)
        self.sequence_encoding = SequenceEncoding(embedding_dim=C, batch_size=batch_size, vocab_size=vocab_size, sequence_length=sequence_length) # (B, 3*T) -> (B, 3*T, C)
        self.decoder_block = tf.keras.Sequential([
            DecoderLayer(embedding_dim=C, n_head=n_head, dropout=dropout, ffwd_dim=ffwd_dim)
            for _ in range(n_layers)
        ]) # (B, 3*T, C) -> (B, 3*T, C)
        self.output_projection = OutputProjection(prop_dim=prop_dim, vocab_size=vocab_size) # (B, 3*T, C) -> [[(identifier, event_props, trace_props)]] (B, T, Tuple-Element-Dependent )
        
    def call(self, x):
        # x = [(identifier, event_props, trace_props)] = batches of tuples of sequences 
        # each tuple is one sequence
        # Tuple = (
            #   identifier: tf.Tensor: shape=(sequence_length,), dtype=int32, 
            #   event_props: tf.Tensor: shape=(sequence_length, prop_dim), dtype=float32, 
            #   trace_props: tf.Tensor: shape=(sequence_length, prop_dim), dtype=float32
            # )
        x = self.input_tokeniser(x) # (B, T) -> (B, 3*T)
        x = self.sequence_encoding(x) # (B, 3*T) -> (B, 3*T, C)
        x = self.decoder_block(x) # (B, 3*T, C) -> (B, 3*T, C)
        x = self.output_projection(x) # (B, 3*T, C) -> [[(identifier, event_props, trace_props)]] (B, T, Tuple-Element-Dependent )
        
        return x
        
        