import json
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Ensure that the working directory is the file's directory
file_dir = os.path.dirname(__file__)
os.chdir(file_dir)

from model import Transformer

year = 17 # [2017,2019]
path = f"./datasets/processor_results/bpi_{year}"

with open(path + "/trace/traces.json", 'r') as f:
    i_seq = json.load(f)
with open(path + "/trace/events_props.json", 'r') as f:
    e_seq = json.load(f)
with open(path + "/trace/traces_props.json", 'r') as f:
    t_seq = json.load(f)
    
with open(path + "/maps/event_map.json", 'r') as f:
    event_map = json.load(f)


# print(f"i_seq: {len(i_seq)} \ne_seq: {len(e_seq)}, {len(e_seq[0])} \nt_seq: {len(t_seq)}, {len(t_seq[0])} \nevent_map: {len(event_map)}")

vocab_size = len(event_map)

# Split into train and val
split = int(len(i_seq) * 0.8)

train_i , val_i = i_seq[:split], i_seq[split:]
train_e , val_e = e_seq[:split], e_seq[split:]
train_t , val_t = t_seq[:split], t_seq[split:]

def generate_batch(batch_size=60, sequence_length=200, split='train'):
    # Generates batched data
    # btach_size = number of batches to generate
    
    # Select data
    (data_i, data_e, data_t) = (train_i, train_e, train_t) if split == 'train' else (val_i, val_e, val_t)    

    # Randomly select a starting position for the batch
    idx = tf.random.uniform(shape=(batch_size,), maxval=len(data_i) - sequence_length, dtype=tf.int32) 
    
    # Create the input sequence
    x_i = tf.stack([tf.cast(tf.convert_to_tensor(data_i[i:i+sequence_length]), dtype=tf.int32) for i in idx])
    x_e = tf.stack([tf.convert_to_tensor(data_e[i:i+sequence_length]) for i in idx])
    x_t = tf.stack([tf.convert_to_tensor(data_t[i:i+sequence_length]) for i in idx])
    
    # Combine to input format
    # x = [(identifier, event_props, trace_props)] = batches of tuples of sequences 
        # each tuple is one sequence
        # Tuple = (
            #   identifier: tf.Tensor: shape=(sequence_length,), dtype=int32, 
            #   event_props: tf.Tensor: shape=(sequence_length, prop_dim), dtype=float32, 
            #   trace_props: tf.Tensor: shape=(sequence_length, prop_dim), dtype=float32
            # )
    x = [*zip(x_i, x_e, x_t)]
    
    # Create the target sequence
    y_i = tf.stack([tf.cast(tf.convert_to_tensor(data_i[(i+1):(i+1)+sequence_length]), dtype=tf.int32) for i in idx])
    y_e = tf.stack([tf.convert_to_tensor(data_e[(i+1):(i+1)+sequence_length]) for i in idx])
    y_t = tf.stack([tf.convert_to_tensor(data_t[(i+1):(i+1)+sequence_length]) for i in idx])
    
    y = [*zip(y_i, y_e, y_t)]
    
    return x, y

def weighted_loss_fn(y_true, y_pred, weights=(1, 0.01, 0.01), step=0):
    # Assuming y_pred is of shape (B, T, vocab_size)
    # and y_true is of shape (B, T) with integer labels
    loss_fn_i = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)#, reduction=tf.keras.losses.Reduction.NONE)
    mse_loss_fn = tf.keras.losses.MeanAbsoluteError() # tf.keras.losses.MeanSquaredError()#reduction=tf.keras.losses.Reduction.NONE)
    

    i_pred, e_props_pred, t_props_pred = [tf.convert_to_tensor(t) for t in zip(*y_pred)]
    i_true, e_props_true, t_props_true = [tf.convert_to_tensor(t) for t in zip(*y_true)]

    # Loss for i
    loss_i = loss_fn_i(i_true, i_pred)
    # print(f"i_pred: {i_pred.shape} \n i_true: {i_true.shape} \n loss: {loss_i.shape}")
    # mean_loss_i = tf.reduce_mean(loss_i)
    # print(f"result: {result_i}")
    # Calculate the mean loss across the batch
    
    # Loss for e and t

    # Compute the MSE loss for each tensor
    loss_e_props = mse_loss_fn(e_props_true, e_props_pred)
    loss_t_props = mse_loss_fn(t_props_true, t_props_pred)

    # Compute the mean loss across the batch for each
    # mean_loss_e_props = tf.reduce_mean(loss_e_props)
    # mean_loss_t_props = tf.reduce_mean(loss_t_props)

    if step % 2 == 0: 
        return weights[0] * loss_i
    else:  
        return weights[1] * loss_e_props + weights[2] * loss_t_props

    #############
    # Combine the losses with the loss for "i_pred"
    total_loss = weights[0] * mean_loss_i + weights[1] * mean_loss_e_props + weights[2] * mean_loss_t_props
    
    return total_loss

def mean_accuracy(y_true, y_pred):
    # Extract the identifier predictions
    i_pred, _, _ = [tf.convert_to_tensor(t) for t in zip(*y_pred)]
    i_true, _, _ = [tf.convert_to_tensor(t) for t in zip(*y_true)]
    
    i_true = tf.cast(i_true, dtype=tf.int32)
    
    # Calculate accuracy
    correct_predictions = tf.equal(i_true, tf.cast(tf.argmax(i_pred, -1), dtype=tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    return accuracy

def estimate_current_loss(model, iters=200, batch_size=60, sequence_length=200):
    result = {
        'id': {'loss': {}, 'accuracy': {}},
        'props': {}
    }
    model.trainable = False
    
    for split in ['train', 'validate']:
        losses = []
        accuracies = []
        for _ in range(iters):
            xb, yb = generate_batch(batch_size=batch_size, sequence_length=sequence_length, split=split)
            y_pred = model(xb)
            loss = weighted_loss_fn(yb, y_pred, step=0)
            losses.append(loss)
            
            accuracy = mean_accuracy(yb, y_pred)
            accuracies.append(accuracy)
            
        result['id']['loss'][split] = np.mean(losses)
        result['id']['accuracy'][split] = np.mean(accuracies)
        
    for split in ['train', 'validate']:
        losses = []
        for _ in range(iters):
            xb, yb = generate_batch(batch_size=batch_size, sequence_length=sequence_length, split=split)
            y_pred = model(xb)
            loss = weighted_loss_fn(yb, y_pred, step=1)
            losses.append(loss)
            
        result['props'][split] = np.mean(losses)
    
    model.trainable = True
    
    return result
        
    
    

def training(model=None, epochs=10, iters_per_epoch=500, batch_size=60, sequence_length=200, learning_rate=3e-4, save_path='./models', model_name='transformer'):
    # Create the model
    model = model if model else Transformer(
        vocab_size=vocab_size,
        embedding_dim=1_000,
        n_head=6,
        n_layers=6,
        batch_size=batch_size,
        dropout=0.1,
        prop_dim=100,
        sequence_length=sequence_length,
        ffwd_dim=None
    )
    
    # Create the optimizer
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    
    # Initialize model with dummy data
    x, _ = generate_batch(batch_size=batch_size, sequence_length=sequence_length, split='train')
    
    model(x)
    
    # Get the full list of trainable variables
    variables = model.trainable_variables

    # Build the optimizer with the full list of trainable variables
    optimizer.build(variables)
    
    loss_fn = weighted_loss_fn
    loss_col = []
    
    for epoch in range(epochs):
        
        for iter in tqdm(range(iters_per_epoch), desc=f"Epoch {epoch+1}", unit="iteration"):
            
            xb, yb = generate_batch(batch_size=batch_size, sequence_length=sequence_length, split='train')
            with tf.GradientTape() as tape:
                # Forward pass
                y_pred = model(xb)
                
                # Compute the loss value
                loss = loss_fn(yb, y_pred, step=iter) 
                
            # Compute gradients and update parameters
            gradients = tape.gradient(loss, model.trainable_variables)
            # Filter out None gradients
            gradients_and_variables = [(grad, var) for grad, var in zip(gradients, model.trainable_variables) if grad is not None]
            optimizer.apply_gradients(gradients_and_variables)
            # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
        # After each epoch estimate the current loss
        losses = estimate_current_loss(
            model, 
            iters=int(max((iters_per_epoch / 10), 1)), 
            batch_size=batch_size, 
            sequence_length=sequence_length)
        print(f"Epoch: {epoch+1} ({iters_per_epoch} iterations) \n Train-loss: \t\t id: {losses['id']['loss']['train']:.4f} \t props: {losses['props']['train']:.0f} \n Validation-loss: \t id: {losses['id']['loss']['validate']:.4f} \t props: {losses['props']['validate']:.0f} \n Train-accuracy: \t id: {losses['id']['accuracy']['train']:.4f} \n Validation-accuracy: \t id: {losses['id']['accuracy']['validate']:.4f}")
        
        loss_col.append(losses)
    
    # Save the model
    model.save(f"{save_path}/model_{model_name}.keras")
    
    return model, loss_col
        
def convert(o):
    if isinstance(o, np.float32) or isinstance(o, np.float64):
        return float(o)
    elif isinstance(o, np.int32) or isinstance(o, np.int64):
        return int(o)  
    raise TypeError

if __name__ == '__main__':
    print("GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    epochs=1
    iters_per_epoch=10
    batch_size=2
    sequence_length=50
    m, losses = training( 
        epochs=epochs, 
        iters_per_epoch=iters_per_epoch, 
        batch_size=batch_size, 
        sequence_length=sequence_length, 
        model_name=f"17_dataset_{epochs*iters_per_epoch}_iters_{sequence_length}_seqL"
    )
    
    with open(f"{year}_dataset_{epochs*iters_per_epoch}_iters_{sequence_length}_seqL_losses.json", 'w') as f:
        json.dump(losses, f, default=convert)
    

    
