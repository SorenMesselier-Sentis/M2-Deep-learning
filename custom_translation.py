import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
import ast
import time
from tqdm import tqdm

from transformer.custom_transformer import Transformer, Encoder, Decoder, EncoderBlock, MultiHeadAttentionBlock, \
    FeedForwardBlock, DecoderBlock, InputEmbeddings, ProjectionLayer, PositionalEncoding

# Determine which device to use
device = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using {device} device")

# BPE Tokenizer Class
class BpeTokenizer:
    def __init__(self, corpus, vocab_size=32000):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], vocab_size=vocab_size)
        self.tokenizer.train_from_iterator(corpus, trainer)

        # Post-processing
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
        )

        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

# Load Dataset and Extract Translations
dataset = load_dataset("Helsinki-NLP/opus_books", "fr-pl")
df = dataset["train"].to_pandas()

# Ensure correct column extraction
def extract_translation(row, lang):
    try:
        return ast.literal_eval(row["translation"]).get(lang, "")
    except (ValueError, SyntaxError):
        return ""

df["fr"] = df.apply(lambda row: extract_translation(row, "fr"), axis=1)
df["pl"] = df.apply(lambda row: extract_translation(row, "pl"), axis=1)

# Prepare corpus
all_text = df["fr"].dropna().tolist() + df["pl"].dropna().tolist()
bpe_tokenizer = BpeTokenizer(all_text)
print(f"BPE Vocabulary Size: {bpe_tokenizer.vocab_size}")

# Tokenize sentences
df["fr_tokenized"] = df["fr"].apply(lambda x: bpe_tokenizer.encode(x))
df["pl_tokenized"] = df["pl"].apply(lambda x: bpe_tokenizer.encode(x))

# Convert to tensors with padding
src_sentences = torch.nn.utils.rnn.pad_sequence(
    [torch.tensor(x) for x in df["fr_tokenized"]], batch_first=True, padding_value=0
)
tgt_sentences = torch.nn.utils.rnn.pad_sequence(
    [torch.tensor(x) for x in df["pl_tokenized"]], batch_first=True, padding_value=0
)

# Transformer Model Setup
d_model = 128
vocab_size = bpe_tokenizer.vocab_size
seq_len = src_sentences.shape[1]
num_heads = 8
dropout = 0.5
num_layers = 4
ff_dim = 1024

# Define model
model = Transformer(
    encoder=Encoder(d_model, nn.ModuleList([
        EncoderBlock(d_model, MultiHeadAttentionBlock(d_model, num_heads, dropout),
                     FeedForwardBlock(d_model, ff_dim, dropout), dropout)
        for _ in range(num_layers)
    ])),
    decoder=Decoder(d_model, nn.ModuleList([
        DecoderBlock(d_model, MultiHeadAttentionBlock(d_model, num_heads, dropout),
                     MultiHeadAttentionBlock(d_model, num_heads, dropout),
                     FeedForwardBlock(d_model, ff_dim, dropout), dropout)
        for _ in range(num_layers)
    ])),
    src_embed=InputEmbeddings(d_model, vocab_size),
    tgt_embed=InputEmbeddings(d_model, vocab_size),
    src_pos=PositionalEncoding(d_model, seq_len, dropout),
    tgt_pos=PositionalEncoding(d_model, seq_len, dropout),
    projection_layer=ProjectionLayer(d_model, vocab_size)
)
model.to(device)

# Define loss and optimizer with weight decay (L2 regularization)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD token
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Function to create a causal mask for the decoder
def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones((seq_len, seq_len))).to(device)  # Lower triangular mask
    return mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)

# Gradient clipping function
def clip_gradients(optimizer, clip_value=1.0):
    for param in optimizer.param_groups:
        torch.nn.utils.clip_grad_norm_(param['params'], clip_value)

# Learning rate scheduler
def get_scheduler(optimizer, warmup_steps=4000):
    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) ** (-0.5), (step + 1) * (warmup_steps ** (-1.5)))
    )

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="runs/transformer_experiment")

# Training Setup
num_epochs = 5
batch_size = 32
clip_value = 1.0
lr = 0.0001
num_batches = len(df) // batch_size

# Set up learning rate scheduler
scheduler = get_scheduler(optimizer, num_epochs)

# Early stopping logic
patience = 3  # Stop training if no improvement in validation loss for 3 epochs
best_val_loss = float('inf')
patience_counter = 0

# Training loop
for epoch in range(num_epochs):
    start_time = time.time()  # Start time tracking

    epoch_loss = 0
    progress_bar = tqdm(range(0, len(df), batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}")

    for i in progress_bar:
        # Get batch data
        src_batch = src_sentences[i:i + batch_size].to(device)
        tgt_batch = tgt_sentences[i:i + batch_size].to(device)

        # Create masks
        src_mask = (src_batch != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = create_causal_mask(tgt_batch.shape[1])

        # Forward pass
        encoder_output = model.encode(src_batch, src_mask)
        decoder_output = model.decode(encoder_output, src_mask, tgt_batch, tgt_mask)
        output_logits = model.project(decoder_output)

        # Compute loss
        loss = criterion(output_logits.view(-1, vocab_size), tgt_batch.view(-1).long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent explosion
        clip_gradients(optimizer, clip_value)

        optimizer.step()

        epoch_loss += loss.item()

        # Log loss to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch * num_batches + i)

        # Log histograms of model parameters and gradients
        for name, param in model.named_parameters():
            writer.add_histogram(f'parameters/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'gradients/{name}', param.grad, epoch)

        # Update progress bar with loss info
        progress_bar.set_postfix(loss=loss.item())

    # Update learning rate with scheduler
    scheduler.step()

    # Calculate elapsed time per epoch
    elapsed_time = time.time() - start_time
    avg_time_per_iteration = elapsed_time / num_batches
    print(f"Epoch {epoch + 1} completed in {elapsed_time:.2f} seconds (Avg per batch: {avg_time_per_iteration:.2f}s)")
    print(f"Epoch {epoch + 1} Loss: {epoch_loss / num_batches:.4f}")

    # Early stopping check
    if epoch_loss < best_val_loss:
        best_val_loss = epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    # Log average epoch loss to TensorBoard
    writer.add_scalar('Loss/epoch', epoch_loss / num_batches, epoch)

    # Generate and log a dummy sentence after each epoch
    dummy_input = torch.randint(0, vocab_size, (1, seq_len)).to(device)  # Create a random input tensor
    dummy_input_mask = (dummy_input != 0).unsqueeze(1).unsqueeze(2)  # Create mask for dummy input

    # Pass the dummy input through the encoder and decoder
    dummy_encoder_output = model.encode(dummy_input, dummy_input_mask)
    dummy_decoder_output = model.decode(dummy_encoder_output, dummy_input_mask, dummy_input, tgt_mask)

    # Project the decoder output to get logits
    dummy_output_logits = model.project(dummy_decoder_output)

    # Get the predicted token ids by taking argmax over the logits
    predicted_ids = dummy_output_logits.argmax(dim=-1).squeeze().cpu().numpy()

    # Decode the predicted token ids into a sentence
    dummy_sentence = bpe_tokenizer.decode(predicted_ids)

    # Print the dummy sentence to the terminal
    print(f"Dummy Sentence after Epoch {epoch + 1}: {dummy_sentence}")

# Final model save
model_save_path = "models/transformer_model_final.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Final model saved at {model_save_path}")

# Close TensorBoard writer
writer.close()
