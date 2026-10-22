import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import time
from datetime import datetime
import MTGPforNewSimulatorBasedOnJava.LoadIndividual as loadInd
from deap.gp import Primitive, Terminal, PrimitiveTree
from TransGP.Transformer import DrawTree

function_set = ['add', 'subtract', 'multiply', 'protected_div', 'maximum', 'minimum']
terminal_set = ['WIQ', 'NIQ', 'MWT', 'PT', 'NPT', 'OWT', 'WKR', 'NOR', 'TIS', 'SLACK']

def is_function(token):
    return token in function_set


def is_terminal(token):
    return token in terminal_set

# Parser from prefix list to nested tree
def prefix_list_to_tree(tokens):
    def parse(index):
        token = tokens[index]
        if is_function(token):
            left, next_index = parse(index + 1)
            right, next_index = parse(next_index)
            return (token, left, right), next_index
        elif is_terminal(token):
            return token, index + 1
        else:
            raise ValueError(f"Unknown token: {token}")

    return parse(0)[0]



def primitive_tree_to_string_tree(primitive_tree):
    return [node.name for node in primitive_tree]

def string_tree_to_primitive_tree(string_tree):
    primitive_tree = []
    for name in string_tree:
        if is_function(name):
            prim = Primitive(name, (float, float), float)
        else:
            prim = Terminal(name, symbolic=True, ret=float)
        primitive_tree.append(prim)
    return primitive_tree


def stringify_tree(tree):
    if isinstance(tree, str):
        return tree
    func, left, right = tree
    return f"{func}({stringify_tree(left)}, {stringify_tree(right)})"


# =======================
# Training Logger Class
# =======================

class TrainingLogger:
    def __init__(self, model_name, save_dir="./training_logs"):
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.epoch_logs = []
        self.start_time = None
        self.total_time = 0

    def start_training(self):
        """Mark the start of training"""
        self.start_time = time.time()

    def log_epoch(self, epoch, train_loss, epoch_time, **kwargs):
        """Log information for each epoch"""
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'epoch_time_seconds': epoch_time,
            'cumulative_time_seconds': time.time() - self.start_time
        }
        # Add any additional metrics (e.g., KL loss for VAE)
        log_entry.update(kwargs)
        self.epoch_logs.append(log_entry)

    def end_training(self):
        """Mark the end of training"""
        self.total_time = time.time() - self.start_time

    def save_to_excel(self, filename=None):
        """Save training logs to Excel file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name}_training_log.xlsx"

        filepath = os.path.join(self.save_dir, filename)

        # Create DataFrame from logs
        df = pd.DataFrame(self.epoch_logs)

        # Create summary statistics
        summary_data = {
            'Metric': [
                'Total Training Time (seconds)',
                'Total Training Time (minutes)',
                'Average Epoch Time (seconds)',
                'Final Train Loss',
                'Best Train Loss',
                'Epoch with Best Loss',
                'Number of Epochs'
            ],
            'Value': [
                self.total_time,
                self.total_time / 60,
                df['epoch_time_seconds'].mean(),
                df['train_loss'].iloc[-1],
                df['train_loss'].min(),
                df['train_loss'].idxmin() + 1,
                len(df)
            ]
        }
        summary_df = pd.DataFrame(summary_data)

        metadata = {
            'Parameter': ['Model Name', 'Training Date', 'Total Epochs', 'Dataset Size'],
            'Value': [
                self.model_name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(df),
                'N/A'
            ]
        }
        metadata_df = pd.DataFrame(metadata)

        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Epoch Details', index=False)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            print(f"Training log saved to: {filepath}")
            return filepath
        except ModuleNotFoundError as error:
            if error.name != 'openpyxl':
                raise
            csv_base = os.path.splitext(filepath)[0]
            epoch_path = f"{csv_base}_epochs.csv"
            summary_path = f"{csv_base}_summary.csv"
            metadata_path = f"{csv_base}_metadata.csv"
            df.to_csv(epoch_path, index=False)
            summary_df.to_csv(summary_path, index=False)
            metadata_df.to_csv(metadata_path, index=False)
            print("openpyxl is not installed; training logs saved as CSV files:")
            print(f"  {epoch_path}")
            print(f"  {summary_path}")
            print(f"  {metadata_path}")
            return epoch_path

    def print_summary(self):
        """Print training summary"""
        df = pd.DataFrame(self.epoch_logs)
        print("\n" + "=" * 60)
        print(f"Training Summary - {self.model_name}")
        print("=" * 60)
        print(f"Total Training Time: {self.total_time:.2f} seconds ({self.total_time / 60:.2f} minutes)")
        print(f"Average Epoch Time: {df['epoch_time_seconds'].mean():.2f} seconds")
        print(f"Final Train Loss: {df['train_loss'].iloc[-1]:.4f}")
        print(f"Best Train Loss: {df['train_loss'].min():.4f} (Epoch {df['train_loss'].idxmin() + 1})")
        print("=" * 60 + "\n")

# --- Simulated Data (e.g., scheduling heuristics represented as vectors) ---
class HeuristicDataset(Dataset):
    def __init__(self, dataset_name, rule_type, archive_count=30, task_embedding_dict=None,
                 base_path="./archive_gp/train", seed_offset=0, archive_seed_map=None):
        self.data = []
        self.tasks = []
        self.data_pairwise = []
        self.task_to_index = {}
        self.archive_seed_map = archive_seed_map or objective_archive_seed_map(seed_offset)
        rule_type_index = 0
        if rule_type == "sequencing":
            rule_type_index = 0
        elif rule_type == "routing":
            rule_type_index = 1
        else:
            raise ValueError("Unknown rule type!")

        parts = dataset_name.split('-')
        objective_name = parts[0]
        if objective_name not in self.archive_seed_map:
            raise ValueError(f"No archive seed mapping configured for objective: {objective_name}")
        archive_seeds = self.archive_seed_map[objective_name]
        print(f"Loading {objective_name} Transformer dataset archives with seeds: {archive_seeds}")
        task_index_counter = 0
        for key, value in task_embedding_dict.items():
            task_name = key
            dataset_name = parts[0] + "-" + str(value[0]) + "-1.5"

            # for i in range(archive_count):
            for i in archive_seeds:
                topInds = loadInd.load_archive_with_filename(i, dataset_name, base_path)
                for key, individual in topInds.items():
                    try:
                        # Assume `key` is of format like "task1_ind123", so split on "_"
                        # task_name = f'task_{i}'
                        if task_name not in self.task_to_index:
                            self.task_to_index[task_name] = task_index_counter
                            task_index_counter += 1
                        task_idx = self.task_to_index[task_name]

                        tree = prefix_list_to_tree(individual[rule_type_index])
                        self.data.append(tree)  # Save tree and its task index
                        self.tasks.append(task_name)
                        self.data_pairwise.append((tree, task_idx))  # Save tree and its task index

                    except Exception as e:
                        print(f"Error processing individual {key} in archive {i}: {e}")

        self.index_to_task = {v: k for k, v in self.task_to_index.items()}
        print(f"Dataset loaded: overall {len(self.data)} samples from all tasks")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tree, task_idx = self.data[idx]
        return tree, task_idx

    def get_taskwise_trees(self):
        task_tree_dict = {idx: [] for idx in self.index_to_task}
        for tree, task_idx in self.data_pairwise:
            task_tree_dict[task_idx].append(tree)
        return task_tree_dict, self.index_to_task

    def remove_duplicates(self):
        """
        Removes duplicate entries from self.data, self.tasks, and self.data_pairwise.
        Duplicates are detected based on the string representation of trees.
        """
        seen = set()
        new_data = []
        new_tasks = []
        new_data_pairwise = []

        for tree, task_name, (tree_pair, task_idx) in zip(self.data, self.tasks, self.data_pairwise):
            tree_str = str(tree)  # You may replace str(tree) with a hash function if needed
            if tree_str not in seen:
                seen.add(tree_str)
                new_data.append(tree)
                new_tasks.append(task_name)
                new_data_pairwise.append((tree_pair, task_idx))

        removed_count = len(self.data) - len(new_data)
        self.data = new_data
        self.tasks = new_tasks
        self.data_pairwise = new_data_pairwise

        print(f"Removed {removed_count} duplicated samples. Remaining: {len(self.data)}.")

    def remove_duplicates_each_task(self):
        """
        Removes duplicate entries from self.data, self.tasks, and self.data_pairwise,
        but only within the same task_name. That is, the same tree can appear in
        different tasks, but not more than once in the same task.
        """
        seen_per_task = {}  # task_name -> set of tree strings
        new_data = []
        new_tasks = []
        new_data_pairwise = []

        for tree, task_name, (tree_pair, task_idx) in zip(self.data, self.tasks, self.data_pairwise):
            tree_str = str(tree)  # Customize this if tree has a better serialization
            if task_name not in seen_per_task:
                seen_per_task[task_name] = set()

            if tree_str not in seen_per_task[task_name]:
                seen_per_task[task_name].add(tree_str)
                new_data.append(tree)
                new_tasks.append(task_name)
                new_data_pairwise.append((tree_pair, task_idx))

        removed_count = len(self.data) - len(new_data)
        self.data = new_data
        self.tasks = new_tasks
        self.data_pairwise = new_data_pairwise

        print(f"Removed {removed_count} duplicated samples (within-task only). Remaining: {len(self.data)}.")



# =========================
# Tokenizer and Vocabulary
# =========================

class SymbolicTokenizer:
    def __init__(self, function_set, terminal_set):
        self.special_tokens = ['PAD', 'START', 'END']
        self.vocab = self.special_tokens + function_set + terminal_set
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, tree):
        tokens = ['START'] + self.flatten(tree) + ['END']
        return [self.token_to_id[tok] for tok in tokens]

    def decode(self, token_ids):
        tokens = [self.id_to_token[i] for i in token_ids if i != self.token_to_id['PAD']]
        return tokens

    def flatten(self, node):
        if isinstance(node, str):
            return [node]
        elif isinstance(node, tuple):
            tokens = [node[0]]
            for child in node[1:]:
                tokens += self.flatten(child)
            return tokens
        else:
            raise ValueError("Unexpected node type")

    def vocab_size(self):
        return len(self.vocab)


# ================
# Dataset Example
# ================

class SymbolicTreeDataset(Dataset):
    def __init__(self, trees, tokenizer, max_len, task_names, task_embedding_dict):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_embedding_dict = task_embedding_dict
        self.task_names = task_names  # parallel to `trees`, same length

        self.data = [self.pad(tokenizer.encode(t)) for t in trees]

    def pad(self, tokens):
        return tokens + [0] * (self.max_len - len(tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.data[idx][1:], dtype=torch.long)
        task_name = self.task_names[idx]
        task_emb = torch.tensor(self.task_embedding_dict[task_name], dtype=torch.float32)
        return x, y, task_emb


def objective_archive_seed_map(seed_offset=0):
    return {
        "Fmax": [0 + seed_offset, 1 + seed_offset, 2 + seed_offset],
        "Fmean": [3 + seed_offset, 4 + seed_offset, 5 + seed_offset],
        "Tmean": [6 + seed_offset, 7 + seed_offset, 8 + seed_offset],
    }


# =======================
# Transformer Model Core
# =======================

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dropout=0.1, max_len=512, task_emb_dim=16):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        self.task_proj = nn.Linear(task_emb_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x, task_embedding):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        pos = self.pos_embedding[:, :seq_len, :]
        x = self.token_embedding(x) + pos

        # Expand and add task embedding as conditioning
        task_cond = self.task_proj(task_embedding).unsqueeze(1).expand(-1, seq_len, -1)
        x = x + task_cond

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        memory = torch.zeros_like(x)
        out = self.transformer(x, memory, tgt_mask=tgt_mask)
        return self.fc_out(out)



class TransformerTree:
    def __init__(self, seed, dataset_name, rule_type, epochs=4, lr=1e-3, max_len=256, task_emb_dim=2, log_dir="./training_logs"):
        self.set_seed(seed)
        self.dataset_name = dataset_name
        self.rule_type = rule_type
        self.model_dir = f"./Transformer/{self.rule_type}_transformer_multitask"
        self.log_dir = log_dir
        print("Rule type: ", self.rule_type)
        self.max_len = max_len

        self.tokenizer = SymbolicTokenizer(function_set, terminal_set)

        # self.model = TransformerDecoder(vocab_size=self.tokenizer.vocab_size(), max_len=max_len, task_emb_dim=task_emb_dim)

        self.model = TransformerDecoder( # lighter model # original
            vocab_size=self.tokenizer.vocab_size(),
            d_model=256,  # ↓ from 256
            nhead=4,  # ↓ from 4
            num_layers=4,  # ↓ from 4
            dropout=0.1,  # ↑ from 0.1 (more regularization)
            max_len=max_len,
            task_emb_dim=task_emb_dim
        )
        # self.model = TransformerDecoder(  # lighter model
        #     vocab_size=self.tokenizer.vocab_size(),
        #     d_model=256,  # ↓ from 256
        #     nhead=2,  # ↓ from 4
        #     num_layers=2,  # ↓ from 4
        #     dropout=0.2,  # ↑ from 0.1 (more regularization)
        #     max_len=max_len,
        #     task_emb_dim=task_emb_dim
        # )
        self.epochs = epochs
        self.lr = lr

        self.logger = TrainingLogger(
            model_name=f"Transformer_{self.rule_type}_{self.dataset_name}",
            save_dir=log_dir
        )

    def set_seed(self, seed):
        """Sets random seeds for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if using multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False  # set to false, if not using cudnn.benchmark
        np.random.seed(seed)

    # ===================
    # Training Loop (CPU)
    # ===================

    def train(self, dataloader, save_logs=True):
        device = torch.device("cpu")
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        best_loss = np.inf

        self.logger.start_training()

        print(f"\nStarting Transformer training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.model.train()
            total_loss = 0

            for x, y, task_emb in dataloader:
                x, y, task_emb = x.to(device), y.to(device), task_emb.to(device)
                optimizer.zero_grad()
                logits = self.model(x, task_emb)  # Pass task embedding in
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                # Add gradient clipping to prevent exploding gradients. This limits the maximum value of the gradients.
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # max_norm is a common value
                optimizer.step()
                total_loss += loss.item()

            epoch_loss = total_loss / len(dataloader)
            epoch_time = time.time() - epoch_start_time

            self.logger.log_epoch(
                epoch=epoch + 1,
                train_loss=epoch_loss,
                epoch_time=epoch_time
            )

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                # Then, check the checkpoint frequency
                if epoch % 10 == 0:
                    self.save_model(epoch)

            cumulative_time = time.time() - self.logger.start_time
            print(f"Epoch {epoch + 1}/{self.epochs} | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"Total: {cumulative_time:.2f}s")

        self.logger.end_training()
        self.logger.print_summary()

        if save_logs:
            self.logger.save_to_excel()


    def save_model(self, epoch=None, suffix=None):
        os.makedirs(self.model_dir, exist_ok=True)
        if epoch is None:
            checkpoint_suffix = "diffusion_transformer" if suffix is None else f"diffusion_transformer_{suffix}"
            torch.save(self.model.state_dict(),
                       os.path.join(self.model_dir, f"{self.dataset_name}_{checkpoint_suffix}.pt"))
            print(f"Model saved to {self.model_dir}/{self.dataset_name}_{checkpoint_suffix}.pt")
        else:
            torch.save(self.model.state_dict(), os.path.join(self.model_dir, f"{self.dataset_name}_diffusion_transformer_{epoch}.pt"))
            print(f"Model saved to {self.model_dir}/{self.dataset_name}_diffusion_transformer_{epoch}.pt")


    def load_model(self, model_dir=None, epoch=None):
        device = torch.device("cpu")
        if model_dir is not None:
            self.model_dir = model_dir
        if epoch is None:
            model_path = os.path.join(self.model_dir, f"{self.dataset_name}_diffusion_transformer_original.pt")
        else:
            model_path = os.path.join(self.model_dir, f"{self.dataset_name}_diffusion_transformer_{epoch}.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        print(f"Model loaded from {model_path}")
        return self.model


    # ==============================================
    # Sampling Function to Generate/Optimize a Tree
    # ==============================================
    def sample_valid_tree(self, task_embedding: torch.Tensor):
        self.model.eval()
        device = torch.device("cpu")
        self.model.to(device)

        input_ids = torch.tensor([[self.tokenizer.token_to_id['START']]], dtype=torch.long).to(device)
        generated = input_ids
        stack = [2]  # Start with root expecting 2 children

        task_embedding = task_embedding.to(device).unsqueeze(0)  # Shape: [1, embedding_dim]

        with torch.no_grad():
            for _ in range(self.max_len):
                logits = self.model(generated, task_embedding=task_embedding)
                next_token_logits = logits[:, -1, :]

                if stack and stack[-1] > 0:
                    valid_tokens = function_set + terminal_set
                else:
                    break

                valid_token_ids = [self.tokenizer.token_to_id[tok] for tok in valid_tokens]
                logits_masked = torch.full_like(next_token_logits, float('-inf'))
                logits_masked[:, valid_token_ids] = next_token_logits[:, valid_token_ids]

                next_token = torch.argmax(logits_masked, dim=-1, keepdim=True)
                tok_str = self.tokenizer.id_to_token[next_token.item()]
                generated = torch.cat((generated, next_token), dim=1)

                if is_function(tok_str):
                    stack[-1] -= 1
                    stack.append(2)
                elif is_terminal(tok_str):
                    stack[-1] -= 1
                    while stack and stack[-1] == 0:
                        stack.pop()
                else:
                    break

                if not stack:
                    end_token = torch.tensor([[self.tokenizer.token_to_id['END']]], dtype=torch.long).to(device)
                    generated = torch.cat((generated, end_token), dim=1)
                    break

        token_ids = generated.squeeze(0).tolist()
        decoded = self.tokenizer.decode(token_ids)
        valid = self.validate_tokens(decoded, function_set)
        return decoded, valid

    def sample_valid_tree_V1(self, task_embedding: torch.Tensor, pset):
        self.model.eval()
        device = torch.device("cpu")
        self.model.to(device)


        input_ids = torch.tensor([[self.tokenizer.token_to_id['START']]], dtype=torch.long).to(device)
        generated = input_ids
        stack = [2]  # Start with root expecting 2 children

        task_embedding = task_embedding.to(device).unsqueeze(0)  # Shape: [1, embedding_dim]

        with torch.no_grad():
            for _ in range(self.max_len):
                logits = self.model(generated, task_embedding=task_embedding)
                next_token_logits = logits[:, -1, :]

                if not stack:
                    break

                valid_tokens = function_set + terminal_set
                valid_token_ids = [self.tokenizer.token_to_id[tok] for tok in valid_tokens]
                logits_masked = torch.full_like(next_token_logits, float('-inf'))
                logits_masked[:, valid_token_ids] = next_token_logits[:, valid_token_ids]

                temperature = 1.2
                probs = torch.softmax(logits_masked / temperature, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                next_token = dist.sample().unsqueeze(-1)

                tok_str = self.tokenizer.id_to_token[next_token.item()]
                generated = torch.cat((generated, next_token), dim=1)

                if is_function(tok_str):
                    stack[-1] -= 1
                    stack.append(2)
                elif is_terminal(tok_str):
                    stack[-1] -= 1
                    while stack and stack[-1] == 0:
                        stack.pop()
                else:
                    break

                if not stack:
                    end_token = torch.tensor([[self.tokenizer.token_to_id['END']]], dtype=torch.long).to(device)
                    generated = torch.cat((generated, end_token), dim=1)
                    break

            while stack and len(generated[0]) < self.max_len:
                dummy_terminal = self.tokenizer.token_to_id[terminal_set[0]]
                generated = torch.cat((generated, torch.tensor([[dummy_terminal]], dtype=torch.long).to(device)), dim=1)
                stack[-1] -= 1
                while stack and stack[-1] == 0:
                    stack.pop()

            if len(generated[0]) < self.max_len:
                end_token = torch.tensor([[self.tokenizer.token_to_id['END']]], dtype=torch.long).to(device)
                generated = torch.cat((generated, end_token), dim=1)

        while stack:
            dummy_terminal = self.tokenizer.token_to_id[terminal_set[0]]
            generated = torch.cat((generated, torch.tensor([[dummy_terminal]], dtype=torch.long).to(device)), dim=1)
            stack[-1] -= 1
            while stack and stack[-1] == 0:
                stack.pop()

        token_ids = generated.squeeze(0).tolist()
        decoded_tokens = self.tokenizer.decode(token_ids)
        decoded = self.tokens_to_tree(decoded_tokens, function_set)
        tree_str = stringify_tree(decoded)
        primitive_tree = PrimitiveTree.from_string(tree_str, pset)
        return primitive_tree


    def sample_valid_tree_V2(self, task_embedding: torch.Tensor, pset):
        self.model.eval()
        device = torch.device("cpu")
        self.model.to(device)


        input_ids = torch.tensor([[self.tokenizer.token_to_id['START']]], dtype=torch.long).to(device)
        generated = input_ids
        stack = [2]  # Start with root expecting 2 children

        task_embedding = task_embedding.to(device).unsqueeze(0)  # Shape: [1, embedding_dim]

        with torch.no_grad():
            for _ in range(self.max_len):
                logits = self.model(generated, task_embedding=task_embedding)
                next_token_logits = logits[:, -1, :]

                if stack and stack[-1] > 0:
                    valid_tokens = function_set + terminal_set
                else:
                    break

                valid_token_ids = [self.tokenizer.token_to_id[tok] for tok in valid_tokens]
                logits_masked = torch.full_like(next_token_logits, float('-inf'))
                logits_masked[:, valid_token_ids] = next_token_logits[:, valid_token_ids]

                next_token = torch.argmax(logits_masked, dim=-1, keepdim=True)
                tok_str = self.tokenizer.id_to_token[next_token.item()]
                generated = torch.cat((generated, next_token), dim=1)

                if is_function(tok_str):
                    stack[-1] -= 1
                    stack.append(2)
                elif is_terminal(tok_str):
                    stack[-1] -= 1
                    while stack and stack[-1] == 0:
                        stack.pop()
                else:
                    break

                if not stack:
                    end_token = torch.tensor([[self.tokenizer.token_to_id['END']]], dtype=torch.long).to(device)
                    generated = torch.cat((generated, end_token), dim=1)
                    break

        token_ids = generated.squeeze(0).tolist()
        decoded_tokens = self.tokenizer.decode(token_ids)
        decoded = self.tokens_to_tree(decoded_tokens, function_set)
        tree_str = stringify_tree(decoded)
        primitive_tree = PrimitiveTree.from_string(tree_str, pset)
        return primitive_tree


    def mutate_tree(self, parent_tree, task_embedding: torch.Tensor, pset=None):
        self.model.eval()
        device = torch.device("cpu")
        self.model.to(device)

        task_embedding = torch.tensor(task_embedding, dtype=torch.float32)
        task_embedding = task_embedding.to(device).unsqueeze(0)  # [1, embedding_dim]

        parent_tree = primitive_tree_to_string_tree(parent_tree)
        parent_tree = ['START'] + parent_tree + ['END']
        parent_tree = self.tokens_to_tree(parent_tree, function_set)

        input_ids = self.tokenizer.encode(parent_tree)
        mutate_point = np.random.randint(1, len(input_ids) - 1)
        input_ids = input_ids[:mutate_point]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        generated = input_tensor.clone()

        # Reconstruct grammar stack
        token_ids = generated.squeeze(0).tolist()
        decoded_tokens = self.tokenizer.decode(token_ids)
        stack = [2]
        for tok in decoded_tokens[1:]:
            if is_function(tok):
                stack[-1] -= 1
                stack.append(2)
            elif is_terminal(tok):
                stack[-1] -= 1
                while stack and stack[-1] == 0:
                    stack.pop()

        with torch.no_grad():
            for _ in range(self.max_len - len(input_ids)):
                logits = self.model(generated, task_embedding=task_embedding)
                next_token_logits = logits[:, -1, :]

                if not stack:
                    break

                valid_tokens = function_set + terminal_set
                valid_token_ids = [self.tokenizer.token_to_id[tok] for tok in valid_tokens]
                logits_masked = torch.full_like(next_token_logits, float('-inf'))
                logits_masked[:, valid_token_ids] = next_token_logits[:, valid_token_ids]

                temperature = 0.5
                probs = torch.softmax(logits_masked / temperature, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                next_token = dist.sample().unsqueeze(-1)

                tok_str = self.tokenizer.id_to_token[next_token.item()]
                generated = torch.cat((generated, next_token), dim=1)

                if is_function(tok_str):
                    stack[-1] -= 1
                    stack.append(2)
                elif is_terminal(tok_str):
                    stack[-1] -= 1
                    while stack and stack[-1] == 0:
                        stack.pop()
                else:
                    break

                if not stack:
                    end_token = torch.tensor([[self.tokenizer.token_to_id['END']]], dtype=torch.long).to(device)
                    generated = torch.cat((generated, end_token), dim=1)
                    break

            while stack and len(generated[0]) < self.max_len:
                dummy_terminal = self.tokenizer.token_to_id[terminal_set[0]]
                generated = torch.cat((generated, torch.tensor([[dummy_terminal]], dtype=torch.long).to(device)), dim=1)
                stack[-1] -= 1
                while stack and stack[-1] == 0:
                    stack.pop()

            if len(generated[0]) < self.max_len:
                end_token = torch.tensor([[self.tokenizer.token_to_id['END']]], dtype=torch.long).to(device)
                generated = torch.cat((generated, end_token), dim=1)

        while stack:
            dummy_terminal = self.tokenizer.token_to_id[terminal_set[0]]
            generated = torch.cat((generated, torch.tensor([[dummy_terminal]], dtype=torch.long).to(device)), dim=1)
            stack[-1] -= 1
            while stack and stack[-1] == 0:
                stack.pop()

        token_ids = generated.squeeze(0).tolist()
        decoded_tokens = self.tokenizer.decode(token_ids)
        decoded = self.tokens_to_tree(decoded_tokens, function_set)
        tree_str = stringify_tree(decoded)
        primitive_tree = PrimitiveTree.from_string(tree_str, pset)
        return [primitive_tree]

    # ====================================================
    # Convert flat token sequence to nested tuple syntax tree
    # ====================================================
    def tokens_to_tree(self, tokens, function_set):
        def helper(index):
            token = tokens[index]
            # print("index: ", index)
            # print("token: ", token)
            if token in function_set:
                left, next_idx = helper(index + 1)
                right, final_idx = helper(next_idx)
                return (token, left, right), final_idx
            else:
                return token, index + 1

        assert tokens[0] == 'START'
        tree, _ = helper(1)  # skip START
        return tree


    def validate_tokens(self, tokens, function_set):
        """
        Validates that the token sequence from START forms a valid prefix binary tree,
        even if there are extra tokens after the first valid tree.
        """

        def helper(index):
            if index >= len(tokens) - 1:  # Stop before END
                return None, index

            token = tokens[index]
            if token in function_set:
                left, next_idx = helper(index + 1)
                if left is None:
                    return None, next_idx
                right, final_idx = helper(next_idx)
                if right is None:
                    return None, final_idx
                return (token, left, right), final_idx
            elif token in self.tokenizer.vocab:
                return token, index + 1
            else:
                return None, index

        if tokens[0] != 'START':
            return False

        tree, final_idx = helper(1)
        return tree is not None


def main_train(dataset_name, rule_type, seed_offset=0, save_original=False, epochs=50):
    seed = 777

    # =======================Training=======================
    # task information
    # Example: each task gets a 16-dim embedding
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    task_embedding_dict = {
        "task_0": [0.75, 6],
        "task_1": [0.85, 8],
        "task_2": [0.95, 10]
    }
    task_emb_dim = len(task_embedding_dict["task_0"])

    # Load multitask dataset
    dataset_raw = HeuristicDataset(dataset_name, rule_type=rule_type, archive_count=3,
                                   task_embedding_dict=task_embedding_dict,
                                   seed_offset=seed_offset)
    # dataset_raw.remove_duplicates_each_task()
    trees = dataset_raw.data
    task_names = dataset_raw.tasks
    task_tree_dict, index_to_task = dataset_raw.get_taskwise_trees()

    task_names_no_repeat = [index_to_task[i] for i in sorted(index_to_task)]
    trees_per_task = [task_tree_dict[i] for i in sorted(index_to_task)]

    print("Task Names:", task_names_no_repeat)
    print("Number of Trees per Task:", [len(trees) for trees in trees_per_task])

    tokenizer = SymbolicTokenizer(function_set, terminal_set)
    dataset = SymbolicTreeDataset(trees, tokenizer, max_len=128,
                                  task_names=task_names,  # Same length as trees
                                  task_embedding_dict=task_embedding_dict)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    transformerRouting = TransformerTree(seed, dataset_name,
                                         rule_type=rule_type,
                                         epochs=epochs, lr=1e-3,
                                         max_len=128,
                                         task_emb_dim=task_emb_dim)
    transformerRouting.train(dataloader)
    transformerRouting.save_model()
    if save_original:
        transformerRouting.save_model(suffix="original")

    # =======================Test=======================
    # transformerRouting = TransformerTree(seed, dataset_name, rule_type="routing", epochs=10, lr=1e-3, max_len=100)
    # transformerRouting.load_model()
    # transformerRouting.set_seed(seed)

    task_emb = torch.tensor(task_embedding_dict['task_0'], dtype=torch.float32)
    print("\nGenerated Task Embedding: ", task_emb)

    # Generate a tree
    generated_tokens, valid = transformerRouting.sample_valid_tree(task_emb)
    print("\nGenerated Tokens: ", generated_tokens)
    print("Valid: ", valid)

    if not valid:
        print("Generated tree is invalid; skip demo tree parsing/drawing.")
        return

    # Convert to nested tree
    nested_tree = transformerRouting.tokens_to_tree(generated_tokens, function_set)
    print("Tree 0: ", nested_tree)
    print("Tree 0 length: ", len(generated_tokens))
    DrawTree.draw_tree(nested_tree, filename="Tree0")


# ================
# Example Usage
# ================

if __name__ == '__main__':
    dataset_name = "Fmean-0.75-0.85-0.95-1.5"
    rule_type = "routing"
    main_train(dataset_name, rule_type)
