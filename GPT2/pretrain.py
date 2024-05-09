from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as f
from config import GPT2Config
from dataset import GPT2Dataset, causal_mask, get_or_build_tokenizer

# Huggingface datasets and tokenizers
from model import build_gpt2
from rich.progress import track
from torch.utils.data import DataLoader


# Find the latest weights file in the weights folder
def latest_weights_file_path(config: GPT2Config) -> Path:
    model_folder = config.model_weights_path
    model_filename = f"{config.model_name}*"
    weights_files = list(model_folder.glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return weights_files[-1]


def greedy_decode(model, input, mask, tokenizer, seq_len, device):

    decoder_input = (
        torch.empty(1, seq_len)
        .fill_(tokenizer.token_to_id("[PAD]"))
        .type_as(input)
        .to(device)
    )
    decoder_input[0][0 : seq_len // 2] = input[0][0 : seq_len // 2]
    while True:

        if decoder_input.size(1) == seq_len:
            break

        # build mask for target
        decoder_mask = causal_mask(input.size(1)).type_as(mask).to(device)

        out_prob = model(decoder_input, decoder_mask)[:, -1]
        _, next_word = torch.max(out_prob, dim=1)

        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(input).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

    return decoder_input.squeeze(0)


def run_validation(
    model, val_data, tokenizer, device, seq_len, num_examples=2
):
    model.eval()
    count = 0

    source_texts = []
    predicted_texts = []

    console_width = 80

    with torch.no_grad():
        for batch in val_data:
            input = torch.Tensor(batch["input"]).to(device)  # (B, seq_len)
            mask = torch.Tensor(batch["mask"]).to(device)  # (B, 1, seq_len, seq_len)

            assert input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, input, mask, tokenizer, seq_len, device)

            source_text = tokenizer.decode(input.cpu().numpy()[0])
            model_out_text = tokenizer.decode(model_out.cpu().numpy())

            source_texts.append(source_text)
            predicted_texts.append(model_out_text)

            # Print the source, target and model output
            print("-" * console_width)
            print(f"{f'SOURCE: ':>12}{source_text}")
            print(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print("-" * console_width)
                break


def train_model(config: GPT2Config):
    # Define the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    )
    print("Using device:", device)
    device = torch.device(device)

    # Make sure the weights folder exists
    config.model_weights_path.mkdir(parents=True, exist_ok=True)

    # Get Datasets
    tokenizer = get_or_build_tokenizer(config.train_path, config.tokenizer_path)
    train_dataset = GPT2Dataset(config.seq_len, config.train_path, tokenizer)
    test_dataset = GPT2Dataset(config.seq_len, config.test_path, tokenizer)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    model = build_gpt2(
        tokenizer.get_vocab_size(),
        config.d_model,
        config.d_mlp,
        config.seq_len,
        config.heads,
        config.N,
        config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    model_filename = (
        latest_weights_file_path(config)
        if config.preload == "latest"
        else (
            config.model_weights_path / f"{config.model_name}{config.preload}.pt"
            if config.preload
            else None
        )
    )
    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    loss_fn = nn.NLLLoss(ignore_index=tokenizer.token_to_id("[PAD]")).to(device)

    for epoch in range(initial_epoch, config.epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = track(
            train_dataloader, description=f"Processing Epoch {epoch:02d}"
        )
        for batch in batch_iterator:

            input = torch.Tensor(batch["input"]).to(device)  # (B, seq_len)
            mask = torch.Tensor(batch["mask"]).to(device)  # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            decoder_output = model(input, mask)  # (B, seq_len, vocab_size)

            # (B, seq_len, vocab_size) -> (B * seq_len, vocab_size)
            decoder_output = decoder_output.view(-1, decoder_output.size(-1))
            # (B, seq_len) -> (B * seq_len)
            input = input.view(-1)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(decoder_output, input)
            # print({"loss": f"{loss.item():6.3f}"})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(
            model,
            test_dataloader,
            tokenizer,
            device,
            config.seq_len,
        )

        # Save the model at the end of every epoch
        model_filename = (
            config.model_weights_path / f"{config.model_name}{epoch:02d}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    config = GPT2Config()
    train_model(config)
