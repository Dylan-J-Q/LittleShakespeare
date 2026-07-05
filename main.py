from preprocessing import CharTokenizer, ShakespeareDataset
from model_components import TransformerModel
from torch.utils.data import DataLoader
from training import Config, Trainer
import torch 

DATA_FILE_PATH = "data/LittleShakespeare.txt"
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")  # Force CPU for testing purposes


def main():
    with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Split text into training, validation, and test sets (60/20/20)
    train_size = int(len(raw_text) * 0.6)
    val_size = int(len(raw_text) * 0.2)
    
    train_text = raw_text[:train_size]
    val_text = raw_text[train_size:train_size + val_size]
    test_text = raw_text[train_size + val_size:]

    config = Config(
        d_model=512,
        num_heads=8,
        ff_hidden_dim=2048,
        learning_rate=3e-4,
        batch_size=64,
        epochs=50,
        weight_decay=0.01,
        device="cpu"
    )
    tokenizer = CharTokenizer(raw_text)

    train_dataset = ShakespeareDataset(train_text, tokenizer)
    val_dataset = ShakespeareDataset(val_text, tokenizer)
    test_dataset = ShakespeareDataset(test_text, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"Data Preprocessing Complete. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    model = TransformerModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config.d_model,
        num_heads=config.num_heads,
        ff_hidden_dim=config.ff_hidden_dim,
        num_layers=6 
    )
    model.to(DEVICE)
    
    trainer = Trainer(config, model)
    trainer.train(train_loader, val_loader)

    print("Training Complete.")


if __name__ == "__main__":
    main()