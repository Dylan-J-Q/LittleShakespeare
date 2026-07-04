import torch 
import torch.nn as nn
from preprocessing import CharTokenizer, ShakespeareDataset
from model_components import EmbeddingLayer
from torch.utils.data import DataLoader

def main():
    raw_text = "To be or not to be, that is the question."
    unk_text = "This text contains characters not in the original dataset: 1234567890"
    # with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
    #     raw_text = f.read()

    tokenizer = CharTokenizer(raw_text)

    print(tokenizer.encode(raw_text))
    print(tokenizer.decode(tokenizer.encode(raw_text)))
    print(tokenizer.encode(unk_text))
    print(tokenizer.decode(tokenizer.encode(unk_text)))

    dataset = ShakespeareDataset(raw_text, tokenizer)

    #loader = DataLoader(dataset, batch_size=32, shuffle=True)

    #print(len(dataset.data))


if __name__ == "__main__":
    main()