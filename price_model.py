import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from configs import *


class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()

        self.seq_emb = nn.Embedding(vocab_size, d_model)

        position_idx = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(-1)
        position_emb_fill = position_idx * torch.exp(-torch.arange(0, d_model, 2) * math.log(10000.0) / d_model)
        pos_encoding = torch.zeros(max_seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position_emb_fill)
        pos_encoding[:, 1::2] = torch.cos(position_emb_fill)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):  # x: (batch_size,seq_len)
        # x = self.seq_emb(x)  # x: (batch_size,seq_len,dim), 对于nn.Embedding可能需要整数去匹配vocabulary
        x = x + self.pos_encoding.unsqueeze(0)[:, :x.size()[1], :]  # x: (batch_size,seq_len,dim)
        return x


class price_GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, max_seq_len):
        super(price_GPT, self).__init__()

        self.emb = EmbeddingWithPosition(vocab_size, d_model, max_seq_len)

        self.decoder_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, n_head, d_ff, batch_first=True) for _ in range(NUM_LAYER)
        ])

        # self.linear1 = nn.Linear(d_model, vocab_size)
        self.linear1 = nn.Linear(d_model, 1)  # Output shape: [batch_size, seq_len, 1]

    def forward(self, x):
        src_mask = torch.triu(torch.ones(x.size()[1], x.size()[1]), diagonal=1).type(torch.bool).to(x.device)
        x = self.emb(x)
        for block in self.decoder_blocks:
            # x = block(x, x, padding_mask)
            x = block(x, x, src_mask)

        logits = self.linear1(x)
        return logits


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    import time


    class TimeSeriesDataset(Dataset):
        def __init__(self, data, window_size):
            self.sample = []
            self.target = []
            for i in range(len(data) - window_size):
                self.sample.append(data[i:i + window_size])
                self.target.append(data[i + window_size])

        def __len__(self):
            return len(self.target)

        def __getitem__(self, index):
            x = self.sample[index]
            x = (torch.tensor(x, dtype=torch.float32)).unsqueeze(0)

            y = self.target[index]
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
            # print(f"x shape ==> {x.shape}, y shape ==> {y.shape}")
            return x, y


    # def generate_text(model, prompt, device, max_length=50, temperature=1.0):
    #     model.eval()
    #     prompt_ids = torch.tensor(prompt)
    #     generated_ids = prompt_ids
    #
    #     for _ in range(max_length):
    #         with torch.no_grad():
    #             outputs = model(generated_ids)
    #             logits = outputs[:, -1, :]
    #         logits = logits / temperature
    #         probabilities = torch.softmax(logits, dim=-1)
    #         next_token_id = torch.multinomial(probabilities, 1).item()
    #         generated_ids = torch.cat((generated_ids, torch.tensor([[next_token_id]], device=device)), dim=1)
    #         # if next_token_id == torch.tensor([102]):
    #         #     break
    #
    #     generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    #     return generated_text

    print(f"Using device ==> {DEVICE}")
    model = price_GPT(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_head=N_HEAD, d_ff=D_FF, max_seq_len=MAX_SEQ_LEN)
    model.train()
    model.to(DEVICE)
    total_param = sum([param.nelement() for param in model.parameters()])
    print(f"Model created, model parameters = {total_param}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    data_sample = np.array([np.sin(_) for _ in range(1000)])
    window_size = 6
    dataset = TimeSeriesDataset(data_sample, window_size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print("Dataset & DataLoader ready")

    print("Start training")
    for epoch in tqdm(range(NUM_EPOCH)):
        running_loss = 0.0
        for batch in dataloader:
            ids1, ids2 = batch[0].to(DEVICE), batch[1].to(DEVICE)
            # print(f"ids1 ==> {ids1.shape}, ids2 ==> {ids2.shape}")
            logits = model(ids1)

            loss = criterion(logits.squeeze(-1), ids2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{NUM_EPOCH}, Loss: {running_loss / len(dataloader)}")
        time.sleep(0.5)
    torch.save(model, "./price_model_v1.pth")
    print("Training stop, now evaluating")
    model.eval()

    init_set = data_sample.astype(np.float32)
    print(f"Initial set ==> {init_set}")
    for i in range(50):
        with torch.no_grad():
            eval_input = init_set[-1 * window_size:]
            eval_input = torch.tensor(eval_input).unsqueeze(0).to(DEVICE)

            eval_output = model(eval_input)
            prediction = torch.mean(eval_output).cpu().numpy()
            print(f"{i + 1}-th eval_input ==> {eval_input.squeeze(0).cpu().numpy()} prediction ==> {prediction}")
            init_set = np.append(init_set, prediction)

    for _ in range(50):
        data_sample = np.append(data_sample, 0)
    print(f"data_sample len ==> {len(data_sample)}")
    history_tick = np.linspace(0, len(data_sample), len(data_sample))
    future_tick = np.linspace(0, len(init_set), len(init_set))
    plt.plot(history_tick, data_sample, label="history")
    plt.plot(future_tick, init_set, label="future")
    plt.grid()
    plt.legend()
    plt.show()