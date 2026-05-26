import torch

def autoregressive_rollout(model, x_full, split_idx):
    model.eval()
    predictions = []

    # 1) Run RNN on prefix (teacher forcing)
    prefix = x_full[:, :split_idx, :]   # [1, split_idx, N]
    out = model(prefix)
    known_sequence = model.fc(out)         # predictions for the prefix
    predictions.append(known_sequence)

    # 2) Start autoregression from the last hidden state
    last_state = prefix[:, -1:, :]      # last *input* (ground truth)
    for t in range(split_idx, x_full.size(1)-1):
        out, hidden = model.rnn(last_state, hidden)
        next_pred = model.fc(out)       # [1, 1, N]
        predictions.append(next_pred)
        last_state = next_pred          # feed prediction back in

    return torch.cat(predictions, dim=1)      # [1, T-1, N]
