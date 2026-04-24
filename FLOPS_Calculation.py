
import torch
import torch.nn as nn

# -----------------------------
# MODEL DEFINITION
# -----------------------------
class ConvBiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # BiLSTM layers
        self.lstm1 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)

    def forward(self, x):
        # Conv
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)

        x = torch.relu(self.conv5(x))

        # Convert to sequence
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, T, C)

        # LSTM
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)

        return x


# -----------------------------
# FLOPs FUNCTIONS
# -----------------------------

def conv_flops(layer, x, out):
    Cin = layer.in_channels
    Cout = layer.out_channels
    Kh, Kw = layer.kernel_size
    Hout, Wout = out.shape[2], out.shape[3]

    flops = Hout * Wout * Cout * (Cin * Kh * Kw)
    return flops


def lstm_flops(layer, x, out):
    # x shape: (B, T, D)
    B, T, D = x[0].shape
    H = layer.hidden_size

    # FLOPs per timestep
    flops_per_timestep = 4 * (D * H + H * H)

    # Bidirectional => x2
    flops = 2 * T * flops_per_timestep
    return flops


# -----------------------------
# HOOKS TO CAPTURE FLOPs
# -----------------------------
layer_flops = {}

def register_hooks(model):
    for name, layer in model.named_modules():

        if isinstance(layer, nn.Conv2d):
            def hook(layer, inp, out, name=name):
                flops = conv_flops(layer, inp[0], out)
                layer_flops[name] = {
                    "FLOPs": flops,
                    "Output Shape": list(out.shape)
                }
            layer.register_forward_hook(hook)

        elif isinstance(layer, nn.LSTM):
            def hook(layer, inp, out, name=name):
                flops = lstm_flops(layer, inp, out)
                layer_flops[name] = {
                    "FLOPs": flops,
                    "Output Shape": list(out[0].shape)
                }
            layer.register_forward_hook(hook)


# -----------------------------
# RUN MODEL
# -----------------------------
model = ConvBiLSTMModel()
register_hooks(model)

x = torch.randn(1, 3, 32, 32)
model(x)

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\nLayer-wise FLOPs:\n")
for name, info in layer_flops.items():
    print(f"{name}:")
    print(f"   FLOPs = {info['FLOPs']}")
    # print(f"   Input Shape = {info['Input Shape']}\n")
    print(f"   Output Shape = {info['Output Shape']}\n")
