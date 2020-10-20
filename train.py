# ! pip install torch==1.6.0

import torch
import torch.nn as nn
import torch.nn.functional as F

xor_data = [
    [[1, 1], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[0, 0], 0]
]

class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([[0.1, -0.1], [-0.1, 0.1]]))
        self.w2 = nn.Parameter(torch.tensor([[0.1, -0.1], [-0.1, 0.1]]))
        self.b1 = nn.Parameter(torch.tensor([0.1, -0.1]))
        self.b2 = nn.Parameter(torch.tensor([-0.1, 0.1]))

    def forward(self, x):
        x = F.linear(x, weight=self.w1, bias=self.b1)
        x = torch.sigmoid(x)
        x = F.linear(x, weight=self.w2, bias=self.b2)
        x = torch.sigmoid(x)
        return x

epoch = 1000
log_interval = 100
learning_rate = 1

model = XORNet()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for e_index in range(1, epoch+1):
    for index, data in enumerate(xor_data):
        _input = torch.tensor(data[0]).to(torch.float)
        target = torch.zeros(2).scatter_(0, torch.tensor(data[1]), 1).to(torch.float)

        output = model(_input)
        loss = F.binary_cross_entropy(output, target)
        optimizer.zero_grad()
        if e_index == 1 and index == 0:
            print(f"\nweight(before): {model.state_dict()}")

        loss.backward()
        optimizer.step()

        if e_index == 1 and index == 0:
            print(f"\nweight(after): {model.state_dict()}]\n")
            for name, parms in model.named_parameters():
                print(f"{name} grad ({parms.requires_grad}): {parms.grad}")

        if not e_index % log_interval:
            print(f"epoch: {e_index} input: {_input.data} output: {output.data}, target: {target.data} loss: {loss.data}")
