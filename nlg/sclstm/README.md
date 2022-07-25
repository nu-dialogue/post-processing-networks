# SC-LSTM
## Setup
1. Change using module name from nn.functional.* to torch.* on `ConvLab-2/convlab2/nlg/sclstm/model/layers/decoder_deep.py` for deprecated warning.
    - from
        ```python
        F.sigmoid
        F.tanh
        ```
    - to
        ```python
        torch.simoid
        torch.tanh
        ```