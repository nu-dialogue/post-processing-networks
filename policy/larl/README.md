# LaRL
## Setup
1. Avoid using lambda tokenize function for multiprocessing pickle error.
    - define tokenize function out of `NormMultiWozCorpus()` on `ConvLab-2/convlab2/policy/larl/multiwoz/corpora_inference.py`.
        ```python
        def tokenize(x):
            return x.split()
        ```
    - use pre defined tokenize function instead of lambda function.
        ```python
        self.tokenize = lambda x: x.split() # remove
        self.tokenize = tokenize # add
        ```
2. Add exception handling in `Pack() > __getattr__()` method on `ConvLab-2/convlab2/policy/larl/multiwoz/latent_dialog/utils.py` to aviod pickle error.
    - from
        ```python
        def __getattr__(self, name):
            return self[name]
        ```
    - to 
        ```python
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)
        ```
3. Change using module name from nn.functional.* to torch.* on `ConvLab-2/convlab2/policy/larl/multiwoz/latent_dialog/enc2dec/decoders.py` for deprecated warning.
    - from
        ```python
        F.sigmoid
        F.tanh
        ```
    - to
        ```python
        th.simoid
        th.tanh
        ```