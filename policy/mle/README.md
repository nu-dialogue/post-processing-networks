# MLE Policy
## Prepare the pretrained model
1. Go to MLE Policy directory in ConvLab-2
    ```bash
    $ cd ../../ConvLab-2/convlab2/policy/mle
    ```
2. Make `save` direcotry & Download and unzip pretrained model
    ```bash
    $ mkdir save && cd save
    $ curl -OL https://convlab.blob.core.windows.net/convlab-2/mle_policy_multiwoz.zip
    $ unzip mle_policy_multiwoz.zip
    ```