## Predict language of a name
This model will predict the origin of a given name.

```
INPUT: A single name
Batsakis

OUTPUT: Possible match with origin
(-0.17) Greek
(-2.43) Russian
(-3.92) Dutch
```


### Tools Used
- Python
- PyTorch
- Matplotlib

### Concepts covered
- RNN (Recurrent Neural Network)
- One-hot encoded tensors
- Memory cell
- Dynamic computation graphs
- Multi-class text classification

### Details
We built and trained a recurrent neural network to predict the language given the name of an individual. We got our recurrent neural network to operate on individual characters rather than words. We fed in the characters as the input sequence. We do not generate word embeddings when we work with character data. Instead, we represented these characters as one-hot encoded tensors. This involved building up a vocabulary of the characters that we would use, upper and lowercase English alphabet, plus a few special characters. we hand rolled our own memory cell where we kept track of the output state and the hidden state at every time instance.

This was possible only because of the dynamic computation graphs that PyTorch offers. This allowed us to feed in words of different lengths while training our model without padding the words to be of the same length. The resulting model that we built for language prediction performed multi-class text classification.
