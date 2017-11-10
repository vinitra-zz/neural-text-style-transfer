# neural-text-style-transfer
Style Transfer for non-parallel text

We propose to explore the space of neural natural language generation (NNLG) by transferring writing style. Most of the previous work in this field focuses on controlling content instead of attempting to learn stylistic features. We aim to bridge this gap with an approach based on to neural image style transfer networks, where the strategy often includes a set of layers designated to learn content (i.e. objects in the images and larger structure trends) and another set of layers designated to learn style (i.e. the RGB values for individual pixels). This work is also inspired from related problems such as author disambiguation and neural machine translation. We can use these fields to explore a variety of problem formulations and approaches in NNLG to lead to a more general way of style transfer that diverges from recent work that focuses on manually extracted stylistic features. Our main challenge will be separating stylistic from content-based features.

## Task

Given an input sequence (that could be a sentence or even a paragraph) and a target “style”, we want to generate an output
sequence. This output sequence, while retaining important information presented in the input sequence, “rewrites” the input sequence in the target “style”.

## Data 

Twitter Influencers

## Literature Review

The paper that we found to be most closely aligned to our research goal of style transfer for
non-parallel text is Style Transfer from Non-Parallel Text by Cross-Alignment by Shen et. al. The
MIT CSAIL team formulates the technical challenge of neural style transfer as separating the
content from text characteristics using a variational autoencoder, a discrimination network, and
a decoding RNN. The first step in the process is identifying a content representation for a given
sentence, which Shen et. al. attempt by minimizing the reconstruction error for a sentence
(condensing the representation and then expanding it back to its original form with the highest
possible accuracy). The problem with this approach is that the autoencoder might learn style
features, and we want to guarantee it only learns content.

The solution proposed by the CSAIL team is a discriminator neural network that takes the latent
variable z from the autoencoder and attempts to predict the initial sentence style it was
extracted from. If our discriminator network is good at identifying the input style, then we know
that z encodes some sort of stylistic representation, which was not our intention. Therefore, we
jointly optimize over the variational autoencoder and the discriminator network such that the the
latent variable z is effective in minimizing reconstruction error as well as maximizing loss for the
discriminator network, and the discriminator network has the highest accuracy it can obtain.
How do we jump from the latent z content variable to stylized output? CSAIL uses a third neural
network (decoder RNN) that outputs content in the target style given z and the target style as
input parameters.

Other paper reviews can be found in the "reports" folder of this repository.