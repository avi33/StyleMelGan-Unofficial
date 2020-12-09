# StyleMelGan-Unofficial
This is an unofficial pytorch implementation of stylemelgan - https://arxiv.org/pdf/2011.01557.pdf
The repo is mainly https://github.com/descriptinc/melgan-neurips style
PQMF was used from https://github.com/kan-bayashi/ParallelWaveGAN repo

Several parameters are differ from the paper - segment length, window length in RWD

To do:
1. softmax gating is not working --> replaced with sigmoid gating
2. Discrminitor training still not tested
