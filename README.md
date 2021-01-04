# StyleMelGan-Unofficial
This is an unofficial pytorch implementation of stylemelgan - https://arxiv.org/pdf/2011.01557.pdf<br/>
The repo is mainly https://github.com/descriptinc/melgan-neurips style <br/>
PQMF and stft-losses was used from https://github.com/kan-bayashi/ParallelWaveGAN repo <br/>

Several parameters are differ from the paper - segment length, window length in RWD

To do:
1. softmax gating is not working --> replaced with sigmoid gating<br/>
#2. Discrminitor training still not tested <br/> -
2. Discriminitor from universal melgan produced better results - combination of time and frequency domain discriminators
