import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
import time
import argparse
from pathlib import Path
from dataset.dataset import AudioDataset
from modules.generator import Generator
from modules.helper_functions import save_sample
from modules.stft import Audio2Mel
from modules.stft_losses import MultiResolutionSTFTLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default='logs/inv')
    #parser.add_argument("--save_path", required=True)
    parser.add_argument("--load_path", default=None)

    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--downsamp_factor", type=int, default=4)   
    parser.add_argument("--data_path", default=None, type=Path)
    #parser.add_argument("--data_path", default=None, type=Path)    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=8192)

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--n_test_samples", type=int, default=8)    
    args = parser.parse_args()
    return args


def main():    
    args = parse_args()        
    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None
    print(load_root)
    root.mkdir(parents=True, exist_ok=True)

    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))

    #######################
    # Load PyTorch Models #
    #######################
    netG = Generator(args.n_mel_channels).cuda()
    fft = Audio2Mel(n_mel_channels=args.n_mel_channels, mel_fmin=40, mel_fmax=None, sampling_rate=22050).cuda()

    print(netG)

    #####################
    # Create optimizers #
    #####################
    optG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

    if load_root and load_root.exists():
        netG.load_state_dict(torch.load(load_root / "best_netG.pt"))
        optG.load_state_dict(torch.load(load_root / "optG.pt"))        
        print('checkpoints loaded')

    #######################
    # Create data loaders #
    #######################
    train_set = AudioDataset(
        Path(args.data_path) / "train_files.txt", args.seq_len, sampling_rate=22050
    )
    test_set = AudioDataset(
        Path(args.data_path) / "test_files.txt",
        ((22050*4//256)//32)*32*256,
        sampling_rate=22050,
        augment=False,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1)

    mr_stft_loss = MultiResolutionSTFTLoss().cuda()
    ##########################
    # Dumping original audio #
    ##########################
    test_voc = []
    test_audio = []
    for i, x_t in enumerate(test_loader):
        x_t = x_t.cuda()
        s_t = fft(x_t).detach()

        test_voc.append(s_t.cuda())
        test_audio.append(x_t.cpu())

        audio = x_t.squeeze().cpu()
        save_sample(root / ("original_%d.wav" % i), 22050, audio)
        writer.add_audio("original/sample_%d.wav" % i, audio, 0, sample_rate=22050)

        if i == args.n_test_samples - 1:
            break

    costs = []
    start = time.time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True
    best_mel_reconst = 1000000
    steps = 0
    for epoch in range(1, args.epochs + 1):
        for iterno, x_t in enumerate(train_loader):            
            x_t = x_t.cuda()            
            s_t = fft(x_t).detach()
            n = torch.randn(x_t.shape[0], 128, 1).cuda()
            x_pred_t = netG(s_t.cuda(), n)            
            
            ###################
            # Train Generator #
            ###################            
            with torch.no_grad():
                s_pred_t = fft(x_pred_t.detach())
                s_error = F.l1_loss(s_t, s_pred_t).item()
                
            sc_loss, mag_loss = mr_stft_loss(x_pred_t, x_t)
            
            loss_G = sc_loss + mag_loss
            
            netG.zero_grad()
            loss_G.backward()
            optG.step()

            ######################
            # Update tensorboard #
            ######################
            costs.append([loss_G.item(), sc_loss.item(), mag_loss.item(), s_error])
            
            writer.add_scalar("loss/generator", costs[-1][0], steps)
            writer.add_scalar("loss/spectral_convergence", costs[-1][1], steps)
            writer.add_scalar("loss/log_spectrum", costs[-1][2], steps)
            writer.add_scalar("loss/mel_reconstruction", costs[-1][3], steps)
            steps += 1

            if steps % args.save_interval == 0:
                st = time.time()
                with torch.no_grad():
                    for i, (voc, _) in enumerate(zip(test_voc, test_audio)):
                        n = torch.randn(1, 128, 10).cuda()
                        pred_audio = netG(voc, n)
                        pred_audio = pred_audio.squeeze().cpu()
                        save_sample(root / ("generated_%d.wav" % i), 22050, pred_audio)
                        writer.add_audio(
                            "generated/sample_%d.wav" % i,
                            pred_audio,
                            epoch,
                            sample_rate=22050,
                        )

                torch.save(netG.state_dict(), root / "netG.pt")
                torch.save(optG.state_dict(), root / "optG.pt")
                                
                if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]                    
                    torch.save(netG.state_dict(), root / "best_netG.pt")

                print("Took %5.4fs to generate samples" % (time.time() - st))
                print("-" * 100)

            if steps % args.log_interval == 0:
                print(
                    "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iterno,
                        len(train_loader),
                        1000 * (time.time() - start) / args.log_interval,
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time.time()


if __name__ == "__main__":
    main()
