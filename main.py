import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pickle
import time

from args import ArgParser
from datasets import ImagePairDataset
from models import UNet, Discriminator
from utils import show_tensor_images, ScoreMeter, Recorder


def train(args):
    train_set = ImagePairDataset("./data/mecs_steel", args.size,
                                 "LOM640", "SEM640", "all.txt")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    gen = UNet(input_channels=3, output_channels=1,
               hidden_channels=32).to(args.device)
    disc = Discriminator(input_channels=4,
                         hidden_channels=32,
                         level=args.patch_level).to(args.device)
    recon_criterion = nn.L1Loss().to(args.device)
    if args.adv_loss_type == 'CE':
        adv_criterion = nn.BCEWithLogitsLoss().to(args.device)
    elif args.adv_loss_type == 'LS':
        adv_criterion = nn.MSELoss().to(args.device)
    else: raise ValueError(f"{args.adv_loss_type} not exist")
    gen_opt = optim.Adam(gen.parameters(), lr=args.gen_lr)
    disc_opt = optim.Adam(disc.parameters(), lr=args.disc_lr)

    cur_step = 0
    score_meter = ScoreMeter()
    recorder = Recorder(('iter', 'gen_loss', 'gen_adv_loss', 'gen_recon_loss',
                         'disc_loss', 'disc_real_loss', 'disc_fake_loss'))
    start = time.time()
    for epoch in range(args.n_epochs):
        for real_A, real_B in train_loader:
            cur_step += 1
            real_A, real_B = real_A.to(args.device), real_B.to(args.device)
            fake_B = gen(real_A) # shape (N,1,H,W)

            # Update discriminator
            disc_opt.zero_grad()
            disc_loss, disc_real_loss, disc_fake_loss = get_disc_loss(disc, real_B, fake_B, real_A, adv_criterion)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Update generator
            gen_opt.zero_grad()
            gen_loss, gen_adv_loss, gen_recon_loss = get_gen_loss(gen, disc, real_B, real_A, adv_criterion, recon_criterion, args.lambda_recon)
            gen_loss.backward()
            gen_opt.step()

            score_meter.update(gen_loss.item(), gen_adv_loss, gen_recon_loss,
                               disc_loss.item(), disc_real_loss, disc_fake_loss, n=args.batch_size)

            if cur_step % args.display_step == 0:
                image_tensor = torch.cat([real_A[:args.n_images_row],
                                          torch.cat([real_B]*3, dim=1)[:args.n_images_row],
                                          torch.cat([fake_B]*3, dim=1)[:args.n_images_row]])
                title = f"step {cur_step}"
                show_tensor_images(image_tensor, title, nrow=args.n_images_row, num_images=args.n_images_row*3)

            if not args.no_save and cur_step % args.record_step == 0:
                print(f"epoch {epoch} | step {cur_step} | {score_meter.stats_string()} | "
                      f"time {time.time() - start:.2f}")
                recorder.update([cur_step, *score_meter.stats()])
                score_meter.reset()
                torch.save({'gen': gen.state_dict(),
                            'gen_opt': gen_opt.state_dict(),
                            'disc': disc.state_dict(),
                            'disc_opt': disc_opt.state_dict(),
                            }, f"{args.checkpoint_dir}/model_step{cur_step}.pth")
                with open(args.train_record_path, 'wb') as f:
                    pickle.dump(recorder.record, f)


def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    """
    Get generator loss (adversarial loss + reconstruction loss).
    Args:
        gen: The generator model.
        disc: The discriminator model.
        real: Target real images.
        condition: Source real images.
        adv_criterion: The adversarial loss function.
        recon_criterion: The reconstruction loss function.
        lambda_recon: The coefficient of reconstruction loss.

    Returns: Generator loss.
    """
    fake = gen(condition)
    pred = disc(fake, condition)
    gen_adv_loss = adv_criterion(pred, torch.ones_like(pred))
    gen_recon_loss = recon_criterion(fake, real)
    gen_loss = gen_adv_loss + lambda_recon * gen_recon_loss
    return gen_loss, gen_adv_loss.item(), gen_recon_loss.item()

def get_disc_loss(disc, real, fake, condition, adv_criterion):
    """
    Get discriminator loss.
    Args:
        disc: The discriminator model.
        real: Target real images.
        fake: Generated images.
        condition: Source real images.
        adv_criterion: The adversarial loss function.

    Returns: Discriminator loss.
    """
    fake_pred = disc(fake.detach(), condition)
    fake_loss = adv_criterion(fake_pred, torch.zeros_like(fake_pred))
    real_pred = disc(real, condition)
    real_loss = adv_criterion(real_pred, torch.ones_like(real_pred))
    return real_loss + fake_loss, real_loss.item(), fake_loss.item()


if __name__ == '__main__':
    arg_parser = ArgParser()
    args = arg_parser.parse_args(verbose=True)
    train(args)