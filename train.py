import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import logging
import os

from data_prep import Password as P
from model import Generator, Discriminator
from training_helper import *
from config import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

g = Generator(GEN_HIDDEN_SIZE, GEN_NEURON_SIZE).to(device)
d = Discriminator(DISC_HIDDEN_SIZE, DISC_NEURON_SIZE).to(device)

opt_g = torch.optim.RMSprop(g.parameters(), lr=0.0002)
opt_d = torch.optim.RMSprop(d.parameters(), lr=0.0002)

p = P()
batch_gen = p.string_gen

TRAIN_FROM_CKPT = True

if os.path.isfile(CHECKPOINT_PATH) and TRAIN_FROM_CKPT:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location = device)
    g.load_state_dict(checkpoint['gen_state_dict'])
    d.load_state_dict(checkpoint['disc_state_dict'])
    g = g.to(device)
    d = d.to(device)
    opt_g.load_state_dict(checkpoint['gen_optimizer_state_dict'])
    opt_d.load_state_dict(checkpoint['disc_optimizer_state_dict'])
    start_len = checkpoint['seq_len']
    start_iter = checkpoint['iter']
else:
    start_len = 1
    start_iter = 1

for seq_len in range(start_len, MAX_LEN + 1):
    logging.info("---------- Adversarial Training with Seq Len %d, Batch Size %d ----------\n" % (seq_len, BATCH_SIZE))
    
    for i in range(start_iter, ITERS_PER_SEQ_LEN + 1):
                
        if i % SAVE_CHECKPOINTS_EVERY == 0:
            torch.save({
                'seq_len': seq_len,
                'gen_state_dict': g.state_dict(),
                'disc_state_dict': d.state_dict(),
                'gen_optimizer_state_dict': opt_g.state_dict(),
                'disc_optimizer_state_dict': opt_d.state_dict(),
                'iter': i
                }, CHECKPOINT_PATH)
            logging.info("  *** Model Saved on Iter " + str(i) + " ***\n")
        
        logging.debug("----------------- %d / %d -----------------\n" % (i, ITERS_PER_SEQ_LEN))

        logging.debug("Training discriminator...\n")
        
        d.requiresGrad()
        d.zero_grad()
        g.zero_grad()
        for j in range(CRITIC_ITERS):
            with torch.backends.cudnn.flags(enabled=False):
                L = 0
                
                data = next(batch_gen)
                pred = g(data, seq_len)
                real, fake = get_train_dis(data, pred, seq_len)
                interpolate = get_interpolate(real, fake)

                # Genuine
                disc_real = d(real, seq_len)
                loss_real = -disc_real.mean()
                logging.debug("real loss: "+str(loss_real.item()))
                L += loss_real

                # Fake
                disc_fake = d(fake, seq_len)
                loss_fake = disc_fake.mean()
                logging.debug("fake loss: "+str(loss_fake.item()))
                L += loss_fake

                # Gradient penalty
                interpolate = torch.autograd.Variable(interpolate, requires_grad=True)
                disc_interpolate = d(interpolate, seq_len)
                grad = torch.ones_like(disc_interpolate).to(device)
                gradients = torch.autograd.grad(
                        outputs=disc_interpolate,
                        inputs=interpolate,
                        grad_outputs=grad,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                loss_gp = ((gradients.norm(2, dim=2) - 1) ** 2).mean() * LAMBDA
                logging.debug("grad loss: "+str(loss_gp.item()))
                L += loss_gp

                L.backward(retain_graph=False)
                opt_d.step()
                
                logging.debug("Critic Iter " + str(j+1) + " Loss: " + str(L.item()) + "\n")


        logging.debug("Done training discriminator.\n")    

        logging.debug("Training generator...")

        d.requiresNoGrad()

        for j in range(GEN_ITERS):
            data = next(batch_gen)
            pred = g(data, seq_len)
            fake = get_train_gen(data, pred, seq_len)
            loss_gen = -d(fake, seq_len).mean()
            logging.debug("Gen Iter " + str(j+1) + " Loss: "+str(loss_gen.item()))
            loss_gen.backward(retain_graph=False)
            opt_g.step()

        logging.debug("Done training generator.\n")
    
    start_iter = 1