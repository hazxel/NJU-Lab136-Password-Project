import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import logging
import os
import time

from data_prep import Password as P
from improved_model import Generator, Discriminator
from improved_training_helper import *
from config import *

import sys
from getopt import getopt

############################
## example:          ##
## python train.py -d -c  ##
############################

logger_data = logging.getLogger("data_prep.py")
logger_data.setLevel(level=logging.DEBUG)
logger = logging.getLogger("train.py")
logger.setLevel(level=logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setLevel(level=logging.INFO)

checkpoint_path = DEFAULT_CHECKPOINT_PATH
train_from_ckpt = True

opts, args = getopt(sys.argv[1:], "-h-r-f-l:-c:", ["help", "restart", "log2file", "logging=", "checkpoint_path="])

for opt_name, opt_value in opts:
    if opt_name in ('-h', '--help'):
        print("Ooops, we don't have help info now :)")
    elif opt_name in ('-r', '--restart'):
        train_from_ckpt = False
        print("Restart training from scratch...")
    elif opt_name in ('-f', '--log2file'):
        file_handler = logging.FileHandler('./log/' + time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) + '.log')
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger_data.addHandler(file_handler)
    elif opt_name in ('-l', '--logging'):
        if opt_value == "debug":
            stream_handler.setLevel(level=logging.DEBUG)
    elif opt_name in ('-c', '--checkpoint_path'):
        checkpoint_path = opt_value   

logger.addHandler(stream_handler)
logger_data.addHandler(stream_handler)


logger.info("Training on " + str(device) + ".")

g = Generator(GEN_HIDDEN_SIZE, GEN_NEURON_SIZE, GEN_LAYERS).to(device)
d = Discriminator(DISC_HIDDEN_SIZE, DISC_NEURON_SIZE, DISC_LAYERS).to(device)

opt_g = torch.optim.RMSprop(g.parameters(), lr=0.0002)
opt_d = torch.optim.RMSprop(d.parameters(), lr=0.0002)

p = P()
batch_gen = p.string_gen

if os.path.isfile(checkpoint_path) and train_from_ckpt:
    checkpoint = torch.load(checkpoint_path, map_location = device)
    g.load_state_dict(checkpoint['gen_state_dict'])
    d.load_state_dict(checkpoint['disc_state_dict'])
    g = g.to(device)
    d = d.to(device)
    opt_g.load_state_dict(checkpoint['gen_optimizer_state_dict'])
    opt_d.load_state_dict(checkpoint['disc_optimizer_state_dict'])
    start_len = checkpoint['seq_len']
    start_iter = checkpoint['iter']
    disc_loss = checkpoint['disc_loss']
    gen_loss = checkpoint['gen_loss']
    real_loss = checkpoint['real_loss']
else:
    logger.debug("Pretraining generator...")
    logger.debug("Done.")
    start_len = 1
    start_iter = 1
    disc_loss = []
    gen_loss = []
    real_loss = []

for seq_len in range(start_len, MAX_LEN + EXTRA_LEN + 1):
    logger.info("---------- Adversarial Training with Seq Len %d, Batch Size %d ----------\n" % (seq_len, BATCH_SIZE))
    
    for i in range(start_iter, ITERS_PER_SEQ_LEN + 1):
                
        if i % SAVE_CHECKPOINTS_EVERY == 0:
            torch.save({
                'seq_len': seq_len,
                'gen_state_dict': g.state_dict(),
                'disc_state_dict': d.state_dict(),
                'gen_optimizer_state_dict': opt_g.state_dict(),
                'disc_optimizer_state_dict': opt_d.state_dict(),
                'iter': i,
                'gen_loss': gen_loss,
                'disc_loss': disc_loss,
                'real_loss': real_loss
                }, checkpoint_path)
            logger.info("         *** Model Saved on Iter " + str(i) + " ***\n")
        
        logger.debug("----------------- %d / %d -----------------\n" % (i, ITERS_PER_SEQ_LEN))

        logger.debug("Training discriminator...\n")
        
        d.requiresGrad()
        d.zero_grad()
        g.zero_grad()
        sq = MAX_LEN if seq_len > MAX_LEN else seq_len
        for j in range(CRITIC_ITERS):
            with torch.backends.cudnn.flags(enabled=False):
                L = 0
                
                data = next(batch_gen)
                fake = g(sq)
                real = get_real(data,sq)
                interpolate = get_interpolate(real, fake)

                # Genuine
                disc_real = d(real)
                loss_real = -disc_real.mean()
                logger.debug("real loss: "+str(loss_real.item()))
                L += loss_real

                # Fake
                disc_fake = d(fake)
                loss_fake = disc_fake.mean()
                logger.debug("fake loss: "+str(loss_fake.item()))
                L += loss_fake

                # Gradient penalty
                interpolate = torch.autograd.Variable(interpolate, requires_grad=True)
                disc_interpolate = d(interpolate)
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
                logger.debug("grad loss: "+str(loss_gp.item()))
                L += loss_gp

                L.backward(retain_graph=False)
                opt_d.step()
                
                logger.debug("Critic Iter " + str(j+1) + " Loss: " + str(L.item()) + "\n")


        logger.debug("Done training discriminator.\n")    

        logger.debug("Training generator...")

        d.requiresNoGrad()

        for j in range(GEN_ITERS):
            data = next(batch_gen)
            fake = g(sq)
            loss_gen = -d(fake).mean()
            logger.debug("Gen Iter " + str(j+1) + " Loss: "+str(loss_gen.item()))
            loss_gen.backward(retain_graph=False)
            opt_g.step()

        logger.debug("Done training generator.\n")

        if i % SAVE_CHECKPOINTS_EVERY == 0:
            real_loss.append(loss_real)
            disc_loss.append(L)
            gen_loss.append(loss_gen)
    
    start_iter = 1

logging.info("All Done.")
