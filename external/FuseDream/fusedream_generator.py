import sys
sys.path.insert(0, "./external/FuseDream")
sys.path.insert(0, "./external/FuseDream/BigGAN_utils")

import torch
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision
import BigGAN_utils.utils as utils
from external.TransformerMMExplainability.CLIP import clip
import torch.nn.functional as F
from DiffAugment_pytorch import DiffAugment
import numpy as np
from fusedream_utils import FuseDreamBaseGenerator, FuseDreamBaseGeneratorStyleGAN, get_G, save_image

parser = utils.prepare_parser()
parser = utils.add_sample_parser(parser)
args = parser.parse_args()

INIT_ITERS = 1000
OPT_ITERS = 1000
NUM_BASIS=10

utils.seed_rng(args.seed) 

sentence = args.text

print('Generating:', sentence)
G, config = get_G(512, stylegan=False) # Choose from 256 and 512
generator = FuseDreamBaseGenerator(G, config, 10) 
# G, config = get_G(512, stylegan=True) # Choose from 256 and 512
# generator = FuseDreamBaseGeneratorStyleGAN(G, config, 10) 
z_cllt, y_cllt = generator.generate_basis(sentence, 
                                          init_iters=INIT_ITERS, 
                                          num_basis=NUM_BASIS,
                                          desired_words = args.desired_words,
                                          expl_lambda=args.expl_lambda)

z_cllt_save = torch.cat(z_cllt).cpu().numpy()
y_cllt_save = torch.cat(y_cllt).cpu().numpy()

img, z, y = generator.optimize_clip_score(z_cllt,
                                          y_cllt, 
                                          sentence, 
                                          latent_noise=True, 
                                          augment=True, 
                                          opt_iters=OPT_ITERS, 
                                          optimize_y=True,
                                          desired_words = args.desired_words,
                                          expl_lambda=args.expl_lambda)
score = generator.measureAugCLIP(z, y, sentence, augment=True, num_samples=20)
print('AugCLIP score:', score)
import os
if not os.path.exists('./external/FuseDream/samples'):
    os.mkdir('./external/FuseDream/samples')
if not os.path.exists('./external/FuseDream/samples/fusedream_%s_desired_words_%s_seed_%d_score_%.4fexpl_%.4f/'%(sentence, args.desired_words, args.seed, score, args.expl_lambda)):
    os.mkdir('./external/FuseDream/samples/fusedream_%s_desired_words_%s_seed_%d_score_%.4fexpl_%.4f/'%(sentence, args.desired_words, args.seed, score, args.expl_lambda))
save_image(img, './external/FuseDream/samples/fusedream_%s_desired_words_%s_seed_%d_score_%.4fexpl_%.4f/output.png'%(sentence, args.desired_words, args.seed, score, args.expl_lambda))
img_init = generator.G(z_cllt[0], y_cllt[0])
save_image(img_init, './external/FuseDream/samples/fusedream_%s_desired_words_%s_seed_%d_score_%.4fexpl_%.4f/basis.png'%(sentence, args.desired_words, args.seed, score, args.expl_lambda), n_per_row=5)