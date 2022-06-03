import torch
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision
import BigGAN_utils.utils as utils
from external.TransformerMMExplainability.CLIP import clip
import torch.nn.functional as F
from DiffAugment_pytorch import DiffAugment
import numpy as np
import lpips
from captum.attr import visualization
import os

LATENT_NOISE = 0.01
Z_THRES = 2.0
POLICY = 'color,translation,resize,cutout'
TEST_POLICY = 'color,translation,resize,cutout'
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

from external.TransformerMMExplainability.CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def get_desired_tokens_from_words(desired_words, text):
    if desired_words is None:
        num_of_tokens = len(_tokenizer.encode(text))
        desired_tokens = torch.full((num_of_tokens,), 1 / num_of_tokens)
    
    else:

        target_words = text.split(" ")
        desired_words = np.array(desired_words.split(' ')).astype(float)
        desired_tokens = torch.zeros((len(_tokenizer.encode(text))))

        token_id = 0
        for word_idx, word in enumerate(target_words):
            num_of_tokens = len(_tokenizer.encode(word))

            for t in range(num_of_tokens):
                desired_tokens[token_id] = desired_words[word_idx]
                token_id = token_id + 1

        if desired_tokens.min() != desired_tokens.max():
            desired_tokens /= desired_tokens.max()
        else:
            desired_tokens = desired_tokens / (desired_tokens > 0).sum()

    print(desired_tokens)
    
    return desired_tokens

# Explainability utils
def interpret(image, text, model, device, index=None, softmax_temp=10.):
    model.zero_grad()
    
    text = clip.tokenize([text]).to(device)
    CLS_idx = text.argmax(dim=-1)
    

    with torch.enable_grad():
        image = image.detach().clone().requires_grad_() 
        logits_per_image, logits_per_text = model(image, text)
        if index is None:
            index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
        one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(logits_per_image.device)  * logits_per_image)
        
        text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())
        num_tokens = text_attn_blocks[0].attn.attn_output_weights.shape[-1]
        R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn.attn_output_weights.dtype).to(logits_per_image.device)
        for blk_idx, blk in enumerate(text_attn_blocks):
            grad = torch.autograd.grad(one_hot, [blk.attn.attn_output_weights], retain_graph=True)[0].detach()
            cam = blk.attn.attn_output_weights.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            R_text = R_text + torch.matmul(cam, R_text)
        text_relevance = R_text[CLS_idx, 1:CLS_idx]
        text_relevance = text_relevance / text_relevance.sum()
        text_relevance = text_relevance / text_relevance.max()

    return text_relevance

def interpret_batch(image, text, model, device, index=None, softmax_temp=10.):
    model.zero_grad()

    batch_size = image.shape[0]
    text = clip.tokenize([text]).to(device)
    text = text.repeat(batch_size, 1)
    index = [i for i in range(batch_size)]
    clip_c = model.logit_scale.exp()
    model.zero_grad()

    with torch.enable_grad():
        logits_per_image, logits_per_text = model(image, text)
        logits_per_image = logits_per_image
        logits_per_image = logits_per_image / clip_c

        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(image.device) * logits_per_image)

        text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())
        num_tokens = text_attn_blocks[0].attn.attn_output_weights.shape[-1]
        R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn.attn_output_weights.dtype).to(image.device)
        R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for blk in text_attn_blocks:
            grad = torch.autograd.grad(one_hot, [blk.attn.attn_output_weights], retain_graph=True)[0]
            cam = blk.attn.attn_output_weights
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R_text = R_text + torch.bmm(cam, R_text)
        text_relevance = R_text
        text_relevance[:, [i for i in range(num_tokens)], [i for i in range(num_tokens)]] = 0
        text_relevance = text_relevance[torch.arange(text_relevance.shape[0]),
                                        text[0].argmax(dim=-1),
                                        1:text[0].argmax(dim=-1)]
        text_relevance = text_relevance / torch.sum(text_relevance, dim=-1, keepdim=True)
        text_relevance = text_relevance / text_relevance.max(dim=-1, keepdim=True)[0]

    return text_relevance

def show_heatmap_on_text(text, text_encoding, R_text):
  text_scores = R_text.flatten()
  text_tokens=_tokenizer.encode(text)
  text_tokens_decoded=[_tokenizer.decode([a]) for a in text_tokens]
  vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,text_tokens_decoded,1)]
  html = visualization.visualize_text(vis_data_records)
  return html

def explainability_loss(image, model, text_tokenized, batch_size, desired_words):
    text = text_tokenized.repeat(batch_size, 1)
    index = [i for i in range(batch_size)]
    clip_c = model.logit_scale.exp()
    model.zero_grad()

    with torch.enable_grad():
        logits_per_image, logits_per_text = model(image, text)
        logits_per_image = logits_per_image
        logits_per_image = logits_per_image / clip_c

        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(image.device) * logits_per_image)

        text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())
        num_tokens = text_attn_blocks[0].attn.attn_output_weights.shape[-1]
        R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn.attn_output_weights.dtype).to(image.device)
        R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for blk in text_attn_blocks:
            grad = torch.autograd.grad(one_hot, [blk.attn.attn_output_weights], retain_graph=True, create_graph=True)[0]
            cam = blk.attn.attn_output_weights
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R_text = R_text + torch.bmm(cam, R_text)
        text_relevance = R_text
        text_relevance[:, [i for i in range(num_tokens)], [i for i in range(num_tokens)]] = 0
        text_relevance = text_relevance[torch.arange(text_relevance.shape[0]),
                                        text_tokenized[0].argmax(dim=-1),
                                        1:text_tokenized[0].argmax(dim=-1)]
        text_relevance = text_relevance / torch.sum(text_relevance, dim=-1, keepdim=True)
        text_relevance = text_relevance / text_relevance.max(dim=-1, keepdim=True)[0]

        target_word_expl_score = torch.zeros((batch_size, 1)).to(text_relevance.device)

        for word in desired_words:
            word_expl = text_relevance[:, word]
            target_word_expl_score = target_word_expl_score + word_expl.max(dim=-1, keepdim=True)[0]
        
        target_word_expl_score = target_word_expl_score / len(desired_words)

    model.zero_grad()
    return target_word_expl_score * (-1)

def AugmentLoss(img, clip_model, text, replicate=10, interp_mode='bilinear', policy=POLICY, expl_lambda=0., desired_words=[]):

    clip_c = clip_model.logit_scale.exp()
    img_aug = DiffAugment(img.repeat(replicate, 1, 1, 1), policy=policy)
    img_aug = (img_aug+1.)/2.
    img_aug = F.interpolate(img_aug, size=224, mode=interp_mode)
    img_aug.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])


    logits_per_image, logits_per_text = clip_model(img_aug, text)
    logits_per_image = logits_per_image / clip_c
    concept_loss = (-1.) * logits_per_image

    if expl_lambda > 0:
        expl_loss = explainability_loss(image=img_aug, model=clip_model, text_tokenized=text, batch_size=replicate, desired_words=desired_words)
        concept_loss = concept_loss + expl_lambda * expl_loss
        print(f"concept loss {concept_loss.detach().mean().item()} expl loss {expl_loss.detach().mean().item()}")
        # print(concept_loss.mean().shape)
        # concept_grad = torch.autograd.grad(concept_loss.mean(), [img], retain_graph=True)[0]
        # print(f"concept grad {concept_grad.item()}")
        # expl_grad = torch.autograd.grad(expl_loss.mean(), [img_aug], retain_graph=True)[0]
        # print(f"concept loss grads {concept_grad.detach().mean().item()} expl loss {expl_grad.detach().mean().item()}")
         
    return concept_loss.mean(dim=0, keepdim=False)

def NaiveSemanticLoss(img, clip_model, text, interp_mode='bilinear'):

    clip_c = clip_model.logit_scale.exp()
    img = (img+1.)/2.
    img = F.interpolate(img, size=224, mode=interp_mode)
    img.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

    logits_per_image, logits_per_text = clip_model(img, text)
    logits_per_image = logits_per_image / clip_c
    concept_loss = (-1.) * logits_per_image 
     
    return concept_loss.mean(dim=0, keepdim=False)

def get_gaussian_mask(size=256):
    x, y = np.meshgrid(np.linspace(-1,1, size), np.linspace(-1,1,size))
    dst = np.sqrt(x*x+y*y)
      
    # Intializing sigma and muu
    sigma = 1
    muu = 0.000
      
    # Calculating Gaussian array
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    
    return gauss

def save_image(img, path, n_per_row=1):
    with torch.no_grad():
        torchvision.utils.save_image(
            torch.from_numpy(img.cpu().numpy()), ##hack, to turn Distribution back to tensor
            path,
            nrow=n_per_row,
            normalize=True,
        )

def load_vqgan_model(model_name):
    config_path = os.path.join('./pretrained_models', model_name + '.yaml')
    checkpoint_path = os.path.join('./pretrained_models', model_name + '.ckpt')

    global gumbel
    gumbel = False
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = True
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def get_StyleGAN_G():

    from external.StyleCLIP.models.stylegan2.model import Generator
    generator = Generator(256, 512, 8, channel_multiplier=1)
    checkpoint = torch.load('./external/FuseDream/BigGAN_utils/weights/stylegan-CUB-018841.pt')

    generator.load_state_dict(checkpoint["g_ema"], strict=False)
    generator = generator.eval()
    generator = generator.to('cuda')

    return generator, {}

def get_VQGAN_G(model_name="gumbel_f8-8192"):
    pretrained_model_dir = './pretrained_model_dir'
    model = load_vqgan_model(f'{pretrained_model_dir}/{model_name}.yaml', f'{pretrained_model_dir}/{model_name}.ckpt').to('cuda')

    return model, {}

def get_G(resolution=256, stylegan=False):
    if resolution == 256:
        parser = utils.prepare_parser()
        parser = utils.add_sample_parser(parser)
        config = vars(parser.parse_args())

        # See: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/scripts/sample_BigGAN_bs256x8.sh.
        config["resolution"] = utils.imsize_dict["I128_hdf5"]
        config["n_classes"] = utils.nclass_dict["I128_hdf5"]
        config["G_activation"] = utils.activation_dict["inplace_relu"]
        config["D_activation"] = utils.activation_dict["inplace_relu"]
        config["G_attn"] = "128"
        config["D_attn"] = "128"
        config["G_ch"] = 96
        config["D_ch"] = 96
        config["hier"] = True
        config["dim_z"] = 140
        config["shared_dim"] = 128
        config["G_shared"] = True
        config = utils.update_config_roots(config)
        config["skip_init"] = True
        config["no_optim"] = True
        config["device"] = "cuda"
        config["resolution"] = 256

        # Set up cudnn.benchmark for free speed.
        torch.backends.cudnn.benchmark = True

        # Import the model.
        model = __import__(config["model"])
        G = model.Generator(**config).to(config["device"])
        utils.count_parameters(G)

        # Load weights.
        weights_path = "./external/FuseDream/BigGAN_utils/weights/biggan-256.pth"  # Change this.
        G.load_state_dict(torch.load(weights_path), strict=False)
    elif resolution == 512:
        parser = utils.prepare_parser()
        parser = utils.add_sample_parser(parser)
        config = vars(parser.parse_args())

        # See: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/scripts/sample_BigGAN_bs128x8.sh.
        config["resolution"] = 512
        config["n_classes"] = utils.nclass_dict["I128_hdf5"]
        config["G_activation"] = utils.activation_dict["inplace_relu"]
        config["D_activation"] = utils.activation_dict["inplace_relu"]
        config["G_attn"] = "64"
        config["D_attn"] = "64"
        config["G_ch"] = 96
        config["D_ch"] = 64
        config["hier"] = True
        config["dim_z"] = 128
        config["shared_dim"] = 128
        config["G_shared"] = True
        config = utils.update_config_roots(config)
        config["skip_init"] = True
        config["no_optim"] = True
        config["device"] = "cuda"

        # Set up cudnn.benchmark for free speed.
        torch.backends.cudnn.benchmark = True

        # Import the model.
        model = __import__(config["model"])
        #print(config["model"])
        G = model.Generator(**config).to(config["device"])
        utils.count_parameters(G)
        #print('G parameters:')
        #for p, m in G.named_parameters():
        #    print(p)
        # Load weights.
        weights_path = "./external/FuseDream/BigGAN_utils/weights/biggan-512.pth"  # Change this.
        G.load_state_dict(torch.load(weights_path), strict=False)

    return G, config

class FuseDreamBaseGenerator():
    def __init__(self, G, G_config, G_batch_size=10, clip_mode="ViT-B/32", interp_mode='bilinear'):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.G = G
        self.clip_model, _ = clip.load(clip_mode, device=device, jit=False) 
        
        (self.z_, self.y_) = utils.prepare_z_y(
            G_batch_size,
            self.G.dim_z,
            G_config["n_classes"],
            device=G_config["device"],
            fp16=G_config["G_fp16"],
            z_var=G_config["z_var"],
        )

        self.G.eval()

        for p in self.G.parameters():
            p.requires_grad = False
        # for p in self.clip_model.parameters():
        #     p.requires_grad = False

        self.interp_mode = interp_mode 
  
    def generate_basis(self, text, init_iters=500, num_basis=5, desired_words='', desired_tokens=None, expl_lambda=0.):
        text_tok = clip.tokenize([text]).to(self.device)
        clip_c = self.clip_model.logit_scale.exp() 

        if expl_lambda > 0:
            if desired_tokens is None:
                assert desired_words, "It is not possible to use explainability loss without sspecifying desired words"
                desired_tokens = get_desired_tokens_from_words(desired_words, text)
        else:
            desired_tokens = []

        z_init_cllt = []
        y_init_cllt = []
        z_init = None
        y_init = None
        score_init = None
        with torch.no_grad():
            for i in tqdm(range(init_iters)):
                self.z_.sample_()
                self.y_.sample_()
                self.z_.data = torch.clamp(self.z_.data.detach().clone(), min=-Z_THRES, max=Z_THRES)
                z_ = torch.tensor(self.z_).detach().clone()

                image_tensors = self.G(z_, self.G.shared(self.y_))
                image_tensors = (image_tensors+1.) / 2.
                image_tensors = F.interpolate(image_tensors, size=224, mode=self.interp_mode)
                image_tensors.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
                
                logits_per_image, logits_per_text = self.clip_model(image_tensors, text_tok)
                logits_per_image = logits_per_image/clip_c

                if expl_lambda > 0:
                    expl_scores = explainability_loss(image=image_tensors, 
                                                      model=self.clip_model, 
                                                      text_tokenized=text_tok, 
                                                      batch_size=len(image_tensors), 
                                                      desired_words=desired_tokens)
                    logits_per_image = logits_per_image - expl_scores * expl_lambda



                if z_init is None:
                    z_init = self.z_.data.detach().clone()
                    y_init = self.y_.data.detach().clone()
                    score_init = logits_per_image.squeeze()
                else:
                    z_init = torch.cat([z_init, self.z_.data.detach().clone()], dim=0)
                    y_init = torch.cat([y_init, self.y_.data.detach().clone()], dim=0)
                    score_init = torch.cat([score_init, logits_per_image.squeeze()])

                sorted, indices = torch.sort(score_init, descending=True)
                z_init = z_init[indices]
                y_init = y_init[indices]
                score_init = score_init[indices]
                z_init = z_init[:num_basis]
                y_init = y_init[:num_basis]
                score_init = score_init[:num_basis]
        
        #save_image(self.G(z_init, self.G.shared(y_init)), 'samples/init_%s.png'%text, 1)
        

        z_init_cllt.append(z_init.detach().clone())
        y_init_cllt.append(self.G.shared(y_init.detach().clone()))

        return z_init_cllt, y_init_cllt


    def optimize_clip_score(self, z_init_cllt, y_init_cllt, text, latent_noise=False, augment=True, opt_iters=500, optimize_y=False, desired_words='', desired_tokens=None, expl_lambda=0.):

        if expl_lambda > 0:
            if desired_tokens is None:
                assert desired_words, "It is not possible to use explainability loss without sspecifying desired words"
                desired_tokens = get_desired_tokens_from_words(desired_words, text)
        else:
            desired_tokens = []

        text_tok = clip.tokenize([text]).to(self.device)
        clip_c = self.clip_model.logit_scale.exp()

        z_init_ans = torch.stack(z_init_cllt)
        y_init_ans = torch.stack(y_init_cllt)
        z_init_ans = z_init_ans.view(-1, z_init_ans.shape[-1])
        y_init_ans = y_init_ans.view(-1, y_init_ans.shape[-1])

        w_z = torch.randn((z_init_ans.shape[0], z_init_ans.shape[1])).to(self.device)
        w_y = torch.randn((y_init_ans.shape[0], y_init_ans.shape[1])).to(self.device)
        w_z.requires_grad = True
        w_y.requires_grad = True

        opt_y = torch.zeros(y_init_ans.shape).to(self.device)
        opt_y.data = y_init_ans.data.detach().clone()
        opt_z = torch.zeros(z_init_ans.shape).to(self.device)
        opt_z.data = z_init_ans.data.detach().clone()
        opt_z.requires_grad = True
        
        if not optimize_y:
            optimizer = torch.optim.Adam([w_z, w_y, opt_z], lr=5e-3, weight_decay=0.0)
        else:
            opt_y.requires_grad = True
            optimizer = torch.optim.Adam([w_z, w_y,opt_y,opt_z], lr=5e-3, weight_decay=0.0)

        for i in tqdm(range(opt_iters)):
            #print(w_z.shape, w_y.shape)
            optimizer.zero_grad()
            
            if not latent_noise:
                s_z = torch.softmax(w_z, dim=0)
                s_y = torch.softmax(w_y, dim=0)
                #print(s_z)
            
                cur_z = s_z * opt_z
                cur_y = s_y * opt_y
                cur_z = cur_z.sum(dim=0, keepdim=True)
                cur_y = cur_y.sum(dim=0, keepdim=True)

                image_tensors = self.G(cur_z, cur_y)
            else:
                s_z = torch.softmax(w_z, dim=0)
                s_y = torch.softmax(w_y, dim=0)
            
                cur_z = s_z * opt_z
                cur_y = s_y * opt_y
                cur_z = cur_z.sum(dim=0, keepdim=True)
                cur_y = cur_y.sum(dim=0, keepdim=True)
                cur_z_aug = cur_z + torch.randn(cur_z.shape).to(cur_z.device) * LATENT_NOISE
                cur_y_aug = cur_y + torch.randn(cur_y.shape).to(cur_y.device) * LATENT_NOISE
                
                image_tensors = self.G(cur_z_aug, cur_y_aug)
            
            loss = 0.0
            for j in range(image_tensors.shape[0]):
                if augment:
                    loss = loss + AugmentLoss(image_tensors[j:(j+1)], self.clip_model, text_tok, replicate=50, interp_mode=self.interp_mode, desired_words=desired_tokens, expl_lambda=expl_lambda)
                else:
                    loss = loss + NaiveSemanticLoss(image_tensors[j:(j+1)], self.clip_model, text_tok) 

            loss.backward()
            optimizer.step()

            opt_z.data = torch.clamp(opt_z.data.detach().clone(), min=-Z_THRES, max=Z_THRES)

        z_init_ans = cur_z.detach().clone()
        y_init_ans = cur_y.detach().clone()

        #save_image(self.G(z_init_ans, y_init_ans), 'samples/opt_%s.png'%text, 1)
        return self.G(z_init_ans, y_init_ans), z_init_ans, y_init_ans    

    def measureAugCLIP(self, z, y, text, augment=False, num_samples=20):
        text_tok = clip.tokenize([text]).to(self.device)
        avg_loss = 0.0
        for itr in range(num_samples):
            image_tensors = self.G(z, y)

            for j in range(image_tensors.shape[0]):
                if augment:
                    loss = AugmentLoss(image_tensors[j:(j+1)], self.clip_model, text_tok, replicate=50, interp_mode=self.interp_mode, policy=TEST_POLICY)
                else:
                    loss = NaiveSemanticLoss(image_tensors[j:(j+1)], self.clip_model, text_tok) 
            avg_loss += loss.item()

        avg_loss /= num_samples
        return avg_loss * (-1.)

class FuseDreamBaseGeneratorStyleGAN():
    def __init__(self, G, G_config, G_batch_size=10, clip_mode="ViT-B/32", interp_mode='bilinear'):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.G = G
        self.mean_latent = self.G.mean_latent(4096)
        self.clip_model, _ = clip.load(clip_mode, device=device, jit=False) 
        
        z_ = utils.Distribution(torch.randn(G_batch_size, 512, requires_grad=False))
        z_.init_distribution('normal', mean=0, var=1.)
        self.z_dist = z_.to(device)

        self.G.eval()

        for p in self.G.parameters():
            p.requires_grad = False
        # for p in self.clip_model.parameters():
        #     p.requires_grad = False

        self.interp_mode = interp_mode 
  
    def generate_basis(self, text, init_iters=500, num_basis=5):
        text_tok = clip.tokenize([text]).to(self.device)
        clip_c = self.clip_model.logit_scale.exp() 

        z_init_cllt = []
        z_init = None
        score_init = None
        with torch.no_grad():
            for i in tqdm(range(init_iters)):
                self.z_dist.sample_()

                # self.z_.data = torch.clamp(self.z_.data.detach().clone(), min=-Z_THRES, max=Z_THRES)

                _, self.z_, _ = self.G([self.z_dist], return_latents=True, truncation=0.7, truncation_latent=self.mean_latent)

                image_tensors, _ = self.G([self.z_], input_is_latent=True, randomize_noise=False)
                image_tensors = (image_tensors+1.) / 2.
                image_tensors = F.interpolate(image_tensors, size=224, mode=self.interp_mode)
                image_tensors.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
                
                logits_per_image, logits_per_text = self.clip_model(image_tensors, text_tok)
                logits_per_image = logits_per_image/clip_c
                if z_init is None:
                    z_init = self.z_.data.detach().clone()
                    score_init = logits_per_image.squeeze()
                else:
                    z_init = torch.cat([z_init, self.z_.data.detach().clone()], dim=0)
                    score_init = torch.cat([score_init, logits_per_image.squeeze()])

                sorted, indices = torch.sort(score_init, descending=True)
                z_init = z_init[indices]
                score_init = score_init[indices]
                z_init = z_init[:num_basis]
                score_init = score_init[:num_basis]
        
        #save_image(self.G(z_init, self.G.shared(y_init)), 'samples/init_%s.png'%text, 1)

        z_init_cllt.append(z_init.detach().clone())

        return z_init_cllt, [torch.tensor([0])]


    def optimize_clip_score(self, z_init_cllt, y_init_cllt, text, latent_noise=False, augment=True, opt_iters=500, optimize_y=False, desired_words='', expl_lambda=0.):

        if expl_lambda > 0:
            assert desired_words, "It is not possible to use explainability loss without sspecifying desired words"
            desired_words = get_desired_tokens_from_words(desired_words, text)
        else:
            desired_words = []

        text_tok = clip.tokenize([text]).to(self.device)
        clip_c = self.clip_model.logit_scale.exp()

        z_init_ans = torch.stack(z_init_cllt)
        z_init_ans = z_init_ans.view(-1, z_init_ans.shape[-1])

        w_z = torch.randn((z_init_ans.shape[0], z_init_ans.shape[1])).to(self.device)
        w_z.requires_grad = True

        opt_z = torch.zeros(z_init_ans.shape).to(self.device)
        opt_z.data = z_init_ans.data.detach().clone()
        opt_z.requires_grad = True
        
        optimizer = torch.optim.Adam([w_z, opt_z], lr=5e-3, weight_decay=0.0)

        for i in tqdm(range(opt_iters)):
            #print(w_z.shape, w_y.shape)
            optimizer.zero_grad()
            
            if not latent_noise:
                s_z = torch.softmax(w_z, dim=0)
                #print(s_z)
            
                cur_z = s_z * opt_z
                cur_z = cur_z.sum(dim=0, keepdim=True)

                image_tensors, _ = self.G([cur_z], input_is_latent=True, randomize_noise=False)
            else:
                s_z = torch.softmax(w_z, dim=0)
            
                cur_z = s_z * opt_z
                cur_z = cur_z.sum(dim=0, keepdim=True)
                cur_z_aug = cur_z + torch.randn(cur_z.shape).to(cur_z.device) * LATENT_NOISE
                
                image_tensors, _ = self.G([cur_z_aug], input_is_latent=True, randomize_noise=False)
            
            loss = 0.0
            for j in range(image_tensors.shape[0]):
                if augment:
                    loss = loss + AugmentLoss(image_tensors[j:(j+1)], self.clip_model, text_tok, replicate=50, interp_mode=self.interp_mode, desired_words=desired_words, expl_lambda=expl_lambda)
                else:
                    loss = loss + NaiveSemanticLoss(image_tensors[j:(j+1)], self.clip_model, text_tok) 

            loss.backward()
            optimizer.step()

            opt_z.data = torch.clamp(opt_z.data.detach().clone(), min=-Z_THRES, max=Z_THRES)

        z_init_ans = cur_z.detach().clone()

        #save_image(self.G(z_init_ans, y_init_ans), 'samples/opt_%s.png'%text, 1)
        img, _ = self.G([z_init_ans], input_is_latent=True, randomize_noise=False)   
        return img, z_init_ans, torch.tensor([0]) 

    def measureAugCLIP(self, z, y, text, augment=False, num_samples=20):
        text_tok = clip.tokenize([text]).to(self.device)
        avg_loss = 0.0
        for itr in range(num_samples):
            image_tensors, _ = self.G([z], input_is_latent=True, randomize_noise=False)

            for j in range(image_tensors.shape[0]):
                if augment:
                    loss = AugmentLoss(image_tensors[j:(j+1)], self.clip_model, text_tok, replicate=50, interp_mode=self.interp_mode, policy=TEST_POLICY)
                else:
                    loss = NaiveSemanticLoss(image_tensors[j:(j+1)], self.clip_model, text_tok) 
            avg_loss += loss.item()

        avg_loss /= num_samples
        return avg_loss * (-1.)

class FuseDreamBaseGenerator_VQGAN():
    def __init__(self, G, G_config, G_batch_size=10, clip_mode="ViT-B/32", interp_mode='bilinear', size=(256, 256)):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.G = G
        self.clip_model, _ = clip.load(clip_mode, device=device, jit=False) 
        
        (self.z_, self.y_) = utils.prepare_z_y(
            G_batch_size,
            self.G.dim_z,
            G_config["n_classes"],
            device=G_config["device"],
            fp16=G_config["G_fp16"],
            z_var=G_config["z_var"],
        )

        f = 2**(self.G.decoder.num_resolutions - 1)
        toksX, toksY = size[0] // f, size[1] // f
        sideX, sideY = toksX * f, toksY * f

        if gumbel:
            self.e_dim = 256
            self.n_toks = self.G.quantize.n_embed
            self.z_min = self.G.quantize.embed.weight.min(dim=0).values[None, :, None, None]
            self.z_max = self.G.quantize.embed.weight.max(dim=0).values[None, :, None, None]
        else:
            self.e_dim = self.G.quantize.e_dim
            self.n_toks = self.G.quantize.n_e
            self.z_min = self.G.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
            self.z_max = self.G.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

            z_ = utils.Distribution(torch.randint(n_toks, [toksY * toksX], requires_grad=False))
            z_.init_distribution('categorical', num_categories=n_toks)
            self.z_dist = z_.to(device)       

        self.G.eval()

        for p in self.G.parameters():
            p.requires_grad = False
        # for p in self.clip_model.parameters():
        #     p.requires_grad = False

        self.interp_mode = interp_mode 

    def sample_quantized(self, z, toksY, toksX, e_dim):
        if gumbel:
            z = z @ self.G.quantize.embed.weight
        else:
            z = z @ self.G.quantize.embedding.weight

        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2) 

        return z
  
    def generate_basis(self, text, init_iters=500, num_basis=5, desired_words='', expl_lambda=0.):
        text_tok = clip.tokenize([text]).to(self.device)
        clip_c = self.clip_model.logit_scale.exp() 

        if expl_lambda > 0:
            assert desired_words, "It is not possible to use explainability loss without sspecifying desired words"
            desired_words = get_desired_tokens_from_words(desired_words, text)
        else:
            desired_words = []

        z_init_cllt = []
        z_init = None
        score_init = None
        with torch.no_grad():
            for i in tqdm(range(init_iters)):
                self.z_.sample_()
                self.y_.sample_()
                self.z_.data = torch.clamp(self.z_.data.detach().clone(), min=-Z_THRES, max=Z_THRES)
                z_ = self.sample_quantized(self.z_, self.toksY, self.toksX, self.e_dim)

                image_tensors = self.G.decode(z_)
                image_tensors = (image_tensors+1.) / 2.
                image_tensors = F.interpolate(image_tensors, size=224, mode=self.interp_mode)
                image_tensors.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
                
                logits_per_image, logits_per_text = self.clip_model(image_tensors, text_tok)
                logits_per_image = logits_per_image/clip_c

                if expl_lambda > 0:
                    expl_scores = explainability_loss(image=image_tensors, 
                                                      model=self.clip_model, 
                                                      text_tokenized=text_tok, 
                                                      batch_size=len(image_tensors), 
                                                      desired_words=desired_words).unsqueeze(-1)
                    logits_per_image = logits_per_image - expl_scores * expl_lambda



                if z_init is None:
                    z_init = self.z_.data.detach().clone()
                    score_init = logits_per_image.squeeze()
                else:
                    z_init = torch.cat([z_init, self.z_.data.detach().clone()], dim=0)
                    score_init = torch.cat([score_init, logits_per_image.squeeze()])

                sorted, indices = torch.sort(score_init, descending=True)
                z_init = z_init[indices]
                score_init = score_init[indices]
                z_init = z_init[:num_basis]
                score_init = score_init[:num_basis]
        
        #save_image(self.G(z_init, self.G.shared(y_init)), 'samples/init_%s.png'%text, 1)
        

        z_init_cllt.append(z_init.detach().clone())

        return z_init_cllt, torch.tensor([0])


    def optimize_clip_score(self, z_init_cllt, y_init_cllt, text, latent_noise=False, augment=True, opt_iters=500, optimize_y=False, desired_words='', expl_lambda=0.):

        if expl_lambda > 0:
            assert desired_words, "It is not possible to use explainability loss without sspecifying desired words"
            desired_words = get_desired_tokens_from_words(desired_words, text)
        else:
            desired_words = []

        text_tok = clip.tokenize([text]).to(self.device)
        clip_c = self.clip_model.logit_scale.exp()

        z_init_ans = torch.stack(z_init_cllt)
        z_init_ans = z_init_ans.view(-1, z_init_ans.shape[-1])

        w_z = torch.randn((z_init_ans.shape[0], z_init_ans.shape[1])).to(self.device)
        w_z.requires_grad = True

        opt_z = torch.zeros(z_init_ans.shape).to(self.device)
        opt_z.data = z_init_ans.data.detach().clone()
        opt_z.requires_grad = True
        
        optimizer = torch.optim.Adam([w_z, opt_z], lr=5e-3, weight_decay=0.0)

        for i in tqdm(range(opt_iters)):
            #print(w_z.shape, w_y.shape)
            optimizer.zero_grad()
            
            if not latent_noise:
                s_z = torch.softmax(w_z, dim=0)
                #print(s_z)
            
                cur_z = s_z * opt_z
                cur_z = cur_z.sum(dim=0, keepdim=True)
                cur_z = self.sample_quantized(cur_z, self.toksY, self.toksX, self.e_dim)

                image_tensors = self.G.decode(cur_z)
            else:
                s_z = torch.softmax(w_z, dim=0)
            
                cur_z = s_z * opt_z
                cur_z = cur_z.sum(dim=0, keepdim=True)
                cur_z_aug = cur_z + torch.randn(cur_z.shape).to(cur_z.device) * LATENT_NOISE
                
                cur_z = self.sample_quantized(cur_z, self.toksY, self.toksX, self.e_dim)
                image_tensors = self.G.decode(cur_z_aug)
            
            loss = 0.0
            for j in range(image_tensors.shape[0]):
                if augment:
                    loss = loss + AugmentLoss(image_tensors[j:(j+1)], self.clip_model, text_tok, replicate=50, interp_mode=self.interp_mode, desired_words=desired_words, expl_lambda=expl_lambda)
                else:
                    loss = loss + NaiveSemanticLoss(image_tensors[j:(j+1)], self.clip_model, text_tok) 

            loss.backward()
            optimizer.step()

            opt_z.data = torch.clamp(opt_z.data.detach().clone(), min=-Z_THRES, max=Z_THRES)

        z_init_ans = cur_z.detach().clone()

        #save_image(self.G(z_init_ans, y_init_ans), 'samples/opt_%s.png'%text, 1)
        return self.G.decode(z_init_ans), z_init_ans, torch.tensor([0])    

    def measureAugCLIP(self, z, y, text, augment=False, num_samples=20):
        text_tok = clip.tokenize([text]).to(self.device)
        avg_loss = 0.0
        for itr in range(num_samples):
            image_tensors = self.G.decode(z)

            for j in range(image_tensors.shape[0]):
                if augment:
                    loss = AugmentLoss(image_tensors[j:(j+1)], self.clip_model, text_tok, replicate=50, interp_mode=self.interp_mode, policy=TEST_POLICY)
                else:
                    loss = NaiveSemanticLoss(image_tensors[j:(j+1)], self.clip_model, text_tok) 
            avg_loss += loss.item()

        avg_loss /= num_samples
        return avg_loss * (-1.)
