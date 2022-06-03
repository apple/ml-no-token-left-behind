
import torch
import external.TransformerMMExplainability.CLIP.clip as clip
import numpy as np

class CLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity

class EGCLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(EGCLIPLoss, self).__init__()
        from external.TransformerMMExplainability.CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
        self._tokenizer = _Tokenizer()
        self.description = opts.description

        self.words = opts.description.split(" ")

        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda", jit=False)
        self.desired_words = np.array(opts.desired_words.split(' ')).astype(float)
        
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)
        self.expl_lambda = opts.expl_lambda

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        self.desired_tokens = torch.zeros((len(self._tokenizer.encode(self.description))))

        token_id = 0
        for word_idx, word in enumerate(self.words):
            num_of_tokens = len(self._tokenizer.encode(word))

            for t in range(num_of_tokens):
                self.desired_tokens[token_id] = self.desired_words[word_idx]
                token_id = token_id + 1

        self.desired_tokens = (self.desired_tokens - self.desired_tokens.min()) / (
                self.desired_tokens.max() - self.desired_tokens.min())


        CLS_idx = text.argmax(dim=-1)

        with torch.enable_grad():
            logits_per_image, logits_per_text = self.model(image, text)
            one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
            index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
            one_hot[0, index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * logits_per_image)

            text_attn_blocks = list(dict(self.model.transformer.resblocks.named_children()).values())
            num_tokens = text_attn_blocks[0].attn.attn_output_weights.shape[-1]
            R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn.attn_output_weights.dtype).to(logits_per_image.device)
            for blk in text_attn_blocks:
                grad = torch.autograd.grad(one_hot, [blk.attn.attn_output_weights], retain_graph=True, create_graph=True)[0]
                cam = blk.attn.attn_output_weights
                cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                R_text = R_text + torch.matmul(cam, R_text)
            text_relevance = R_text[CLS_idx, 1:CLS_idx]
            text_relevance = text_relevance / text_relevance.sum()
            target_word_expl_score = (text_relevance * self.desired_tokens.to(logits_per_image.device))
            target_word_expl_score = target_word_expl_score / (self.desired_tokens != 0).int().sum()

        similarity = 1 - logits_per_image / 100
        return similarity - target_word_expl_score.sum() * self.expl_lambda