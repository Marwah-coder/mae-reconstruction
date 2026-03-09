import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from einops import rearrange

device = torch.device('cpu')

def sincos_pos_embed(dim, grid_size):
    assert dim % 4 == 0
    gx, gy = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    gx = gx.reshape(-1); gy = gy.reshape(-1)
    half = dim // 4
    freq = 1.0 / (10000 ** (np.arange(half) / half))
    emb = np.concatenate([
        np.sin(np.outer(gx,freq)), np.cos(np.outer(gx,freq)),
        np.sin(np.outer(gy,freq)), np.cos(np.outer(gy,freq))], axis=1)
    return torch.tensor(emb, dtype=torch.float32)

class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads=heads; self.hd=dim//heads; self.scale=self.hd**-0.5
        self.qkv=nn.Linear(dim,dim*3); self.proj=nn.Linear(dim,dim)
    def forward(self, x):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.heads,self.hd).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        attn=(q@k.transpose(-2,-1)*self.scale).softmax(dim=-1)
        return self.proj((attn@v).transpose(1,2).reshape(B,N,C))

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        h=int(dim*4)
        self.n1=nn.LayerNorm(dim); self.att=Attention(dim,heads)
        self.n2=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,h),nn.GELU(),nn.Linear(h,dim))
    def forward(self,x):
        x=x+self.att(self.n1(x)); x=x+self.ffn(self.n2(x)); return x

class MAEEncoder(nn.Module):
    def __init__(self, img=224, p=16, c=3, dim=768, depth=12, heads=12):
        super().__init__()
        self.dim=dim; self.p=p
        self.embed=nn.Linear(p*p*c, dim)
        self.cls=nn.Parameter(torch.zeros(1,1,dim))
        self.register_buffer('pos', sincos_pos_embed(dim, img//p).unsqueeze(0))
        self.blocks=nn.ModuleList([Block(dim,heads) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim)
    def forward(self, vis, vis_ids):
        B=vis.size(0)
        x=self.embed(vis)
        pos_v=torch.gather(self.pos.expand(B,-1,-1),1,vis_ids.unsqueeze(-1).expand(-1,-1,self.dim))
        x=x+pos_v
        x=torch.cat([self.cls.expand(B,-1,-1), x], dim=1)
        for b in self.blocks: x=b(x)
        return self.norm(x)[:,1:]

class MAEDecoder(nn.Module):
    def __init__(self, N=196, enc_dim=768, dim=384, depth=12, heads=6, p=16, c=3):
        super().__init__()
        self.dim=dim; self.N=N
        self.proj=nn.Linear(enc_dim,dim)
        self.mask_tok=nn.Parameter(torch.zeros(1,1,dim))
        self.register_buffer('pos', sincos_pos_embed(dim, int(N**0.5)).unsqueeze(0))
        self.blocks=nn.ModuleList([Block(dim,heads) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim)
        self.head=nn.Linear(dim, p*p*c)
    def forward(self, lat, vis_ids, msk_ids):
        B,D=lat.size(0),self.dim
        enc=self.proj(lat).float()
        mtok=self.mask_tok.float().expand(B,msk_ids.size(1),-1)
        seq=torch.zeros(B,self.N,D,device=lat.device,dtype=torch.float32)
        seq.scatter_(1,vis_ids.unsqueeze(-1).expand(-1,-1,D),enc)
        seq.scatter_(1,msk_ids.unsqueeze(-1).expand(-1,-1,D),mtok)
        seq=seq+self.pos.float()
        for b in self.blocks: seq=b(seq)
        return self.head(self.norm(seq))

class MAE(nn.Module):
    def __init__(self, img=224, p=16, c=3, mask=0.75):
        super().__init__()
        self.p=p; self.c=c; self.img=img; self.mask=mask
        self.N=(img//p)**2
        self.encoder=MAEEncoder(img,p,c)
        self.decoder=MAEDecoder(self.N)
    def patchify(self,x):
        return rearrange(x,'b c (nh p1)(nw p2)->b (nh nw)(p1 p2 c)',p1=self.p,p2=self.p)
    def unpatchify(self,x):
        n=self.img//self.p
        return rearrange(x,'b (nh nw)(p1 p2 c)->b c (nh p1)(nw p2)',nh=n,nw=n,p1=self.p,p2=self.p,c=self.c)
    def mask_patches(self,patches,mask_ratio):
        B,N,D=patches.shape
        nv=int(N*(1-mask_ratio))
        ids=torch.argsort(torch.rand(B,N,device=patches.device),dim=1)
        vi,_=torch.sort(ids[:,:nv],dim=1)
        mi,_=torch.sort(ids[:,nv:],dim=1)
        vp=torch.gather(patches,1,vi.unsqueeze(-1).expand(-1,-1,D))
        bm=torch.ones(B,N,dtype=torch.bool,device=patches.device)
        bm.scatter_(1,vi,False)
        return vp,vi,mi,bm
    def forward(self,imgs,mask_ratio=0.75):
        pat=self.patchify(imgs)
        vp,vi,mi,bm=self.mask_patches(pat,mask_ratio)
        lat=self.encoder(vp,vi)
        pred=self.decoder(lat,vi,mi)
        return pred,bm

model = MAE().to(device)
model.eval()

MEAN=torch.tensor([0.485,0.456,0.406])
STD =torch.tensor([0.229,0.224,0.225])

def denorm(t):
    t=t.cpu().float()
    for c in range(3): t[c]=t[c]*STD[c]+MEAN[c]
    return t.clamp(0,1).permute(1,2,0).numpy()

def infer(pil_img, pct):
    if pil_img is None: return None, None, None
    tfm=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    t=tfm(pil_img.convert('RGB')).unsqueeze(0).to(device)
    mask_ratio = pct/100
    with torch.no_grad():
        pred,mask=model(t, mask_ratio)
    pat=model.patchify(t); mp=pat.clone(); mp[mask]=0.0
    def to_pil(x):
        arr=denorm(x[0])
        return Image.fromarray((arr*255).clip(0,255).astype(np.uint8))
    return to_pil(model.unpatchify(mp)), to_pil(model.unpatchify(pred)), to_pil(t)

with gr.Blocks(title='MAE') as demo:
    gr.Markdown('# Masked Autoencoder — Image Reconstruction')
    gr.Markdown('**Note: Model weights not loaded — showing architecture demo only**')
    with gr.Row():
        inp=gr.Image(type='pil',label='Upload Image')
        ratio=gr.Slider(10,90,value=75,step=5,label='Masking %')
    btn=gr.Button('Reconstruct',variant='primary')
    with gr.Row():
        o1=gr.Image(label='Masked')
        o2=gr.Image(label='Reconstruction')
        o3=gr.Image(label='Original')
    btn.click(infer,[inp,ratio],[o1,o2,o3])

demo.launch()