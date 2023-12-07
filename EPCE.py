# The above code is not provided. It appears to be a reference to a research paper titled "High
# Dynamic Range Image Reconstruction via Deep Explicit Polynomial Curve Estimation" by Jiaqi Tang. The
# code mentioned in the question is not included in the provided information.
## High Dynamic Range Image Reconstruction via Deep Explicit Polynomial Curve Estimation
## By Jiaqi Tang
## https://ebooks.iospress.nl/doi/10.3233/FAIA230533

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import torch.nn as nn
from torchvision import models
from einops import rearrange

##########################################################################
##---------- Explicit Polynomial Curve Estimation (EPCE) -----------------
class Curve_Estimation(nn.Module):

    def __init__(self):
        """
        The `forward` function takes an input `x` and a parameter `a`, passes `x` through a neural
        network, and then uses the parameters `r0` to `r8` to perform a curve estimation on `x` to
        enhance the image.
        """

        # The above code is initializing an instance of the `Curve_Estimation` class and calling its
        # superclass's `__init__` method. It then creates an instance of the `LPE` class and assigns
        # it to the `network` attribute of the `Curve_Estimation` instance.
        super(Curve_Estimation, self).__init__()

        self.network = LPE()

    def forward(self, x, a):
        """
        Perform forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.
            a (torch.Tensor): Additional tensor.

        Returns:
            torch.Tensor: Enhanced image tensor.
        """

        x = self.network(x)

        x_r = a
     # Assuming x_r is a tensor and you're splitting it into 9 tensors along dim=1
        split_tensors = torch.split(x_r, 3, dim=1)


# Assigning each split tensor to individual variables
        r0, r1, r2, r3, r4, r5, r6, r7, r8 = split_tensors[0], split_tensors[1], split_tensors[2], split_tensors[3], split_tensors[4], split_tensors[5], split_tensors[6], split_tensors[7], split_tensors[8]

# Now you can use these variables individually

        
        enhance_image = r1 * torch.pow(x, 1) + r2 * torch.pow(x, 2) + r3 * torch.pow(x, 3) + r4 * torch.pow(x, 4) + \
                        r5 * torch.pow(x, 5) + r6 * torch.pow(x, 6) + r7 * torch.pow(x, 7) + r8 * torch.pow(x, 8) + \
                        r0 * 1
            
        return enhance_image

def to_3d(x):
    """
    The function `to_3d` rearranges the dimensions of a tensor from 2D to 3D.
    
    :param x: The input tensor with shape (b, c, h, w), where:
    :return: the input tensor `x` rearranged in a 3D format.
    """
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

# The BiasFree_LayerNorm class implements a bias-free layer normalization operation in PyTorch.
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        """
        The function initializes a class called `BiasFree_LayerNorm` and sets the weight and normalized
        shape attributes.
        
        :param normalized_shape: The `normalized_shape` parameter is the shape of the input tensor that
        will be normalized. It can be either an integer or a tuple of integers representing the
        dimensions of the input tensor
        """
       # The above code is defining a class called `BiasFree_LayerNorm` in Python.
        
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## ---------- U-shape Local Forward Block (ULFB) -------------------------
class ULFB(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(ULFB, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        
        self.relu = nn.ReLU(inplace=True)

        number_f=dim
        self.e_conv1 = nn.Conv2d(dim, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, dim, 3, 1, 1, bias=True)
        
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return x

##########################################################################
## --Multi-DConv Head Transposed Self-Attention (MDTA) (from: Restormer)--
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = ULFB(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## ----------------Overlapped image patch embedding with 3x3 Conv --------
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## ------------------ Resizing modules -----------------------------------
class Downsample(nn.Module):
    """
    The `Downsample` class is a neural network module that performs downsampling on an input
    image using convolutional layers.
    
    :param n_feat: The parameter `n_feat` represents the number of input features or channels in
    the convolutional layers of the `Downsample` module
    """
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.Conv2d(n_feat, n_feat*4, kernel_size=3, stride=2, padding=1, bias=False))
                                # The above code is written in Python and it appears to be using a
                                # neural network module called `nn`. It is likely part of a larger
                                # codebase or script. The specific function being used is
                                # `PixelUnShuffle`, which suggests that it is performing some kind of
                                # operation on pixels or images. The number `2` passed as an argument
                                # to the function could indicate that the function is unshuffling
                                # pixels by a factor of 2. However, without more context or
                                # information about the `nn` module and its specific implementation,
                                # it is difficult to determine the exact purpose or outcome of this
                                # code
                                #   nn.PixelUnShuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Pyramid-Path Vision Transformer (PPViT) ---------------------
class PPVisionTransformer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=27,
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):
        super(PPVisionTransformer, self).__init__()

        self.patch_embed_1 = OverlapPatchEmbed(inp_channels, dim*2)

        self.patch_embed_2 = OverlapPatchEmbed(12, dim)
        self.patch_embed_3 = OverlapPatchEmbed(48, dim)

        self.first_1 = nn.Sequential(TransformerBlock(dim=dim*2, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim*2, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim*2, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim*2, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        
        self.first_2 = nn.Sequential(TransformerBlock(dim=dim*2, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim*2, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim*2, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim*2, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        
        self.first_3 = nn.Sequential(TransformerBlock(dim=dim*2, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim*2, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim*2, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim*2, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))

        self.down1 = Downsample(3)
        self.second_1 = nn.Sequential(TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                      TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                      TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                      TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        
        self.second_2 = nn.Sequential(TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                      TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                      TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                      TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        
        self.second_3 = nn.Sequential(TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                      TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                      TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                      TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        self.up1 = Upsample(dim)

        self.down2 = Downsample(12)
        self.third_1 = nn.Sequential(TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        
        self.third_2 = nn.Sequential(TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        
        self.third_3 = nn.Sequential(TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        self.up2 = Upsample(dim)
        self.up3 = Upsample(dim)

        self.relu = nn.PReLU()

        self.refine = nn.Sequential(TransformerBlock(dim=dim*4, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                    TransformerBlock(dim=dim*4, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                    TransformerBlock(dim=dim*4, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                    TransformerBlock(dim=dim*4, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
                
        self.output = nn.Conv2d(dim*4, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.curve_estimation = Curve_Estimation()

    def forward(self, x):
        
        inp_img=x

        inp_img_0=inp_img

        inp_img_1=self.down1(inp_img)

        inp_img_2= self.down2(inp_img_1)

        inp_img_0=self.patch_embed_1(inp_img_0)
        inp_img_1=self.patch_embed_2(inp_img_1)
        inp_img_2=self.patch_embed_3(inp_img_2)
        
        inp_img_0=self.first_1(inp_img_0)
        inp_img_0=self.first_2(inp_img_0)
        inp_img_0=self.first_3(inp_img_0)
        inp_img_0=self.relu(inp_img_0)

        inp_img_1=self.second_1(inp_img_1)
        inp_img_1=self.second_2(inp_img_1)
        inp_img_1=self.second_3(inp_img_1)
        inp_img_1=self.relu(inp_img_1)

        inp_img_2=self.third_1(inp_img_2)
        inp_img_2=self.third_2(inp_img_2)
        inp_img_2=self.third_3(inp_img_2)
        inp_img_2=self.relu(inp_img_2)

        inp_img_0=inp_img_0;
        inp_img_1=self.up1(inp_img_1)
        inp_img_2=self.up2(inp_img_2)
        inp_img_2=self.up3(inp_img_2)
        
        image=torch.cat([inp_img_0,inp_img_1,inp_img_2],1)

        image=self.refine(image)

        out_right = self.output(image)

        output = self.curve_estimation(x,out_right)

        return output

##########################################################################
##---------- Learnable Pixel-Wise Elements (LPE) -------------------------
class LPE(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=27,
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):
        super(LPE, self).__init__()

        self.patch_embed_1 = OverlapPatchEmbed(inp_channels, dim)

        self.first_1 = nn.Sequential(TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        
        self.first_2 = nn.Sequential(TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        
        self.first_3 = nn.Sequential(TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                     TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))

        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(dim//2, dim//4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(dim//4,3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU()
        )

    def forward(self, x):
        
        inp_img=x

        inp_img_0=inp_img

        inp_img_0=self.patch_embed_1(inp_img_0)

        inp_img_0=self.first_1(inp_img_0)
        inp_img_0=self.first_2(inp_img_0)
        inp_img_0=self.first_3(inp_img_0)

        inp_img_0=self.refine(inp_img_0)
        image=inp_img_0+x

        return image

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
