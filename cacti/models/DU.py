import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import init, ReLU
from cacti.models.builder import MODELS


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class SA(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=16, shift=False):
        super().__init__()
        self.heads = dim // dim_head
        self.window_size = window_size
        self.shift = shift

        # num_token = window_size[0] * window_size[1]
        # self.cal_atten = Attention(dim_head, num_token)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_qk = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def cal_attention(self, x):
        q, k = self.to_qk(x).chunk(2, dim=-1)
        v = self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, 'b B (h b0) (w b1) (d c) -> (b h w B) d (b0 b1) c',
                                d=self.heads, b0=self.window_size[0], b1=self.window_size[1]),
            (q, k, v)
        )
        # attn = self.cal_atten(q, k)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

    def forward(self, x):
        b, c, B, h, w = x.shape
        w_size = self.window_size
        if self.shift:
            x = x.roll(shifts=4, dims=3).roll(shifts=4, dims=4)
        x_inp = x.permute(0, 2, 3, 4, 1)
        out = self.cal_attention(x_inp)
        out = rearrange(out, '(b h w B) (b0 b1) c -> b c B (h b0) (w b1)', h=h // w_size[0], w=w // w_size[1],
                        b0=w_size[0], B=B)
        if self.shift:
            out = out.roll(shifts=-4, dims=4).roll(shifts=-4, dims=3)
        return out



class SFFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv3d(dim * mult, dim * mult, (1, 3, 3), 1, (0, 1, 1), bias=False, groups=dim * mult),
            GELU(),
            nn.Conv3d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x)
        return out


class TFFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv3d(dim * mult, dim * mult, (3, 1, 1), 1, (1, 0, 0), bias=False, groups=dim * mult),
            GELU(),
            nn.Conv3d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x)
        return out


class SSAB(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=16, mult=4, shift=False):
        super().__init__()
        self.pos_emb = nn.Conv3d(dim, dim, (1, 5, 5), 1, (0, 2, 2), bias=False, groups=dim)
        self.fa = PreNorm(dim, SA(dim=dim, window_size=window_size, dim_head=dim_head, shift=shift),
                          norm_type='ln')
        self.ffn = PreNorm(dim, SFFN(dim=dim, mult=mult), norm_type='ln')

    def forward(self, x):
        x = x + self.pos_emb(x)
        x = self.fa(x) + x
        x = self.ffn(x) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_type='ln'):
        super().__init__()
        self.fn = fn
        self.norm_type = norm_type
        if norm_type == 'ln':
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.GroupNorm(1, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, *args, **kwargs):
        if self.norm_type == 'ln':
            x = self.norm(x.permute(0, 2, 3, 4, 1).contiguous()).permute(0, 4, 1, 2, 3).contiguous()
        else:
            x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class TSAB(nn.Module):
    def __init__(self, dim, dim_head):
        super().__init__()
        self.pos_emb = nn.Conv3d(dim, dim, (5, 1, 1), 1, (2, 0, 0), bias=False, groups=dim)
        self.tsab = PreNorm(dim, TA(dim, dim_head), norm_type='ln')
        self.ffn = PreNorm(dim, TFFN(dim=dim), norm_type='ln')

    def forward(self, x):
        x = x + self.pos_emb(x)
        x = self.tsab(x) + x
        x = self.ffn(x) + x
        return x


class STSAB(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=16, mult=4, shift=False):
        super().__init__()
        self.SSAB = SSAB(dim=dim, window_size=window_size, dim_head=dim_head, mult=mult, shift=shift)
        self.TSAB = TSAB(dim, dim_head=dim_head)

    def forward(self, x):
        x = self.SSAB(x)
        x = self.TSAB(x)
        return x


class STUNet(nn.Module):
    def __init__(self, dim=32):
        super(STUNet, self).__init__()

        self.conv_in = nn.Conv3d(dim, dim, 3, 1, 1, bias=False)
        self.down1 = STSAB(dim=dim, dim_head=dim, mult=4)
        self.downsample1 = nn.Conv3d(dim, dim * 2, (1, 4, 4), (1, 2, 2), (0, 1, 1), bias=False)
        self.down2 = STSAB(dim=dim * 2, dim_head=dim, mult=4)
        self.downsample2 = nn.Conv3d(dim * 2, dim * 4, (1, 4, 4), (1, 2, 2), (0, 1, 1), bias=False)

        self.bottleneck_local = STSAB(dim=dim * 2, dim_head=dim, mult=4)
        self.bottleneck_swin = STSAB(dim=dim * 2, dim_head=dim, mult=4, shift=True)
        self.upsample2 = nn.ConvTranspose3d(dim * 4, dim * 2, (1, 2, 2), (1, 2, 2))
        self.fusion2 = nn.Conv3d(dim * 4, dim * 2, 1, 1, 0, bias=False)
        self.up2 = STSAB(dim=dim * 2, dim_head=dim, mult=4, shift=True)
        self.upsample1 = nn.ConvTranspose3d(dim * 2, dim, (1, 2, 2), (1, 2, 2))
        self.fusion1 = nn.Conv3d(dim * 2, dim, 1, 1, 0, bias=False)
        self.up1 = STSAB(dim=dim, dim_head=dim, mult=4, shift=True)
        self.conv_out = nn.Conv3d(dim, dim, 3, 1, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
            x: [b,c,B,h,w]
            return out:[b,c,B,h,w]
        """
        b, c, B, h_inp, w_inp = x.shape
        hb, wb = 32, 32
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='constant', value=0)

        x_in = x
        x = self.conv_in(x_in)
        x1 = self.down1(x)
        # stage_outs.append(x1)
        x = self.downsample1(x1)
        x2 = self.down2(x)

        x = self.downsample2(x2)

        x_local = self.bottleneck_local(x[:, :c * 2, :, :, :])

        x_swin = self.bottleneck_swin(x[:, c * 2:, :, :, :] + x_local)

        x = torch.cat([x_local, x_swin], dim=1)

        x = self.upsample2(x)
        x = x2 + self.fusion2(torch.cat([x, x2], dim=1))
        x = self.up2(x)

        x = self.upsample1(x)
        x = x1 + self.fusion1(torch.cat([x, x1], dim=1))
        x = self.up1(x)
        stage_out = self.conv_out(x)
        out = stage_out + x_in
        return out[:, :, :, :h_inp, :w_inp], stage_out


class TDRA(nn.Module):
    def __init__(self, dim, dim_head=8):
        super().__init__()

        self.heads = dim // dim_head

        self.scale = dim_head ** -0.5  * (2 ** -0.5)

        self.to_qk = nn.Linear(dim, dim * 2, bias=False)
        self.to_qk_last = nn.Linear(dim, dim * 2, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def cal_attention(self, x, last):

        x = rearrange(x, 'b c B h w -> b B h w c')
        last = rearrange(last, 'b c B h w -> b B h w c')

        q, k = self.to_qk(x).chunk(2, dim=-1)
        q = q * self.scale
        q_last, k_last = self.to_qk_last(last).chunk(2, dim=-1)
        q_last = q_last * self.scale
        v = self.to_v(x)

        q, k, v, q_last, k_last = map(
            lambda t: rearrange(t, 'b B h w (d c) -> b B d h w c',
                                d=self.heads),
            (q, k, v, q_last, k_last)
        )

        attn_base = torch.einsum('b i d h w c, b j d h w c -> b d h w i j', q, k)
        attn_last = torch.einsum('b i d h w c, b j d h w c -> b d h w i j', q_last, k_last)
        attn_delta = attn_last - attn_base
        attn_delta = attn_delta - attn_delta.max(dim=-1, keepdim=True).values
        attn_delta = attn_delta.softmax(dim=-1)
        out = torch.einsum('b d h w i j, b j d h w c -> b i d h w c', attn_delta, v)
        # out = attn_delta @ v
        out = rearrange(out, 'b B d h w c -> b B h w (c d)')

        return out

    def forward(self, x, last):
        b, c, B, h, w = x.shape
        out = self.cal_attention(x, last)
        out = self.to_out(out)
        out = rearrange(out, 'b B h w c -> b c B h w', b=b, B=B, h=h, w=w)
        return out


class SDRFFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.x_proj = nn.Conv3d(dim, dim, 1, 1, bias=False)

        self.feat_proj = nn.Sequential(
            nn.Conv3d(dim, dim * mult, 1, 1, bias=False),  # 对当前输入进行卷积处理
            GELU(),
            nn.Conv3d(dim * mult, dim * mult, (1, 3, 3), 1, (0, 1, 1), bias=False, groups=dim),
            GELU(),
            nn.Conv3d(dim * mult, dim, 1, 1, bias=False),  # 对当前输入进行卷积处理
        )

        self.last_proj =nn.Conv3d(dim, dim, 1, 1, bias=False)

        self.delta_proj = nn.Conv3d(dim, dim, 1, 1, bias=False)


    def forward(self, x, last):
        feat = self.feat_proj(x)
        x_proj = self.x_proj(x)
        last_proj = self.last_proj(last)
        delta = x_proj - last_proj
        delta_upd = torch.sigmoid(self.delta_proj(delta))
        out = feat * delta_upd
        return out


class TDRFFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        # 当前特征分支 (时序卷积)
        self.x_proj = nn.Conv3d(dim, dim, 1, 1, bias=False)  # 对当前输入进行卷积处理
        self.feat_proj = nn.Sequential(
            nn.Conv3d(dim, dim * mult, 1, 1, bias=False),  # 对当前输入进行卷积处理
            GELU(),
            nn.Conv3d(dim * mult, dim * mult, (3, 1, 1), 1, (1, 0, 0), bias=False, groups=dim),  # 局部信息建模
            GELU(),
            nn.Conv3d(dim * mult, dim, 1, 1, bias=False),  # 局部信息建模
        )

        self.last_proj = nn.Conv3d(dim, dim, 1, 1, bias=False)  # 对上一阶段特征进行卷积处理

        self.delta_proj = nn.Conv3d(dim, dim, 1, 1, bias=False)  # 差分计算后的增量

    def forward(self, x, last):
        x_proj = self.x_proj(x)
        feat = self.feat_proj(x)
        last_proj = self.last_proj(last)
        delta = x_proj - last_proj
        delta_upd = torch.sigmoid(self.delta_proj(delta))
        out = feat * delta_upd
        return out


class SDRA(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=8, shift=True):
        super().__init__()
        self.heads = dim // dim_head
        self.window_size = window_size
        self.shift = shift
        self.scale = dim_head ** -0.5 * (2 ** -0.5)

        self.to_qk_last = nn.Linear(dim, dim * 2, bias=False)
        self.to_qk = nn.Linear(dim, dim * 2, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Linear(dim, dim)


    def cal_attention(self, x, last):
        w_size = self.window_size
        b, c, B, h, w = x.shape

        x = rearrange(x, 'b c B h w -> b B h w c')
        last = rearrange(last, 'b c B h w -> b B h w c')

        q_last, k_last = self.to_qk_last(last).chunk(2, dim=-1)
        q_last = q_last
        q, k = self.to_qk(x).chunk(2, dim=-1)
        v = self.to_v(x)
        q = q

        q, k, v, q_last, k_last = map(
            lambda t: rearrange(t, 'b B (h b0) (w b1) (d c) -> (b h w B) d (b0 b1) c',
                                d=self.heads, b0=w_size[0], b1=w_size[1]),
            (q, k, v, q_last, k_last)
        )

        # attn = torch.einsum('b B d i n, b B d j n -> b B d i j', q, k)
        attn = q @ k.transpose(-2, -1) * self.scale
        # attn_last = torch.einsum('b B d i n, b B d j n -> b B d i j', q_last, k_last)
        attn_last = q_last @ k_last.transpose(-2, -1) * self.scale

        attn_delta = attn_last - attn
        attn_delta = attn_delta - attn_delta.max(dim=-1, keepdim=True).values
        attn_delta = attn_delta.softmax(dim=-1)
        # out = torch.einsum('b B d i j, b B d j n -> b B d i n', attn_delta, v)
        out = attn_delta @ v
        out = rearrange(out, 'b d n c -> b n (c d)')

        return out

    def forward(self, x, last):
        b, c, B, h, w = x.shape
        w_size = self.window_size
        if self.shift:
            x = x.roll(shifts=w_size[0] // 2, dims=3).roll(shifts=w_size[1] // 2, dims=4)
            last = last.roll(shifts=w_size[0] // 2, dims=3).roll(shifts=w_size[1] // 2, dims=4)
        out = self.cal_attention(x, last)
        # 投影回通道
        out = self.to_out(out)
        out = rearrange(
            out,
            '(b h w B) (b0 b1) c -> b c B (h b0) (w b1)',
            h=h // w_size[0], w=w // w_size[1],
            b0=w_size[0], B=B
        )
        if self.shift:
            out = out.roll(shifts=-1 * w_size[0] // 2, dims=4).roll(shifts=-1 * w_size[1] // 2, dims=3)
        return out


class SDRB(nn.Module):
    def __init__(self, dim, window_size=(8, 8), mult=4, shift=False):
        super().__init__()
        self.pos_1 = nn.Conv3d(dim, dim, (1, 5, 5), 1, (0, 2, 2), bias=False, groups=dim)
        self.pos_2 = nn.Conv3d(dim, dim, (1, 5, 5), 1, (0, 2, 2), bias=False, groups=dim)

        self.SDRA = PreNorm(
            dim=dim,
            fn=SDRA(dim=dim, window_size=window_size, dim_head=dim // 2, shift=shift),
            norm_type='ln'
        )
        self.ffn = PreNorm(
            dim=dim,
            fn=SDRFFN(dim=dim, mult=mult),
            norm_type='ln'
        )

    def forward(self, x, last):
        x = self.pos_1(x) + x
        last = self.pos_2(last) + last
        x = self.SDRA(x, last) + x
        x = self.ffn(x, last) + x
        return x

class TDRB(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.pos_1 = nn.Conv3d(dim, dim, (5, 1, 1), 1, (2, 0, 0), bias=False, groups=dim)
        self.pos_2 = nn.Conv3d(dim, dim, (5, 1, 1), 1, (2, 0, 0), bias=False, groups=dim)

        self.TDRA = PreNorm(
            dim=dim,
            fn=TDRA(dim=dim, dim_head=dim // 2),
            norm_type='ln'
        )
        self.ffn = PreNorm(
            dim=dim,
            fn=TDRFFN(dim=dim, mult=mult),
            norm_type='ln'
        )

    def forward(self, x, last):
        x = self.pos_1(x) + x
        last = self.pos_2(last) + last
        x = self.TDRA(x, last) + x
        x = self.ffn(x, last) + x
        return x

class DRB(nn.Module):
    def __init__(self, dim, window_size=(8, 8), mult=4, shift=False):
        super().__init__()
        self.SDRB = SDRB(dim, window_size=window_size, mult=mult, shift=shift)
        self.TDRB = TDRB(dim, mult=mult)

    def forward(self, x, last):
        x = self.SDRB(x, last)
        x = self.TDRB(x, last)
        return x


class DRT(nn.Module):
    def __init__(self, dim=32):
        super(DRT, self).__init__()
        self.conv_in = nn.Conv3d(dim, dim, 3, 1, 1, bias=False)
        self.norm = PreNorm(dim=dim, fn=nn.Identity(), norm_type='ln')

        self.DRB_1 = DRB(dim=dim, mult=4)
        self.DRB_2 = DRB(dim=dim, mult=4, shift=True)

        self.conv_out = nn.Conv3d(dim, dim, 3, 1, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            init.normal_(m.weight.data, mean=0.0, std=0.01)

    def forward(self, x, last=None):
        """
            x: [b,c,B,h,w]
            return out:[b,c,B,h,w]
        """
        b, c, B, h_inp, w_inp = x.shape
        hb, wb = 32, 32
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='constant', value=0)
        x_in = x
        x = self.conv_in(x)
        last = self.norm(last)

        x = self.DRB_1(x, last)
        x = self.DRB_2(x, last)

        stage_outs = self.conv_out(x)
        out = stage_outs + x_in

        return out[:, :, :, :h_inp, :w_inp], stage_outs


class Param_Estimator(nn.Module):
    def __init__(self, in_dim=8, out_dim=1, dim=32):
        super(Param_Estimator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_dim, dim, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.avpool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, dim, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, out_dim, 1, padding=0, bias=True),
            nn.Softplus())

    def forward(self, x):
        x = self.conv(x)
        x = self.avpool(x)
        x = self.mlp(x) + 1e-6
        return x


class TA(nn.Module):
    def __init__(self, dim, num_head, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim
        head_dim = num_head
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        _, _, _, h, w = x.shape
        tsab_in = rearrange(x, "b c B h w->(b h w) B c")
        n, B, C = tsab_in.shape
        qkv = self.qkv(tsab_in)
        qkv = qkv.reshape(3, n, self.num_heads, B, C // self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(n, B, C)
        x = self.proj(x)
        x = rearrange(x, "(b h w) B c->b c B h w", h=h, w=w)
        return x

class FEM(nn.Module):
    def __init__(self, color_dim):
        super().__init__()
        self.fem = nn.Sequential(
            nn.Conv3d(2, color_dim, kernel_size=1, stride=1),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv3d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fem(x)


@MODELS.register_module
class DU(torch.nn.Module):
    def __init__(self, dim, color_dim, stage, skip):
        super(
            DU,
            self).__init__()
        self.skip = skip
        self.stage = stage
        self.dim = dim
        self.color_dim = color_dim
        self.conv3d = FEM(color_dim=color_dim)
        self.color_dim = color_dim
        self.pho = nn.ModuleList()
        self.net = nn.ModuleList()
        for i in range(stage):
            self.pho.append(Param_Estimator(in_dim=color_dim, out_dim=color_dim))
        for i in range(stage):
            if i % (self.skip + 1) == 0:
                self.net.append(
                    STUNet(dim=dim),
                )
            else:
                self.net.append(
                    DRT(dim=dim),
                )

    def mul_PhiTg(self, Phi, g):
        return Phi * g

    def mul_Phif(self, Phi, f):
        Phif = Phi * f
        Phif = torch.sum(Phif, 1)
        return Phif.unsqueeze(1)
    def bayer_init(self, y, Phi, Phi_s):
        bayer = [[0, 0], [0, 1], [1, 0], [1, 1]]
        b, f, h, w = Phi.shape
        y_bayer = torch.zeros(b, 1, h // 2, w // 2, 4).to(y.device)
        Phi_bayer = torch.zeros(b, f, h // 2, w // 2, 4).to(y.device)
        Phi_s_bayer = torch.zeros(b, 1, h // 2, w // 2, 4).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            y_bayer[..., ib] = y[:, :, ba[0]::2, ba[1]::2]
            Phi_bayer[..., ib] = Phi[:, :, ba[0]::2, ba[1]::2]
            Phi_s_bayer[..., ib] = Phi_s[:, :, ba[0]::2, ba[1]::2]
        y_bayer = rearrange(y_bayer, "b f h w ba->(b ba) f h w")
        Phi_bayer = rearrange(Phi_bayer, "b f h w ba->(b ba) f h w")
        Phi_s_bayer = rearrange(Phi_s_bayer, "b f h w ba->(b ba) f h w")

        meas_re = torch.div(y_bayer, Phi_s_bayer)
        maskt = Phi_bayer.mul(meas_re)
        x = meas_re + maskt
        x = rearrange(x, "(b ba) f h w->b f h w ba", b=b)
        x_bayer = torch.zeros(b, f, h, w).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            x_bayer[:, :, ba[0]::2, ba[1]::2] = x[..., ib]
        return x_bayer

    def forward(self, g, input_mask=None):
        ###
        # phi[b, 8, 128, 128]
        # phi_s[b, 1, 128, 128]
        # g [b, 1, 128, 128]
        # print("g", g.shape)
        Phi, PhiPhiT = input_mask
        ratio = Phi.shape[1]

        if self.color_dim == 3:
            g = self.bayer_init(g, Phi, PhiPhiT)
        else:
            meas_re = torch.div(g, PhiPhiT)
            # meas_re = torch.unsqueeze(meas_re, 1)
            maskt = Phi.mul(meas_re)
            g = meas_re + maskt

        f0 = g.unsqueeze(1)
        Phi = Phi.unsqueeze(1)
        PhiPhiT = PhiPhiT.unsqueeze(1)
        g = g.unsqueeze(1)

        f0 = f0 / ratio
        feat = f0.expand(-1, self.dim - self.color_dim, -1, -1, -1)

        f = self.conv3d(torch.cat([f0, Phi], dim=1))

        outs = []

        prev_output = None
        for i in range(self.stage):
            pho = self.pho[i](f)
            if i % (self.skip + 1) == 0:
                z, prev_output = self.net[i](torch.cat([f, feat], dim=1))
            else:
                z, prev_output = self.net[i](torch.cat([f, feat], dim=1), prev_output)
            feat = z[:,self.color_dim:,::]
            z = z[:,:self.color_dim,::]
            Phi_f = self.mul_Phif(Phi, z)
            f = z + pho * self.mul_PhiTg(Phi, torch.div(g - Phi_f, PhiPhiT))
            outs.append(f.squeeze(1))
        return outs
