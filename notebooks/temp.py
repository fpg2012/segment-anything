from typing import Tuple
import torch
import intel_extension_for_pytorch as ipex
import torch.nn.functional as F
import matplotlib.pyplot as plt

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    relative_coords = relative_coords.long()

    return rel_pos_resized[relative_coords]

# def blocked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor, div=4):
#     """
#     note: 
#         1. Q and K should be in the same shape
#         2. H and H must be divisable by `div`
#         3. Q, K, V must on the same device
#     """
#     B, H, W, C = Q.shape
#     assert(H//div * div == H and W//div * div == W)
#     Rh = get_rel_pos(H, H, rel_pos_h).view(div, H//div, div, H//div, C).permute(0, 2, 1, 3, 4)
#     Rw = get_rel_pos(W, W, rel_pos_w).view(W, W, C)

#     Q = Q.view(B, div, H//div, W, C).permute(1, 0, 2, 3, 4)
#     K = K.view(B, div, H//div, W, C).permute(1, 0, 2, 3, 4)
#     V = V.view(B, div, H//div, W, C).permute(1, 0, 2, 3, 4)

#     s = torch.zeros((B, H, W), dtype=torch.float32, device=Q.device)
#     x = torch.zeros((B, H, W, C), dtype=torch.float32, device=Q.device)
#     for i in range(div):
#         for p in range(div):
#             t = torch.einsum('bijc,bpqc->bijpq', Q[i], K[p])
#             t += torch.einsum('bijc,ipc->bijp', Q[i], Rh[i, p]).view(B, H//div, W, H//div, 1)
#             t += torch.einsum('bijc,jqc->bijq', Q[i], Rw).view(B, H//div, W, 1, W)
#             t = torch.exp(t)
#             s[:, i*(H//div):(i+1)*(H//div), :] += torch.sum(t, dim=(3, 4))
#             x[:, i*(H//div):(i+1)*(H//div), :, :] += torch.einsum('bijpq,bpqc->bijc', t, V[p])
#     x /= s.view((B, H, W, 1))
#     return x

def blocked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale=None, div: int =4, use_rel_pos: bool = True, rel_pos_h: torch.Tensor = None, rel_pos_w: torch.Tensor = None):
        """
        note: 
          1. Q and K should be in the same shape
          2. H and H must be divisable by `div`
          3. Q, K, V must on the same device
        """
        B, H, W, C = Q.shape
        assert(H//div * div == H and W//div * div == W)
        if scale is None:
            scale = C**-0.5
        if use_rel_pos:
            Rh = get_rel_pos(H, H, rel_pos_h).view(div, H//div, div, H//div, C).permute(0, 2, 1, 3, 4)
            Rw = get_rel_pos(W, W, rel_pos_w).view(W, W, C)

        Q = Q.view(B, div, H//div, W, C).permute(1, 0, 2, 3, 4)
        K = K.view(B, div, H//div, W, C).permute(1, 0, 2, 3, 4)
        V = V.view(B, div, H//div, W, C).permute(1, 0, 2, 3, 4)

        x = torch.zeros((B, H, W, C), dtype=torch.float32, device=Q.device)
        for i in range(div):
            s = torch.zeros((B, H//div, W, div), dtype=torch.float32, device=Q.device)
            m = torch.zeros((B, H//div, W, div), dtype=torch.float32, device=Q.device)
            y = torch.zeros((B, H//div, W, C, div), dtype=torch.float32, device=Q.device)
            for p in range(div):
                t = torch.einsum('bijc,bpqc->bijpq', Q[i] * scale, K[p])
                if use_rel_pos:
                    t += torch.einsum('bijc,ipc->bijp', Q[i], Rh[i, p]).view(B, H//div, W, H//div, 1)
                    t += torch.einsum('bijc,jqc->bijq', Q[i], Rw).view(B, H//div, W, 1, W)
                m[:, :, :, p], _ = torch.max(t.view(B, H//div, W, -1), dim=-1)
                t = torch.exp(t-(m[:, :, :, p]).view(B, H//div, W, 1, 1)).view(B, H//div, W, H//div, W)
                s[:, :, :, p] = torch.sum(t.view(B, H//div, W, -1), dim=-1)
                y[:, :, :, :, p] = torch.einsum('bijpq,bpqc->bijc', t, V[p])
            real_m, _ = torch.max(m, dim=-1)
            for p in range(div):
                s[:, :, :, p] *= torch.exp(m[:, :, :, p] - real_m)
                y[:, :, :, :, p] *= torch.exp(m[:, :, :, p] - real_m).view(B, H//div, W, 1)
            x[:, i*(H//div):(i+1)*(H//div), :, :] = torch.sum(y, dim=-1) / torch.sum(s, dim=-1).view(B, H//div, W, 1)
        
        return x

def my_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, rel_pos_h, rel_pos_w, scale, B, H, W, C):
    Rh = get_rel_pos(H, H, rel_pos_h)
    Rw = get_rel_pos(W, W, rel_pos_w)

    attn = (Q * scale) @ K.transpose(-1, -2)

    q = Q.view(B, H, W, C)
    rel_h = torch.einsum("bhwc,hkc->bhwk", q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", q, Rw)

    attn = attn.view(B, H, W, H, W)
    attn += rel_h[:, :, :, :, None]
    attn += rel_w[:, :, :, None, :]

    attn = attn.view(B, H*W, H*W)

    attn = torch.softmax(attn, dim=-1)
    x = attn @ V
    return x

def my_attention2(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, rel_pos_h, rel_pos_w, scale, B, H, W, C):
    Rh = get_rel_pos(H, H, rel_pos_h)
    Rw = get_rel_pos(W, W, rel_pos_w)

    Q = Q.view(B, H*W, C)
    K = K.view(B, H*W, C)
    V = V.view(B, H*W, C)
    attn = (Q * scale) @ K.transpose(-1, -2)

    q = Q.view(B, H, W, C)
    rel_h = torch.einsum("bhwc,hkc->bhwk", q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", q, Rw)

    attn = attn.view(B, H, W, H, W)
    attn += rel_h[:, :, :, :, None]
    attn += rel_w[:, :, :, None, :]

    attn = attn.view(B, H*W, H*W)

    attn = torch.softmax(attn, dim=-1)
    x = attn @ V
    return x

def real_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor, scale, H, W):
    attn = (q * scale) @ k.transpose(-2, -1)
    attn = add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, (H, W), (H, W))
    attn = attn.softmax(dim=-1)
    x = (attn @ v)
    return x

def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

if __name__ == '__main__':
    B, n_head, H, W, C = 1, 16, 64, 64, 80
    q = torch.load('q.pt')
    k = torch.load('k.pt')
    v = torch.load('v.pt')

    rel_pos_h = torch.load('rel_pos_h.pt')
    rel_pos_w = torch.load('rel_pos_w.pt')

    scale = C**-0.5

    result1 = real_attention(q, k, v, rel_pos_h, rel_pos_w, scale, H, W)
    result3 = my_attention(q, k, v, rel_pos_h, rel_pos_w, scale, B*n_head, H, W, C)
    

    q1 = q.view(16, H, W, -1)
    k1 = k.view(16, H, W, -1)
    v1 = v.view(16, H, W, -1)

    result5 = my_attention2(q1, k1, v1, rel_pos_h, rel_pos_w, scale, B*n_head, H, W, C)
    result2 = blocked_attention(q1, k1, v1, scale=scale, use_rel_pos=True, rel_pos_h=rel_pos_h, rel_pos_w=rel_pos_w, div=4)

    result4 = torch.load('x1.pt')

    print('eq1: ', result1 - result4)
    print('eq2: ', result2.view(16, H * W, -1) - result3)
    # print('equality3: ', result1 - result3.view(1, 16, H, W, -1).permute(0, 2, 3, 1, 4).reshape(1, H, W, -1))
    print('eq3: ', result1 - result3)
    print('eq4: ', result5 - result3)
    print('eq5: ', q1.view(16, H*W, C) == q.view(16, H*W, C))