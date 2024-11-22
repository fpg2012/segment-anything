from typing import Tuple
import torch
import intel_extension_for_pytorch as ipex

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

def blocked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, rel_pos_emb: torch.Tensor, div=4):
    B, H, W, C = Q.shape
    Rh = get_rel_pos(H, H, rel_pos_emb[0]).view(B, div, H//div, div, H//div, C).permute(1, 3, 0, 2, 4, 5)
    Rw = get_rel_pos(W, W, rel_pos_emb[1]).view(B, W, W, C)

    Q = Q.view(B, div, H//div, W, C).permute(1, 0, 2, 3, 4)
    K = K.view(B, div, H//div, W, C).permute(1, 0, 2, 3, 4)
    V = V.view(B, div, H//div, W, C).permute(1, 0, 2, 3, 4)

    s = torch.zeros((B, H, W), dtype=torch.float32)
    x = torch.zeros((B, H, W, C), dtype=torch.float32)
    for i in range(div):
        for p in range(div):
            t = torch.einsum('bijc,bpqc->bijpq', Q[i], K[p])
            t += torch.einsum('bijc,bipc->bijp', Q[i], Rh[i, p]).view(B, H//div, W, H//div, 1)
            t += torch.einsum('bijc,bjqc->bijq', Q[i], Rw).view(B, H//div, W, 1, W)
            t = torch.exp(t)
            s[:, i*(H//div):(i+1)*(H//div), :] += torch.sum(t, dim=(3, 4))
            x[:, i*(H//div):(i+1)*(H//div), :, :] += torch.einsum('bijpq,bpqc->bijc', t, V[p])
    x /= s.view((B, H, W, 1))
    return x

def real_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, rel_pos_emb: torch.Tensor):
    B, H, W, C = Q.shape
    Rh = get_rel_pos(H, H, rel_pos_emb[0])
    Rw = get_rel_pos(W, W, rel_pos_emb[1])

    Q = Q.view(B, H*W, C)
    K = K.view(B, H*W, C)
    attn = Q @ K.transpose(-1, -2)

    rel_h = torch.einsum("bhwc,hkc->bhwk", Q.view(B, H, W, C), Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", Q.view(B, H, W, C), Rw)

    attn = attn.view(B, H, W, H, W)
    attn += rel_h[:, :, :, :, None]
    attn += rel_w[:, :, :, None, :]

    attn = attn.view(B, H*W, H*W)

    attn = torch.softmax(attn, dim=-1)
    x = attn @ V.view(B, H*W, C)
    return x.view(B, H, W, C)

if __name__ == '__main__':
    B, H, W, C = 1, 16, 16, 8
    q = torch.randn((B, H, W, C))
    k = torch.randn((B, H, W, C))
    v = torch.randn((B, H, W, C))

    rel_pos_emb = torch.randn((2, 2*H - 1, 8))

    result1 = real_attention(q, k, v, rel_pos_emb)
    print('real: ', result1.shape, result1)

    result2 = blocked_attention(q, k, v, rel_pos_emb, div=4)
    print('blocked: ', result2.shape, result2)

    print('equality: ', result1 - result2)