import torch  
import numpy as np 
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


__all__ = ['Loss_mixing', 'Loss_cosine', 'Loss_contrastive', 
            'Loss_cosine_attn', 'Loss_condition_orth_weight']

# Embedding Level Size: (Batch-size, Tokens, Dims * Heads)
# Attention Level Size: (Batch-size, Heads, Tokens, Tokens) -> (Batch-size, Heads, Tokens * Tokens)
# Similarity Regularization, input: (Batch-size, Diverse-Target, Dimension)

################# Main Regularization ###############

def Loss_mixing(output, patch_target):
    # output (B,197,384)
    # patch_target (B,)
    criterion = SoftTargetCrossEntropy()
    patch_num = output.shape[1]
    loss = 0
    for i in range(1,patch_num):
        loss += criterion(output[:,i], patch_target[:,i-1])
    return loss, patch_num

def Loss_cosine(h_emb, eps=1e-8):
    # h_emb (B, Tokens, dims * heads)
    # normalize
    target_h_emb = h_emb[:,1:]
    hshape = target_h_emb.shape 
    target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1)
    a_n = target_h_emb.norm(dim=2).unsqueeze(2)
    a_norm = target_h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # patch-wise absolute value of cosine similarity
    sim_matrix = torch.einsum('abc,acd->abd', a_norm, a_norm.transpose(1,2))
    loss_cos = sim_matrix.mean()

    return loss_cos

def Loss_contrastive(h1_emb, hl_emb, eps=1e-8):
    
    h1_emb_target = h1_emb[:,1:]
    hl_emb_target = hl_emb[:,1:]

    hshape = h1_emb_target.shape 
    # h1_emb_target = h1_emb_target.reshape(hshape[0], hshape[1], -1).detach()
    h1_emb_target = h1_emb_target.reshape(hshape[0], hshape[1], -1)
    h1_n = h1_emb_target.norm(dim=2).unsqueeze(2)
    h1_norm = h1_emb_target/torch.max(h1_n, eps*torch.ones_like(h1_n))

    hl_emb_target = hl_emb_target.reshape(hshape[0], hshape[1], -1)
    hl_n = hl_emb_target.norm(dim=2).unsqueeze(2)
    hl_norm = hl_emb_target/torch.max(hl_n, eps*torch.ones_like(hl_n))

    sim_matrix = torch.einsum('abc,adc->abd', h1_norm, hl_norm)
    sim_diag = torch.diagonal(sim_matrix, dim1=1, dim2=2)
    dim2 = sim_diag.shape[1]
    exp_sim_diag = torch.exp(sim_diag)
    temp_sim = torch.sum(sim_matrix, dim=2)
    temp_sim = torch.exp((temp_sim-sim_diag)/(dim2-1))
    nce = -torch.log(exp_sim_diag/(exp_sim_diag+temp_sim))
    return nce.mean()

def Loss_cosine_attn(h_emb, eps=1e-8):
    # h_emb (B, Tokens, dims * heads)
    # normalize
    target_h_emb = h_emb
    hshape = target_h_emb.shape 
    target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1)
    a_n = target_h_emb.norm(dim=2).unsqueeze(2)
    a_norm = target_h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # patch-wise absolute value of cosine similarity
    sim_matrix = torch.einsum('abc,acd->abd', a_norm, a_norm.transpose(1,2))
    loss_cos = sim_matrix.mean() # also add diagnoal elements
    return loss_cos

def dominant_eigenvalue(A, dev):
    N, _ = A.size()
    x = torch.rand(N, 1, device=dev)

    Ax = (A @ x)
    AAx = (A @ Ax)

    return AAx.permute(1, 0) @ Ax / (Ax.permute(1, 0) @ Ax)

def get_singular_values(A, dev):
    ATA = A.permute(1, 0) @ A
    N, _ = ATA.size()
    largest = dominant_eigenvalue(ATA, dev)
    I = torch.eye(N, device=dev)  
    I = I * largest  
    tmp = dominant_eigenvalue(ATA - I, dev)
    return tmp + largest, largest

def Loss_condition_orth_weight(W):
    W = W.permute(1, 0) # (in, out)
    smallest, largest = get_singular_values(W, W.device)
    return torch.mean((largest - smallest)**2)







################# Additional Regularization ###############
def loss_cosine_reg(h_emb, eps=1e-8):
    # h_emb (B, Tokens, dims * heads)
    # normalize
    target_h_emb = h_emb[:,1:]
    hshape = target_h_emb.shape 
    target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1)
    a_n = target_h_emb.norm(dim=2).unsqueeze(2)
    a_norm = target_h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # patch-wise absolute value of cosine similarity
    sim_matrix = torch.einsum('abc,acd->abd', a_norm, a_norm.transpose(1,2))
    loss_cos = sim_matrix.abs().mean() # also add diagnoal elements

    return loss_cos

def loss_cosine_attn_reg(h_emb, eps=1e-8):
    # h_emb (B, Tokens, dims * heads)
    # normalize
    target_h_emb = h_emb
    hshape = target_h_emb.shape 
    target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1)
    a_n = target_h_emb.norm(dim=2).unsqueeze(2)
    a_norm = target_h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # patch-wise absolute value of cosine similarity
    sim_matrix = torch.einsum('abc,acd->abd', a_norm, a_norm.transpose(1,2))
    loss_cos = sim_matrix.abs().mean() # also add diagnoal elements

    return loss_cos

def loss_cosine_across_reg(h_emb, h_emb2, eps=1e-8):
    # h_emb (B, Tokens, dims * heads)
    # normalize
    target_h_emb = h_emb[:,1:]
    target_h_emb2 = h_emb2[:,1:]

    hshape = target_h_emb.shape 
    # target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1).detach()
    target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1)
    a_n = target_h_emb.norm(dim=2).unsqueeze(2)
    a_norm = target_h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    target_h_emb2 = target_h_emb2.reshape(hshape[0], hshape[1], -1)
    a_n2 = target_h_emb2.norm(dim=2).unsqueeze(2)
    a_norm2 = target_h_emb2 / torch.max(a_n2, eps * torch.ones_like(a_n2))

    # patch-wise absolute value of cosine similarity
    sim_matrix = torch.einsum('abc,acd->abd', a_norm, a_norm2.transpose(1,2))
    loss_cos = sim_matrix.abs().mean() # also add diagnoal elements

    return loss_cos

def loss_cosine_across_attn_reg(h_emb, h_emb2, eps=1e-8):
    # h_emb (B, Tokens, dims * heads)
    # normalize
    target_h_emb = h_emb
    target_h_emb2 = h_emb2

    hshape = target_h_emb.shape 
    # target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1).detach()
    target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1)
    a_n = target_h_emb.norm(dim=2).unsqueeze(2)
    a_norm = target_h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    target_h_emb2 = target_h_emb2.reshape(hshape[0], hshape[1], -1)
    a_n2 = target_h_emb2.norm(dim=2).unsqueeze(2)
    a_norm2 = target_h_emb2 / torch.max(a_n2, eps * torch.ones_like(a_n2))

    # patch-wise absolute value of cosine similarity
    sim_matrix = torch.einsum('abc,acd->abd', a_norm, a_norm2.transpose(1,2))
    loss_cos = sim_matrix.abs().mean() # also add diagnoal elements

    return loss_cos

def loss_cosine_across_reg_noabs(h_emb, h_emb2, eps=1e-8):
    # h_emb (B, Tokens, dims * heads)
    # normalize
    target_h_emb = h_emb[:,1:]
    target_h_emb2 = h_emb2[:,1:]

    hshape = target_h_emb.shape 
    # target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1).detach()
    target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1)
    a_n = target_h_emb.norm(dim=2).unsqueeze(2)
    a_norm = target_h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    target_h_emb2 = target_h_emb2.reshape(hshape[0], hshape[1], -1)
    a_n2 = target_h_emb2.norm(dim=2).unsqueeze(2)
    a_norm2 = target_h_emb2 / torch.max(a_n2, eps * torch.ones_like(a_n2))

    # patch-wise absolute value of cosine similarity
    sim_matrix = torch.einsum('abc,acd->abd', a_norm, a_norm2.transpose(1,2))
    loss_cos = sim_matrix.mean() # also add diagnoal elements

    return loss_cos

def loss_cosine_across_attn_reg_noabs(h_emb, h_emb2, eps=1e-8):
    # h_emb (B, Tokens, dims * heads)
    # normalize
    target_h_emb = h_emb
    target_h_emb2 = h_emb2

    hshape = target_h_emb.shape 
    # target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1).detach()
    target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1)
    a_n = target_h_emb.norm(dim=2).unsqueeze(2)
    a_norm = target_h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    target_h_emb2 = target_h_emb2.reshape(hshape[0], hshape[1], -1)
    a_n2 = target_h_emb2.norm(dim=2).unsqueeze(2)
    a_norm2 = target_h_emb2 / torch.max(a_n2, eps * torch.ones_like(a_n2))

    # patch-wise absolute value of cosine similarity
    sim_matrix = torch.einsum('abc,acd->abd', a_norm, a_norm2.transpose(1,2))
    loss_cos = sim_matrix.mean() # also add diagnoal elements

    return loss_cos

def loss_contrastive_attn_reg(h1_emb_target, hl_emb_target, eps=1e-8):
    
    hshape = h1_emb_target.shape 
    # h1_emb_target = h1_emb_target.reshape(hshape[0], hshape[1], -1).detach()
    h1_emb_target = h1_emb_target.reshape(hshape[0], hshape[1], -1)
    h1_n = h1_emb_target.norm(dim=2).unsqueeze(2)
    h1_norm = h1_emb_target/torch.max(h1_n, eps*torch.ones_like(h1_n))

    hl_emb_target = hl_emb_target.reshape(hshape[0], hshape[1], -1)
    hl_n = hl_emb_target.norm(dim=2).unsqueeze(2)
    hl_norm = hl_emb_target/torch.max(hl_n, eps*torch.ones_like(hl_n))

    sim_matrix = torch.einsum('abc,adc->abd', h1_norm, hl_norm)
    sim_diag = torch.diagonal(sim_matrix, dim1=1, dim2=2)
    dim2 = sim_diag.shape[1]
    exp_sim_diag = torch.exp(sim_diag)
    temp_sim = torch.sum(sim_matrix, dim=2)
    temp_sim = torch.exp((temp_sim-sim_diag)/(dim2-1))
    nce = -torch.log(exp_sim_diag/(exp_sim_diag+temp_sim))
    return nce.mean()


# Uniformity Regularization, weight: (Diverse-Target, Dimention) Embedding: (Batch-size, Diverse-Target, Dimension)
def norm(filt):
    # filt (dim, out_dim)
    filt_norm = ((filt * filt).sum(dim=0) + 1e-8).sqrt()
    filt_norm = filt_norm.reshape(1, filt.shape[1])
    return filt / filt_norm

def cal(filt):
    filt_norm = ((filt * filt).sum(dim=0) + 1e-8).sqrt()
    filt_norm = filt_norm.reshape(1, filt.shape[1])
    norm_mat = torch.matmul(filt_norm.transpose(1,0), filt_norm)
    inner_pro = torch.matmul(filt.transpose(1,0), filt)
    return inner_pro / norm_mat

def loss_mhs_weight_reg(filt):
    # filt (output_dim, input_dim)
    filt = filt.transpose(1,0) # (in, out)
    filt = norm(filt)
    inner_pro = cal(filt)
    final = (2.0 - 2.0 * inner_pro)

    final -= torch.triu(final)
    nonzeros = torch.where(final!=0)
    target = torch.min(final[nonzeros])

    mask = final.eq(target)
    loss = -(final * mask.detach()).sum()

    return loss 

def norm_feature(filt):
    filt_shape = filt.shape # batch-size, output_dim, input_dim
    filt_norm = ((filt * filt).sum(dim=2) + 1e-8).sqrt()
    filt_norm = filt_norm.reshape(filt_shape[0], filt_shape[1], 1)
    return filt / filt_norm

def cal_feature(filt):
    filt_shape = filt.shape # batch-size, output_dim, input_dim

    filt_norm = ((filt * filt).sum(dim=2) + 1e-8).sqrt()
    filt_norm = filt_norm.reshape(filt_shape[0], filt_shape[1], 1)
    norm_mat = torch.einsum('bac,bdc->bad', filt_norm, filt_norm)
    inner_pro = torch.einsum('bac,bdc->bad', filt, filt)

    return inner_pro / norm_mat

def loss_mhs_feature_reg(filt):
    # filt (batch-size, output_dim, input_dim)
    batch_size = filt.shape[0]
    target_dim = filt.shape[1]
    filt = filt.reshape(batch_size, target_dim, -1)
    filt = norm_feature(filt)
    inner_pro = cal_feature(filt)
    final = (2.0 - 2.0 * inner_pro)
    final -= torch.triu(final)

    loss = 0
    for sample in range(batch_size):
        nonzeros = torch.where(final[sample,:,:]!=0)
        if nonzeros[0].shape[0] > 0:
            target = torch.min(final[sample,:,:][nonzeros])
            mask = final[sample,:,:].eq(target)
            loss += (final[sample,:,:] * mask.detach()).sum()

    return -loss/batch_size

def loss_mgd_weight_reg(filt):

    # filt (output_dim, input_dim)
    n_filt = filt.shape[0]
    filt = filt.transpose(1,0) # (in, out)
    filt = norm(filt)
    inner_pro = cal(filt)
    cross_terms = (2.0 - 2.0 * inner_pro)
    final = torch.exp(-1 * cross_terms) + torch.diag(1e-6 * torch.ones(n_filt).to(filt.device))

    loss = -torch.logdet(final)

    return loss 

def loss_mgd_feature_reg(filt):

    # filt (batch-size, output_dim, input_dim)
    batch_size = filt.shape[0]
    out_dim = filt.shape[1]
    filt = filt.reshape(batch_size, out_dim, -1)
    filt = norm_feature(filt)
    inner_pro = cal_feature(filt)
    cross_terms = (2.0 - 2.0 * inner_pro)
    offset = torch.diag(1e-6 * torch.ones(out_dim).to(filt.device)).repeat(batch_size, 1, 1)
    final = torch.exp(-1 * cross_terms) + offset
    loss = -torch.logdet(final).mean()

    return loss 

def loss_condition_orth_weight_reg_inverse(W):
    smallest, largest = get_singular_values(W, W.device)
    return torch.mean((largest - smallest)**2)

def loss_s_orth_weight_reg(A):
    ATA = A @ A.permute(1, 0)
    N, _ = ATA.size()
    I = torch.eye(N, device=A.device)
    fnorm = torch.norm(ATA-I, p='fro')
    return fnorm**2

def features_dominant_eigenvalue(A):
    device = A.device
    B, N, _ = A.size()
    x = torch.randn(B, N, 1).to(device)

    for _ in range(1):
        x = torch.bmm(A, x)

    numerator = torch.bmm(
        torch.bmm(A, x).view(B, 1, N),
        x
    ).squeeze()
    denominator = (torch.norm(x.view(B, N), p=2, dim=1) ** 2).squeeze()

    return numerator / (denominator + 1e-6)

def features_get_singular_values(A):
    device = A.device
    AAT = torch.bmm(A, A.permute(0, 2, 1))
    B, N, _ = AAT.size()
    largest = features_dominant_eigenvalue(AAT)
    I = torch.eye(N).expand(B, N, N).to(device)
    I = I * largest.view(B, 1, 1).repeat(1, N, N)
    tmp = features_dominant_eigenvalue(AAT - I)
    return tmp + largest, largest

def loss_condition_orth_embedding_reg(fea, eps=1e-8):

    # (batch-size, diverse-target, dimension)
    B, N = fea.size(0), fea.size(1)
    new_fea = fea.view(B, N, -1)
    fea_n = new_fea.norm(dim=2).unsqueeze(2)
    new_fea_norm = new_fea/torch.max(fea_n, eps*torch.ones_like(fea_n))
    smallest, largest = features_get_singular_values(new_fea_norm)

    return torch.mean((largest - smallest)**2)

def loss_condition_orth_attn_reg(fea):
    # (bs, diverse-target, dim, dim)
    B, H = fea.size(0), fea.size(1)
    new_fea = fea.view(B, H, -1)
    smallest, largest = features_get_singular_values(new_fea)
    return torch.mean((largest - smallest)**2)

def loss_s_orth_attn_reg(A):
    # attn (Batch-size, Heads, Tokens, Tokens)
    adevice = A.device
    B, H = A.shape[0], A.shape[1]
    A = A.view(B, H, -1)

    ATA = A @ A.permute(0,2,1)
    I = torch.eye(H, device=adevice).repeat(B,1,1)
    norm_pow2 = (ATA-I)**2
    loss = norm_pow2.sum(dim=2).sum(dim=1).mean()
    return loss 

def loss_s_orth_embedding_reg(A, eps=1e-8):
    # (batch-size, diverse-target, dimension)
    adevice = A.device
    B, H = A.shape[0], A.shape[1]
    A = A.view(B, H, -1)
    fea_n = A.norm(dim=2).unsqueeze(2)
    new_fea_norm = A/torch.max(fea_n, eps*torch.ones_like(fea_n))

    ATA = new_fea_norm @ new_fea_norm.permute(0,2,1)
    I = torch.eye(H, device=adevice).repeat(B,1,1)
    norm_pow2 = (ATA-I)**2
    loss = norm_pow2.sum(dim=2).sum(dim=1).mean()
    return loss 

# Gradient Regularization: Only last Embedding: (Batch-size, Diverse-Target, Dimension)
def loss_grad_diversity_reg(grad_tensor, eps=1e-8):
    # grad_tensor (Batch-size, Diverse-Target, Dimension)
    grad_tensor = torch.where(torch.isnan(grad_tensor), eps*torch.ones_like(grad_tensor), grad_tensor)
    token_sum_grad_tensor = grad_tensor.sum(dim=1)
    sum_norm = (token_sum_grad_tensor ** 2).sum(dim=1)
    norm_sum = (grad_tensor ** 2).sum(dim=2).sum(dim=1)
    loss = norm_sum/sum_norm
    return -loss.mean()




