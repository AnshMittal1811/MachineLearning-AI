import torch

def dirup_to_rotmat(dir_vec, up_vec):
    # We do not assume that the inputs are normalized nor orthogonal
    # Debug dot_prod function
    # dot_prod = lambda a, b : torch.sum(a[...,:]*b[...,:], dim=-1)
    # debug = lambda label, tensor : print(f"{label}={tensor.shape}")

    # Renormalize the inputs
    up_vec_norm = up_vec / torch.sqrt(torch.sum(up_vec**2, dim=-1, keepdim=True))
    dir_vec_norm = dir_vec / torch.sqrt(torch.sum(dir_vec**2, dim=-1, keepdim=True))

    # Make up and dir orthogonal and create the side vector
    side_vec_norm = torch.cross(up_vec_norm, dir_vec_norm, dim=-1)
    up_vec_norm = torch.cross(dir_vec_norm, side_vec_norm, dim=-1)

    # Those three vectors create the new rotation basis
    flat_rot_mat = torch.cat((up_vec_norm,
                              dir_vec_norm,
                              side_vec_norm), dim=-1)
    # Row-wise reshaping rot_mat=[up^t; dir^t; side^t]
    rot_mat = flat_rot_mat.reshape(-1, 3, 3)
    return rot_mat


def rotmat_to_euler(rot_mat):
    # XYZ convention is used here
    # debug = lambda label, tensor : print(f"{label}={tensor.shape}")

    norm_ry = torch.sqrt(rot_mat[..., 0, 0]**2 + rot_mat[..., 1, 0]**2)  # [num_rays]
    isSingular = norm_ry < torch.tensor(1e-6)  # [num_rays]

    euler_ang = torch.where(isSingular[..., None],
                            torch.cat((torch.atan2(-rot_mat[..., 1, 2], rot_mat[..., 1, 1])[..., None],
                                       torch.atan2(-rot_mat[..., 2, 0], norm_ry)[..., None],
                                       torch.zeros_like(norm_ry[..., None])), dim=-1),
                            torch.cat((torch.atan2(rot_mat[..., 2, 1], rot_mat[..., 2, 2])[..., None],
                                       torch.atan2(-rot_mat[..., 2, 0], norm_ry)[..., None],
                                       torch.atan2(rot_mat[..., 1, 0], rot_mat[..., 0, 0])[..., None]), dim=-1)
                            )
    return euler_ang


def rotmat_to_quaternion(rot_mat):
    # Pseudo code:
    # -------------------------------
    # if (m22 < 0){
    #     if (m00 > m11) {
    #         t = 1 + m00 - m11 - m22;
    #         q = quat(t, m01 + m10, m20 + m02, m12 - m21);}
    #     else {
    #         t = 1 - m00 + m11 - m22;
    #         q = quat(m01 + m10, t, m12 + m21, m20 - m02);}
    # }
    # else{
    #     if (m00 < -m11){
    #         t = 1 - m00 - m11 + m22;
    #         q = quat(m20 + m02, m12 + m21, t, m01 - m10);}
    #     else{
    #         t = 1 + m00 + m11 + m22;
    #         q = quat(m12 - m21, m20 - m02, m01 - m10, t);}
    # }
    # q *= 0.5 / Sqrt(t);
    # -------------------------------
    r = rot_mat
    q = 0.5 * torch.where(r[..., 2, 2] < 0,
                          torch.where(r[..., 0, 0] > r[..., 1, 1],
                                      # case 1
                                      torch.stack((1. + r[..., 0, 0] - r[..., 1, 1] - r[..., 2, 2],
                                                  r[..., 0, 1] + r[..., 1, 0],
                                                  r[..., 2, 0] + r[..., 0, 2],
                                                  r[..., 1, 2] - r[..., 2, 1]), dim=0) /
                                      torch.sqrt(1. + r[..., 0, 0] - r[..., 1, 1] - r[..., 2, 2]),
                                      # case 2
                                      torch.stack((r[..., 0, 1] + r[..., 1, 0],
                                                  1. - r[..., 0, 0] + r[..., 1, 1] - r[..., 2, 2],
                                                  r[..., 1, 2] + r[..., 2, 1],
                                                  r[..., 2, 0] - r[..., 0, 2]), dim=0) /
                                      torch.sqrt(1. - r[..., 0, 0] + r[..., 1, 1] - r[..., 2, 2]),),
                          torch.where(r[..., 0, 0] < -r[..., 1, 1],
                                      # case 3
                                      torch.stack((r[..., 2, 0] + r[..., 0, 2],
                                                   r[..., 1, 2] + r[..., 2, 1],
                                                   1. - r[..., 0, 0] - r[..., 1, 1] + r[..., 2, 2],
                                                   r[..., 0, 1] - r[..., 1, 0]), dim=0) /
                                      torch.sqrt(1. - r[..., 0, 0] - r[..., 1, 1] + r[..., 2, 2]),
                                      # case 4
                                      torch.stack((r[..., 1, 2] - r[..., 2, 1],
                                                   r[..., 2, 0] - r[..., 0, 2],
                                                   r[..., 0, 1] - r[..., 1, 0],
                                                   1. + r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]), dim=0) /
                                      torch.sqrt(1. + r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]),))
    return q.permute(1, 0)


def rotmat_to_pseudoquat(rot_mat):
    # ad = 0.25*(r32-r23)
    # bd = 0.25*(r13-r31)
    # cd = 0.25*(r21-r12)
    # dd = 0.25*(1+r11+r22+r33)
    r = rot_mat

    dd = 0.25 * (1 + r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2])  # has nothing to do with r
    eps = 1e-3
    q = torch.where(dd >= eps ** 2,
                    torch.stack((0.25 * (r[..., 2, 1] - r[..., 1, 2]),
                                0.25 * (r[..., 0, 2] - r[..., 2, 0]),
                                0.25 * (r[..., 1, 0] - r[..., 0, 1]),
                                dd), dim=0
                                ),
                    torch.stack((0.5 * eps * torch.sign(r[..., 2, 1] - r[..., 1, 2]) * torch.sqrt(r[..., 0, 0] - r[..., 1, 1] - r[..., 2, 2] + 1),
                                 0.5 * eps * torch.sign(r[..., 0, 2] - r[..., 2, 0]) * torch.sqrt(-r[..., 0, 0] + r[..., 1, 1] - r[..., 2, 2] + 1),
                                 0.5 * eps * torch.sign(r[..., 1, 0] - r[..., 0, 1]) * torch.sqrt(-r[..., 0, 0] - r[..., 1, 1] + r[..., 2, 2] + 1),
                                eps * torch.sqrt(dd)), dim=0
                                ))
    return q.permute(1, 0)


def pseudoquat_to_rotmat(q):
    # pseudo quat = [a r, b r, c r, d r]
    # rotmat = [a2-b2-c2+d2, 2(ab-cd), 2(ac+bd),
    #           2(ab+cd),   b2-c2-a2+d2, 2(bc-ad),
    #           2(ac-bd),   2(bc+ad),   c2-a2-b2-d2]
    ad_ = q[..., 0]
    bd_ = q[..., 1]
    cd_ = q[..., 2]
    dd_ = q[..., 3]

    sxd2 = ad_**2 + bd_**2 + cd_**2 + dd_**2

    # Products
    a2 = ad_ ** 2 / sxd2
    b2 = bd_ ** 2 / sxd2
    c2 = cd_ ** 2 / sxd2
    d2 = dd_ ** 2 / sxd2

    ab = ad_*bd_ / sxd2
    ac = ad_*cd_ / sxd2
    cd = cd_*dd_ / sxd2
    bd = bd_*dd_ / sxd2
    bc = bd_*cd_ / sxd2
    ad = ad_*dd_ / sxd2

    rot_mat = torch.cat((a2-b2-c2+d2, 2*(ab-cd),   2*(ac+bd),
                         2*(ab+cd),   b2-c2-a2+d2, 2*(bc-ad),
                         2*(ac-bd),   2*(bc+ad),   c2-a2-b2-d2), dim=-1).reshape(-1, 3, 3)

    return rot_mat


def dist_between_rotmats(rot1, rot2_transpose):
    diff_rot = rot1.matmul(rot2_transpose)
    return torch.acos(.5*(torch.trace(diff_rot)-1.))


# Unit test here
def main():
    # euler = [ x: 0.8, y: -0.9, z: -0.6 ] (XYZ convention)
    #       = [ x: 1.122394, y: -0.045392, z: -1.0314609 ] (ZYX convention)
    # quaternion = [ 0.4533835, -0.2791119, -0.4069129, 0.742268 ] (x,y,z,w)
    rot_mat = torch.tensor([[0.5130368, 0.3509874, -0.7833269],
                            [-0.8571663, 0.2577305, -0.4459157],
                            [0.0453765, 0.9002126, 0.4330798]],
                           dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
    print(f"rot_mat=\n{rot_mat}")

    euler = rotmat_to_euler(rot_mat)
    print(f"euler=\n{euler}")

    quaternion = rotmat_to_quaternion(rot_mat)
    print(f"quaternion=\n{quaternion}")

    pseudoquat = rotmat_to_pseudoquat(rot_mat)
    print(f"pseudoquat=\n{pseudoquat}")
    rot_mat_rec = pseudoquat_to_rotmat(pseudoquat)
    print(f"rec_rot_mat=\n{rot_mat_rec}")


if __name__ == '__main__':
    main()



