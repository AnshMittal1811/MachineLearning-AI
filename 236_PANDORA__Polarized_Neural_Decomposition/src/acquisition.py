import numpy as np
import cv2
# from skimage.transform import rescale
import polanalyser as pa

def demosaic_color_and_upsample(img_raw):
    #img_raw: CPFA H x W x 3
    #return: img_pfa_rgb: H x W x 3 x 4
    height, width = img_raw.shape[:2]
    img_pfa_rgb = np.empty((height//2, width//2, 
                            3,4),
                            dtype=img_raw.dtype)
    for j in range(2):
        for i in range(2):
            # (i,j)
            # (0,0) is 90, (0,1) is 45
            # (1,0) is 135, (1,1) is 0

            # Downsampling by 2
            img_bayer_ij = img_raw[i::2, j::2]
            
            # Color correction
            # img_bayer_cc = np.clip(apply_cc_bayer(img_bayer_ij,
            #                               'data/PMVIR_processed/ccmat.mat'),
            #                               0,1)

            # Convert images to 16 bit
            img_bayer_16b = (img_bayer_ij*(2**16-1)).astype('uint16')
            # Color demosaicking
            img_rgb_ij_16b = cv2.cvtColor(img_bayer_16b,
                                          cv2.COLOR_BayerBG2RGB_EA)
            # Convert back to float 0, 1
            img_rgb_ij = img_rgb_ij_16b.astype('float32')/(2**16-1)

            # import imageio; imageio.imwrite('viz/pmvir_rgb/image_rgb_ij.exr',img_rgb_ij)            
            # img_rgb_us = rescale(img_rgb_ij, 2,
            #                      anti_aliasing=False,
            #                      multichannel=True)
            img_rgb_us = img_rgb_ij
            # Save as stack
            img_pfa_rgb[:,:,:,2*i+j] = img_rgb_us
    
    return img_pfa_rgb

def demosaic_color(img_raw):
    #img_raw: CPFA H x W x 3
    #return: img_pfa_rgb: H//2 x W//2 x 3 x 4
    height, width = img_raw.shape[:2]
    img_pfa_rgb = np.empty((height//2, width//2, 
                            3,4),
                            dtype=img_raw.dtype)
    for j in range(2):
        for i in range(2):
            # (i,j)
            # (0,0) is 90, (0,1) is 45
            # (1,0) is 135, (1,1) is 0

            # Downsampling by 2
            img_bayer_ij = img_raw[i::2, j::2]

            
            # Color demosaicking
            img_rgb_ij = cv2.cvtColor(img_bayer_ij,
                                      cv2.COLOR_BayerBG2RGB)

            # Save as stack
            img_pfa_rgb[:,:,:,2*i+j] = img_rgb_ij
    
    return img_pfa_rgb

def apply_cc_bayer(img_bayer, cc_mat_filename):
    # Apply color correction. Needed for PMVIR image
    # Adapted from PMVIR matlab code
    # img_stack: H W 3 N_stack
    from scipy.io import loadmat
    cc_mat_RGB = loadmat(cc_mat_filename)['ccmat']
    # Copy over G values
    cc_mat_RGGB = np.zeros((4,4))
    cc_mat_RGGB[:2,:2] = cc_mat_RGB[:2,:2]
    cc_mat_RGGB[[0,1,3],3] = cc_mat_RGB[:,2]
    cc_mat_RGGB[3,[0,1,3]] = cc_mat_RGB[2,:]
    cc_mat_RGGB[[0,1,3],2] = cc_mat_RGB[:,1]
    cc_mat_RGGB[2,[0,1,3]] = cc_mat_RGB[1,:]
    cc_mat_RGGB[2,2] = cc_mat_RGB[1,1]
    # Average G values
    cc_mat_RGGB_orig = cc_mat_RGGB
    # cc_mat_RGGB[[1,2],:] = 0.5*cc_mat_RGGB_orig[[1,2],:]
    cc_mat_RGGB[:,[1,2]] = 0.5*cc_mat_RGGB_orig[:,[1,2]]

    H, W = img_bayer.shape
    img_R = img_bayer[0::2,0::2]
    img_G1 = img_bayer[0::2,1::2]
    img_G2 = img_bayer[1::2,0::2]
    img_B = img_bayer[1::2,1::2]
    img_RGGB = np.stack([img_R, img_G1, img_G2, img_B], -1)
    vec_RGGB = img_RGGB.reshape((H//2)*(W//2),4)
    # Apply matrix
    cc_vec_RGGB = vec_RGGB@(cc_mat_RGGB.T)
    cc_img_RGGB = cc_vec_RGGB.reshape(H//2,W//2,4)
    cc_img_bayer = np.zeros(img_bayer.shape)
    cc_img_bayer[0::2,0::2] = cc_img_RGGB[...,0]
    cc_img_bayer[0::2,1::2] = cc_img_RGGB[...,1]
    cc_img_bayer[1::2,0::2] = cc_img_RGGB[...,2]
    cc_img_bayer[1::2,1::2] = cc_img_RGGB[...,3]
    return cc_img_bayer

def apply_cc(img_stack, cc_mat_filename):
    # Apply color correction. Needed for PMVIR image
    # Adapted from PMVIR matlab code
    # img_stack: H W 3 N_stack
    from scipy.io import loadmat
    cc_mat = loadmat(cc_mat_filename)['ccmat']
    for stack_idx in range(img_stack.shape[-1]): 
        img = img_stack[..., stack_idx]
        H, W, _ = img.shape
        vec_RGB = img.reshape(H*W,3)
        
        # Apply matrix
        ccRGB = vec_RGB@(cc_mat.T)
        img_stack[...,stack_idx] = ccRGB.reshape(H,W,3)

    return img_stack

def preprocess_raw(img_raw, scale=1., thres=1.,depth=12,
                   cc_mat_filename='data/PMVIR_processed/ccmat.mat'):
    # import time
    # t = time.time()
    out = {}
    cc_then_dem = 1
    if cc_then_dem :
        img_raw = img_raw.astype('float32')/(2**(depth)-1)
        img_raw = scale*img_raw
        img_raw = np.minimum(img_raw, thres)
        img_pp = demosaic_color_and_upsample(img_raw)
        img_pp = apply_cc(img_pp,cc_mat_filename)
    # import imageio; imageio.imwrite('viz/img_s0.png',img_demosaiced[...,0].astype('float32')/4096)
    else:
        img_demosaiced = demosaic_color(img_raw)
        img_de = img_demosaiced
        img_pp = img_de.astype('float32')/(2**(depth)-1)
        # H x W x 3 x 4
        img_pp = scale*img_pp
        img_pp = np.minimum(img_pp, thres)
        # thres_mask = (img_pp >= thres).max(-2)[:,:,None,:]
        # img_pp = thres_mask*0.15 + (1-thres_mask)*img_pp
        # Apply color correction
        img_pp = apply_cc(img_pp,cc_mat_filename)


    out['min_max_diff'] = np.min(img_pp,-1) 
    out['min_max_spec'] = np.max(img_pp,-1) \
                          - out['min_max_diff']

    angles = np.deg2rad([90, 45, 135, 0])
    # elapsed=time.time() - t
    # print(f'Preprocessing time {elapsed}')
    # t = time.time()
    img_stokes = []
    for i_channel in range(3):
        three_pinv = 1
        if not three_pinv:
            img_stokes_channel = (pa.calcStokes(img_pp[:,:,i_channel], angles))
        else:
            # Select angle that has highest intensity and remove it 
            max_angle_ind = np.argmax(img_pp.sum(axis=(0,1,2)))
            img_pp_channel_rem = np.stack([img_pp[:,:,i_channel,a] 
                                             for a in range(len(angles)) 
                                             if a != max_angle_ind],-1)
            angles_rem = [angles[a] for a in range(len(angles)) 
                                    if a!= max_angle_ind]
            img_stokes_channel = pa.calcStokes(img_pp_channel_rem, angles_rem)
            
        img_stokes.append(img_stokes_channel)
    # elapsed=time.time() - t
    # print(f'Initial separation time {elapsed}')
    out['stokes'] =  np.stack(img_stokes, -2) #HxWx3(RGB)x3(Stokes)

    out['sat_mask'] = np.prod(out['min_max_diff']>=scale,
                              (-1,))[...,None]
    
    out['angles'] = img_pp

    return out

# def read_uint12(data_chunk):
#     # https://stackoverflow.com/a/55382153
#     data = np.frombuffer(data_chunk, dtype=np.uint8)
#     fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
#     fst_uint12 = ((mid_uint8 & 0x0F) << 8) | fst_uint8
#     snd_uint12 = (lst_uint8 << 4) | ((mid_uint8 & 0xF0) >> 4)
#     return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])

def read_uint12(data_chunk):
    data = np.frombuffer(data_chunk, dtype=np.uint16)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = (lst_uint8 << 4) + (np.bitwise_and(15, mid_uint8))
    return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])

def imread_raw(raw_filepath, img_size):
    #https://stackoverflow.com/a/65851364
    #https://stackoverflow.com/a/65952153
    with open(raw_filepath, 'rb') as f:
        data = f.read()
    # img_np = read_uint12(data)
    width=img_size[0]
    height=img_size[1]
    ic = 0
    ii = np.empty(width*height, np.uint16)
    byte_string = data
    # import struct
    # for oo in range(0,len(byte_string)-2,3):
    #     (word,) = struct.unpack('<L', byte_string[oo:oo+3] + b'\x00')
    #     ii[ic+1], ii[ic] = (word >> 12) & 0xfff, word & 0xfff
    #     ic += 2
    # img_np = ii
    # img_np_res = img_np.reshape(height, width)

    import math
    image = np.frombuffer(byte_string, np.uint8)
    num_bytes = math.ceil((width*height)*1.5)
    num_3b = math.ceil(num_bytes / 3)
    last = num_3b * 3
    image = image[:last]
    image = image.reshape(-1,3)
    image = np.hstack( (image, np.zeros((image.shape[0],1), dtype=np.uint8)) )
    image.dtype='<u4' # 'u' for unsigned int
    image = np.hstack( (image, np.zeros((image.shape[0],1), dtype=np.uint8)) )
    image[:,1] = (image[:,0] >> 12) & 0xfff
    image[:,0] = image[:,0] & 0xfff
    image = image.astype(np.uint16)
    image = image.reshape(height, width)
    img_np_res = image
    # import imageio
    # imageio.imwrite('viz/img_raw.png', img_np_res.astype('float32')/img_np_res.max())
    return img_np_res