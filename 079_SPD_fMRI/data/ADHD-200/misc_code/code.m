%%
%define threshold, see spdaux_adjust for threshold interpretation

thr=1e-15;

%%
%initialize

KKI = cat(3,KKI_hc,KKI_adhd);
NYU = cat(3,NYU_hc,NYU_adhd);
PKG = cat(3,PKG_hc,PKG_adhd);
NIM = cat(3,NIM_hc,NIM_adhd);

KKI_spd = spd_initialize(KKI,thr);
NYU_spd = spd_initialize(NYU,thr);
PKG_spd = spd_initialize(PKG,thr);
NIM_spd = spd_initialize(NIM,thr);

KKI_h = spd_initialize(KKI_hc,thr);
KKI_a = spd_initialize(KKI_adhd,thr);
NYU_h = spd_initialize(NYU_hc,thr);
NYU_a = spd_initialize(NYU_adhd,thr);
PKG_h = spd_initialize(PKG_hc,thr);
PKG_a = spd_initialize(PKG_adhd,thr);
NIM_a = spd_initialize(NIM_hc,thr);
NIM_h = spd_initialize(NIM_adhd,thr);

%%
%compute Fréchet means

[KKI_mean_ext, ] = spd_mean(KKI_spd, 1e-8, "extrinsic");
[NIM_mean_ext, ] = spd_mean(NIM_spd, 1e-8, "extrinsic");
[NYU_mean_ext, ] = spd_mean(NYU_spd, 1e-8, "extrinsic");
[PKG_mean_ext, ] = spd_mean(PKG_spd, 1e-8, "extrinsic");

[KKI_mean_int, ] = spd_mean(KKI_spd, 1e-8, "intrinsic");
[NIM_mean_int, ] = spd_mean(NIM_spd, 1e-8, "intrinsic");
[NYU_mean_int, ] = spd_mean(NYU_spd, 1e-8, "intrinsic");
[PKG_mean_int, ] = spd_mean(PKG_spd, 1e-8, "intrinsic");

all_means_ext = cat(3,KKI_mean_ext,NYU_mean_ext,PKG_mean_ext);
all_m_ext = spd_initialize(all_means_ext,thr);
all_means_int = cat(3,KKI_mean_int,NYU_mean_int,PKG_mean_int);
all_m_int = spd_initialize(all_means_int,thr);
[global_mean_ext, ] = spd_mean(all_m_ext, 1e-8, "extrinsic");
[global_mean_int, ] = spd_mean(all_m_int, 1e-8, "intrinsic");

%%
%matrix whitening

KKI_mw=zeros(190,190,40);
NYU_mw=zeros(190,190,40);
PKG_mw=zeros(190,190,40);
NIM_mw=zeros(190,190,40);

for idx=1:40
    KKI_mw(:,:,idx)=((KKI_mean_int)^(-1/2))*KKI(:,:,idx)*((KKI_mean_int)^(-1/2));
    NYU_mw(:,:,idx)=((NYU_mean_int)^(-1/2))*NYU(:,:,idx)*((NYU_mean_int)^(-1/2));
    PKG_mw(:,:,idx)=((PKG_mean_int)^(-1/2))*PKG(:,:,idx)*((PKG_mean_int)^(-1/2));
    NIM_mw(:,:,idx)=((NIM_mean_int)^(-1/2))*NIM(:,:,idx)*((NIM_mean_int)^(-1/2));
end

%%
%rigid log-euclidean translation

logKKI = zeros(190,190,40);
logNYU = zeros(190,190,40);
logPKG = zeros(190,190,40);
logNIM = zeros(190,190,40);
    
for idx = 1:40
    logKKI(:,:,idx) = logm(KKI(:,:,idx));
    logNYU(:,:,idx) = logm(NYU(:,:,idx));
    logPKG(:,:,idx) = logm(Peking_1(:,:,idx));
    logNIM(:,:,idx) = logm(NIM(:,:,idx));
end

%RLET(\overline{\Sigma})

KKI_rlet = zeros(190,190,40);
NYU_rlet = zeros(190,190,40);
PKG_rlet = zeros(190,190,40);
NIM_rlet = zeros(190,190,40);

for idx = 1:40
    KKI_rlet(:,:,idx) = expm(logm(global_mean_ext) + logKKI(:,:,idx) - logm(KKI_mean_ext));
    NYU_rlet(:,:,idx) = expm(logm(global_mean_ext) + logNYU(:,:,idx) - logm(NYU_mean_ext));
    PKG_rlet(:,:,idx) = expm(logm(global_mean_ext) + logPKG(:,:,idx) - logm(PKG_mean_ext));
    NIM_rlet(:,:,idx) = expm(logm(global_mean_ext) + logNIM(:,:,idx) - logm(NIM_mean_ext));
end

%RLET(I)

KKI_rletI = zeros(190,190,40);
NYU_rletI = zeros(190,190,40);
PKG_rletI = zeros(190,190,40);
NIM_rletI = zeros(190,190,40);

for idx = 1:40
    KKI_rletI(:,:,idx) = expm(logKKI(:,:,idx) - logm(KKI_mean_ext));
    NYU_rletI(:,:,idx) = expm(logNYU(:,:,idx) - logm(NYU_mean_ext));
    PKG_rletI(:,:,idx) = expm(logPKG(:,:,idx) - logm(PKG_mean_ext));
    NIM_rletI(:,:,idx) = expm(logNIM(:,:,idx) - logm(NIM_mean_ext));
end

%%
%parallel transport (à la Yair et al., 2019) -> ptmw (first proper pt, then mw)

%proper parallel transport to \overline{\Sigma}

KKI_pt=zeros(190,190,40);
NYU_pt=zeros(190,190,40);
PKG_pt=zeros(190,190,40);
NIM_pt=zeros(190,190,40);

for idx=1:40
    KKI_pt(:,:,idx)=((global_mean_int*(inv(KK_mean_int)))^(1/2))*(KKI(:,:,idx))*(((global_mean_int*(inv(KKI_mean_int)))^(1/2))');
    NYU_pt(:,:,idx)=((global_mean_int*(inv(NYU_mean_int)))^(1/2))*(NYU(:,:,idx))*(((global_mean_int*(inv(NYU_mean_int)))^(1/2))');
    PKG_pt(:,:,idx)=((global_mean_int*(inv(PKG_mean_int)))^(1/2))*(PKG(:,:,idx))*(((global_mean_int*(inv(PKG_meanint)))^(1/2))');
    NIM_pt(:,:,idx)=((global_mean_int*(inv(NIM_mean_int)))^(1/2))*(NIM(:,:,idx))*(((global_mean_int*(inv(NIM_mean_int)))^(1/2))');
end

%last mw step

KKI_ptmw=zeros(190,190,40);
NYU_ptmw=zeros(190,190,40);
PKG_ptmw=zeros(190,190,40);
NIM_ptmw=zeros(190,190,40);

for idx=1:2*N
    KKI_ptmw(:,:,idx)=((global_mean_int)^(-1/2))*KKI_pt(:,:,idx)*((global_mean_int)^(-1/2));
    NYU_ptmw(:,:,idx)=((global_mean_int)^(-1/2))*NYU_pt(:,:,idx)*((global_mean_int)^(-1/2));
    PKG_ptmw(:,:,idx)=((global_mean_int)^(-1/2))*PKG_pt(:,:,idx)*((global_mean_int)^(-1/2));
    NIM_ptmw(:,:,idx)=((global_mean_int)^(-1/2))*NIM_pt(:,:,idx)*((global_mean_int)^(-1/2));
end

%%
%pair-wise LERM (extrinsic)/AIRM (intrinsic) subject distances

all=cat(3,KKI,NYU,PKG,NIM);
%to compute distances for other matrices, concatenate like above, e.g,
%all=cat(3,KKI_rlet,NYU_rlet,PKG_rlet,NIM_rlet);

dist_orig_int=zeros(160,160);
dist_orig_ext=zeros(160,160);

for idx1 = 1:160
    for idx2 = 1:160
        if idx2 > idx1
            distext = spdaux_dist(all(:,:,idx1),all(:,:,idx2),"extrinsic");
            dist_orig_ext(idx1,idx2)=distext;
            dist_orig_ext(idx2,idx1)=distext;
            distin = spdaux_dist(all(:,:,idx1),all(:,:,idx2),"intrinsic");
            dist_orig_int(idx1,idx2)=distint;
            dist_orig_int(idx2,idx1)=distint;
        end
        if idx2 == idx1
            dist=0;
            dist_orig_ext(idx1,idx2)=dist;
            dist_orig_int(idx1,idx2)=dist;
        end
    end
end

%%
%recompute means after transformations
%for rletI and mw, they should all be I=eye(190)

[KKI_mw_mean, ] = spd_mean(spd_initialize(KKI_mw,thr), 1e-8, "intrinsic");
[NYU_mw_mean, ] = spd_mean(spd_initialize(NYU_mw,thr), 1e-8, "intrinsic");
[PKG_mw_mean, ] = spd_mean(spd_initialize(PKG_mw,thr), 1e-8, "intrinsic");
[NIM_mw_mean, ] = spd_mean(spd_initialize(NIM_mw,thr), 1e-8, "intrinsic");

[KKI_ptmw_mean, ] = spd_mean(spd_initialize(KKI_ptmw,thr), 1e-8, "intrinsic");
[NYU_ptmw_mean, ] = spd_mean(spd_initialize(NYU_ptmw,thr), 1e-8, "intrinsic");
[PKG_ptmw_mean, ] = spd_mean(spd_initialize(PKG_ptmw,thr), 1e-8, "intrinsic");
[NIM_ptmw_mean, ] = spd_mean(spd_initialize(NIM_ptmw,thr), 1e-8, "intrinsic");

[KKI_rletI_mean, ] = spd_mean(spd_initialize(KKI_rletI,thr), 1e-8, "extrinsic");
[NYU_rletI_mean, ] = spd_mean(spd_initialize(NYU_rletI,thr), 1e-8, "extrinsic");
[PKG_rletI_mean, ] = spd_mean(spd_initialize(PKG_rletI,thr), 1e-8, "extrinsic");
[NIM_rletI_mean, ] = spd_mean(spd_initialize(NIM_rletI,thr), 1e-8, "extrinsic");

%for rlet, they should all be \overline{\Sigma}=global_mean_ext

[KKI_rlet_mean, ] = spd_mean(spd_initialize(KKI_rlet,thr), 1e-8, "extrinsic");
[NYU_rlet_mean, ] = spd_mean(spd_initialize(NYU_rlet,thr), 1e-8, "extrinsic");
[PKG_rlet_mean, ] = spd_mean(spd_initialize(PKG_rlet,thr), 1e-8, "extrinsic");
[NIM_rlet_mean, ] = spd_mean(spd_initialize(NIM_rlet,thr), 1e-8, "extrinsic");

%%
%compute commutators to show mw \approx ptmw

commKKI = global_mean_int/KKI_mean_int - KKI_mean_int\global_mean_int;
commNYU = global_mean_int/NYU_mean_int - NYU_mean_int\global_mean_int;
commPKG = global_mean_int/PKG_mean_int - PKG_mean_int\global_mean_int;
commNIM = global_mean_int/NIM_mean_int - NIM_mean_int\global_mean_int;

%%
%100 permutation experiments

healthy_orig = cat(3,KKI(:,:,1:20),NYU(:,:,1:20),PKG(:,:,1:20),NIM(:,:,1:20));
adhd_orig = cat(KKI(:,:,21:40),NYU(:,:,21:40),PKG(:,:,21:40),NIM(:,:,21:40));

healthy_mw = cat(3,KKI_mw(:,:,1:20),NYU_mw(:,:,1:20),PKG_mw(:,:,1:20),NIM_mw(:,:,1:20));
adhd_mw = cat(KKI_mw(:,:,21:40),NYU_mw(:,:,21:40),PKG_mw(:,:,21:40),NIM_mw(:,:,21:40));

healthy_rlet = cat(3,KKI_rlet(:,:,1:20),NYU_rlet(:,:,1:20),PKG_rlet(:,:,1:20),NIM_rlet(:,:,1:20));
adhd_rlet = cat(KKI_rlet(:,:,21:40),NYU_rlet(:,:,21:40),PKG_rlet(:,:,21:40),NIM_rlet(:,:,21:40));

healthy_rletI = cat(3,KKI_rletI(:,:,1:20),NYU_rletI(:,:,1:20),PKG_rletI(:,:,1:20),NIM_rletI(:,:,1:20));
adhd_rletI = cat(KKI_rletI(:,:,21:40),NYU_rletI(:,:,21:40),PKG_rletI(:,:,21:40),NIM_rletI(:,:,21:40));

pval_orig = zeros(190,190,100);
pval_mw = zeros(190,190,100);
pval_rlet = zeros(190,190,100);
pval_rletI = zeros(190,190,100);

for i=1:100
    
    rand1 = randperm(80);
    rand2 = randperm(80);
    idx_hc = rand1(1:10);
    idx_adhd = rand2(1:10);
    
    pooled_hc_orig = healthy_orig(:,:,idx_hc); 
    pooled_adhd_orig = adhd_orig(:,:,idx_adhd);
    pooled_hc_mw = healthy_mw(:,:,idx_hc);
    pooled_adhd_mw = adhd_mw(:,:,idx_adhd);
    pooled_hc_rlet = healthy_rlet(:,:,idx_hc);
    pooled_adhd_rlet = adhd_rlet(:,:,idx_adhd);
    pooled_hc_rletI = healthy_rletI(:,:,idx_hc);
    pooled_adhd_rletI = adhd_rletI(:,:,idx_adhd);
    
    pooled_hc_orig_in = spd_initialize(pooled_hc_orig,thr);
    pooled_adhd_orig_in = spd_initialize(pooled_adhd_orig,thr);
    pooled_hc_mw_in = spd_initialize(pooled_hc_mw,thr);
    pooled_adhd_mw_in = spd_initialize(pooled_adhd_orig,thr);
    pooled_hc_rlet_in = spd_initialize(pooled_hc_rlet,thr);
    pooled_adhd_rlet_in = spd_initialize(pooled_adhd_rlet,thr);
    pooled_hc_rletI_in = spd_initialize(pooled_hc_rletI,thr);
    pooled_adhd_rletI_in = spd_initialize(pooled_adhd_rletI,thr);
    
    
    nperms=1000;
    pval_orig(:,:,i) = spd_eqtestelem(pooled_adhd_orig_in,pooled_hc_orig_in,nperms);
    pval_mw(:,:,i) = spd_eqtestelem(pooled_adhd_mw_in,pooled_hc_mw_in,nperms);
    pval_rlet(:,:,i) = spd_eqtestelem(pooled_adhd_rlet_in,pooled_hc_rlet_in,nperms);
    pval_rletI(:,:,i) = spd_eqtestelem(pooled_adhd_rletI_in,pooled_hc_rletI_in,nperms);
    
end

%%
%threshold pvals and binarize

thrpval=0.001;

pval_orig_thr = zeros(190,190,100);
pval_mw_thr = zeros(190,190,100);
pval_rlet_thr = zeros(190,190,100);
pval_rletI_thr = zeros(190,190,100);

for idx1=1:190
    for idx2 = 1:190
        for idx3 = 1:100
            if pval_orig(idx1,idx2,idx3) < thrpval
                pval_orig_thr(idx1,idx2,idx3)=1;
            else
                pval_orig_thr(idx1,idx2,idx3)=0;
            end
            
            if pval_mw(idx1,idx2,idx3) < thrpval
                pval_mw_thr(idx1,idx2,idx3)=1;
            else
                pval_mw_thr(idx1,idx2,idx3)=0;
            end
            
            if pval_rlet(idx1,idx2,idx3) < thrpval
                pval_rlet_thr(idx1,idx2,idx3)=1;
            else
                pval_rlet_thr(idx1,idx2,idx3)=0;
            end
            
            if pval_rletI(idx1,idx2,idx3) < thrpval
                pval_rletI_thr(idx1,idx2,idx3)=1;
            else
                pval_rletI_thr(idx1,idx2,idx3)=0;
            end
        end
    end
end

%create frequency matrices F_{ij}

F_orig = sum(pval_orig_thr,3);
F_mw = sum(pval_mw_thr,3);
F_rlet = sum(pval_rlet_thr,3);
F_rletI = sum(pval_rletI_thr,3);
