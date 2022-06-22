dataset_type = 'WaltDataset'
data_root = 'data/cwalt_train/'
data_root_test = 'data/cwalt_test/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/',
        img_prefix=data_root + '/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root_test + '/',
        img_prefix=data_root_test + '/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root_test + '/',
        img_prefix=data_root_test + '/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
