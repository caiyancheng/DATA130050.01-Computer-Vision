
cfg = dict(
    model_type='bisenetv1',
    num_aux_heads=2,
    lr_start=1e-2,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=80000,
    im_root='/remote-home/source/Cityscapes/',
    train_im_anns='/root/caiyancheng/edll-242/BisenetV2/datasets/cityscapes/foggy/train.txt',#/intru
    val_im_anns='/root/caiyancheng/edll-242/BisenetV2/datasets/cityscapes/foggy/val.txt',
    scales=[0.75, 2.],
    cropsize=[1024,1024],
    ims_per_gpu=8,
    use_fp16=True,
    use_sync_bn=False,
    respth='/root/caiyancheng/edll-242/BisenetV2/tools/res',
    num_classes=2,
)
