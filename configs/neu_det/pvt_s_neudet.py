_base_ = 'pvt_t_neudet.py'
load_from = '../checkpoints/pvt_small.pth'
model = dict(
    backbone=dict(
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(checkpoint=load_from)))
