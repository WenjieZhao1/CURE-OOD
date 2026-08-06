import torch
from torch import nn
from torchmtlr import MTLR

n_clin_var = 11

def conv_3d_block (in_c, out_c, act='relu', norm='bn', num_groups=8, *args, **kwargs):
    activations = nn.ModuleDict ([
        ['relu', nn.ReLU(inplace=True)],
        ['lrelu', nn.LeakyReLU(0.1, inplace=True)]
    ])
    
    normalizations = nn.ModuleDict ([
        ['bn', nn.BatchNorm3d(out_c)],
        ['gn', nn.GroupNorm(int(out_c/num_groups), out_c)]
    ])
    
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, *args, **kwargs),
        normalizations[norm],
        activations[act],
    )

def flatten_layers(arr):
    return [i for sub in arr for i in sub]



class Cli_MTLR(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        if hparams['n_dense'] <=0:
            
            self.fc_layers1 = nn.Dropout(hparams['dropout']) 
            
            self.mtlr1 = MTLR(n_clin_var, hparams['time_bins'])
            self.mtlr2 = MTLR(n_clin_var, hparams['time_bins'])
            self.mtlr3 = MTLR(n_clin_var, hparams['time_bins'])
            self.mtlr4 = MTLR(n_clin_var, hparams['time_bins'])

        elif hparams['n_dense'] ==1:
            self.fc_layers1 = nn.Sequential(nn.Linear(n_clin_var, 64*hparams['dense_factor']), 
                          nn.BatchNorm1d(64*hparams['dense_factor']),
                          nn.ReLU(inplace=True), 
                          nn.Dropout(hparams['dropout']))
            
            self.mtlr1 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr2 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr3 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr4 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            
        elif hparams['n_dense'] > 1:    
            self.fc_layers1 = nn.Sequential(nn.Linear(n_clin_var , 128*hparams['dense_factor']), 
                          nn.BatchNorm1d(128*hparams['dense_factor']),
                          nn.ReLU(inplace=True), 
                          nn.Dropout(hparams['dropout']),
                          nn.Linear(128*hparams['dense_factor'] , 64*hparams['dense_factor']), 
                          nn.BatchNorm1d(64*hparams['dense_factor']),
                          nn.ReLU(inplace=True), 
                          nn.Dropout(hparams['dropout']))
            
            self.mtlr1 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr2 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr3 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr4 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])

    def forward(self, x):
        
        sample_img, clin_var = x

        x = self.fc_layers1(clin_var)

        risk_out1, _ = self.mtlr1(x)
        risk_out2, _ = self.mtlr2(x)
        risk_out3, _ = self.mtlr3(x)
        risk_out4, _ = self.mtlr4(x)
        
        return risk_out1,  risk_out2, risk_out3, risk_out4


"""
Inspired from the work of
Credits:
@article{
  kim_deep-cr_2020,
	title = {Deep-{CR} {MTLR}: a {Multi}-{Modal} {Approach} for {Cancer} {Survival} {Prediction} with {Competing} {Risks}},
	shorttitle = {Deep-{CR} {MTLR}},
	url = {https://arxiv.org/abs/2012.05765v1},
	language = {en},
	urldate = {2021-03-16},
	author = {Kim, Sejin and Kazmierski, Michal and Haibe-Kains, Benjamin},
	month = dec,
	year = {2020}
}
"""
