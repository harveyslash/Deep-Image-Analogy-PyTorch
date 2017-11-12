# Append 'src' directory to import modules from notebooks directory
#################################
import os,sys
src_dir = os.path.join(os.getcwd(), os.pardir)
sys.path.append(src_dir)
##################################


import time
import click
import os 
import torchvision.models as models
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.model_zoo as model_zoo
import cv2
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from collections import OrderedDict
from PIL import Image,ImageOps
import torch.optim as optim 
from src.models import VGG19
from src import Utils
from src.PatchMatch import PatchMatchOrig
import matplotlib.pyplot as plt


import scipy

def blend_features(feat_a, recon_feat_bb, alpha=0.8, tau=0.05, k=300.):
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def clip_at_threshold(neuron_mag_a):
        if True:
            neuron_mag_a[neuron_mag_a < .01] = 0
        return neuron_mag_a

    norm_feat_a = feat_a**2
    neuron_mag_a = (norm_feat_a - norm_feat_a.min(axis=(0,1,2),keepdims=True))/(norm_feat_a.max(axis=(0,1,2),keepdims=True)-norm_feat_a.min(axis=(0,1,2),keepdims=True))
    neuron_mag_a = clip_at_threshold(neuron_mag_a)
    neuron_mag_a = sigmoid(k*(neuron_mag_a - tau))

    weight_a = alpha*neuron_mag_a
    feat_aa = np.multiply(feat_a, weight_a) + np.multiply(recon_feat_bb, (1. - weight_a))
    return feat_aa

def normalize_feat_map(feat_map):
    """
    Normalize the feature map along the channels dimension

    feat_map is a numpy array with channels along the 2nd dimension
    """
    return feat_map/np.linalg.norm(feat_map,ord=2,axis=(2),keepdims=True)



@click.command()
@click.argument('imga_path', type=click.Path(exists=True))
@click.argument('imgbb_path', type=click.Path(exists=True))
@click.argument('out')
def main(imga_path,imgbb_path,out):
    c_feat_ids = [29,20,11,6,1]
    c_alphas = [.8,.7,.6,.1]
    c_patch_sizes = [3,3,3,5,5]
    c_patch_radii = [500,6,6,4,4]

    c_iters = [700,700,700,700]

    model = VGG19()

    imga = Utils.load_image(img_path=imga_path,to_array=True,to_variable=True)
    imgbb = Utils.load_image(img_path=imgbb_path,to_array=True,to_variable=True)

    feata = model.get_features(img_tensor=imga)
    featbb = model.get_features(img_tensor=imgbb)

    ###### Process L 5 ######
    feat5a = feata[c_feat_ids[0]]
    feat5bb = featbb[c_feat_ids[0]]

    feat5a_norm = normalize_feat_map(feat5a)
    feat5bb_norm = normalize_feat_map(feat5bb)

    pm5ab = PatchMatchOrig(feat5a_norm,feat5a_norm,feat5bb_norm,feat5bb_norm, c_patch_sizes[0])
    pm5ab.propagate(iters=5,rand_search_radius=c_patch_radii[0])

    imga_raw = Utils.load_image(img_path=imga_path,to_array=False,to_variable=False).numpy().transpose(1,2,0)
    imgbb_raw = Utils.load_image(img_path=imgbb_path,to_array=False,to_variable=False).numpy().transpose(1,2,0)
    imga_raw.shape

    recon = pm5ab.reconstruct_image(imga_raw)

    pm5ba = PatchMatchOrig(feat5bb_norm,feat5bb_norm,feat5a_norm,feat5a_norm,c_patch_sizes[0])
    pm5ba.propagate(iters=5,rand_search_radius=c_patch_radii[0])
    recon = pm5ba.reconstruct_image(imga_raw)

    warped_feat5bb = pm5ab.reconstruct_image(feat5bb)
    warped_feat5a = pm5ba.reconstruct_image(feat5a)
    ###### Block L 5 Done ######


    ###### Process Block 4 ######
    r4_bb = model.get_deconvoluted_feat(warped_feat5bb,5,iters=c_iters[0])
    r4_a = model.get_deconvoluted_feat(warped_feat5a,5,iters=c_iters[0])

    feat4a = feata[c_feat_ids[1]]
    feat4bb = featbb[c_feat_ids[1]]

    feat4aa = blend_features(feat4a,r4_bb,alpha=c_alphas[0])
    feat4b =  blend_features(feat4bb,r4_a,alpha=c_alphas[0])

    feat4a_norm = normalize_feat_map(feat4a)
    feat4bb_norm = normalize_feat_map(feat4bb)

    feat4aa_norm = normalize_feat_map(feat4aa)
    feat4b_norm = normalize_feat_map(feat4b)


    pm4ab = PatchMatchOrig(feat4a_norm,feat4aa_norm,feat4b_norm,feat4bb_norm, c_patch_sizes[1])
    pm4ab.nnf = pm5ab.upsample_nnf(size=28)

    pm4ab.propagate(iters=5,rand_search_radius=c_patch_radii[1])


    pm4ba = PatchMatchOrig(feat4bb_norm,feat4b_norm,feat4aa_norm,feat4a_norm, c_patch_sizes[1])
    pm4ba.nnf = pm5ba.upsample_nnf(size=28)

    pm4ba.propagate(iters=5,rand_search_radius=c_patch_radii[1])

    warped_feat4bb = pm4ab.reconstruct_image(feat4bb)
    warped_feat4a = pm4ba.reconstruct_image(feat4a)
    ###### Block 4 done ######


    ###### Process Block 3 ######


    r3_bb = model.get_deconvoluted_feat(warped_feat4bb,4,iters=c_iters[1])
    r3_a = model.get_deconvoluted_feat(warped_feat4a,4,iters=c_iters[1])

    feat3a = feata[c_feat_ids[2]]
    feat3bb = featbb[c_feat_ids[2]]

    feat3aa = blend_features(feat3a,r3_bb,alpha=c_alphas[1])
    feat3b =  blend_features(feat3bb,r3_a,alpha=c_alphas[1])

    feat3a_norm = normalize_feat_map(feat3a)
    feat3bb_norm = normalize_feat_map(feat3bb)

    feat3aa_norm = normalize_feat_map(feat3aa)
    feat3b_norm = normalize_feat_map(feat3b)

    pm3ab = PatchMatchOrig(feat3a_norm,feat3aa_norm,feat3b_norm,feat3bb_norm, c_patch_sizes[2])
    pm3ab.nnf = pm4ab.upsample_nnf(size=56)

    pm3ab.propagate(iters=5,rand_search_radius=c_patch_radii[2])

    pm3ba = PatchMatchOrig(feat3bb_norm,feat3b_norm,feat3aa_norm,feat3a_norm, c_patch_sizes[2])
    pm3ba.nnf = pm4ba.upsample_nnf(size=56)

    pm3ba.propagate(iters=5,rand_search_radius=c_patch_radii[2])

    warped_feat3bb = pm3ab.reconstruct_image(feat3bb)
    warped_feat3a = pm3ba.reconstruct_image(feat3a)

    ###### Block 3 done ######

    ###### Process Block 2 ######

    r2_bb = model.get_deconvoluted_feat(warped_feat3bb,3,iters=c_iters[2])
    r2_a = model.get_deconvoluted_feat(warped_feat3a,3,iters=c_iters[2])

    feat2a = feata[c_feat_ids[3]]
    feat2bb = featbb[c_feat_ids[3]]

    feat2aa = blend_features(feat2a,r2_bb,alpha=c_alphas[2])
    feat2b = blend_features(feat2bb,r2_a,alpha=c_alphas[2])

    feat2a_norm = normalize_feat_map(feat2a)
    feat2bb_norm = normalize_feat_map(feat2bb)

    feat2aa_norm = normalize_feat_map(feat2aa)
    feat2b_norm = normalize_feat_map(feat2b)

    pm2ab = PatchMatchOrig(feat2a_norm,feat2aa_norm,feat2b_norm,feat2bb_norm, c_patch_sizes[3])
    pm2ab.nnf = pm3ab.upsample_nnf(size=112)

    pm2ab.propagate(iters=5,rand_search_radius=c_patch_radii[3])


    pm2ba = PatchMatchOrig(feat2bb_norm,feat2b_norm,feat2aa_norm,feat2a_norm, c_patch_sizes[3])
    pm2ba.nnf = pm3ba.upsample_nnf(size=112)

    pm2ba.propagate(iters=5,rand_search_radius=c_patch_radii[3])


    warped_feat2bb = pm2ab.reconstruct_image(feat2bb)
    warped_feat2a = pm2ba.reconstruct_image(feat2a)

    ###### Block 2 done ######

    ###### Process Block 1 ######

    r1_bb = model.get_deconvoluted_feat(warped_feat2bb,2,iters=c_iters[3])
    r1_a = model.get_deconvoluted_feat(warped_feat2a,2,iters=c_iters[3])

    feat1a = feata[c_feat_ids[4]]
    feat1bb = featbb[c_feat_ids[4]]

    feat1aa = blend_features(feat1a,r1_bb,alpha=c_alphas[3])
    feat1b = blend_features(feat1bb,r1_a,alpha=c_alphas[3])

    feat1a_norm = normalize_feat_map(feat1a)
    feat1bb_norm = normalize_feat_map(feat1bb)

    feat1aa_norm = normalize_feat_map(feat1aa)
    feat1b_norm = normalize_feat_map(feat1b)

    pm1ab = PatchMatchOrig(feat1a_norm,feat1aa_norm,feat1b_norm,feat1bb_norm, c_patch_sizes[4])
    pm1ab.nnf = pm2ab.upsample_nnf(size=224)

    pm1ab.propagate(iters=5,rand_search_radius=c_patch_radii[4])

    plt.axis('off')
    # plt.imshow(Utils.deprocess_image(Utils.reconstruct_image(img_a=imgbb_raw,pm=pm1ab)))
    # plt.show()

    ups = pm1ab.upsample_nnf(size=224)
    plt.axis('off')
    resB = np.clip(Utils.deprocess_image(pm1ab.reconstruct_avg(imgbb_raw,patch_size=2)),0,1)

    # plt.imshow(resB)
    plt.imsave(out,resB)
    # plt.show()

if __name__ == '__main__':
    main()
