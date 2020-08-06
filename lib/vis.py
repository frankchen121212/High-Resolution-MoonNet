import matplotlib.pyplot as plt
import os
import numpy as np
from utils.template_match_target import template_match_t
from PIL import Image

def debug_prediction(data, model, debug_freq, debug_path):
    # IMG.shape :(val_num, 256, 256, 1)
    # GT.shape  :(val_num, 256, 256)
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)

    IMG, GT = data[0], data[1]
    val_num = IMG.shape[0]
    for i_v in range(val_num):
        if i_v% (debug_freq) == 0 :
            PRED = model.predict(IMG[i_v][np.newaxis,:,:,:])
            pred_rings = template_match_t(PRED[0].copy(), minrad=2.)
            gt_rings = template_match_t(GT[i_v], minrad=2.)
            fig = plt.figure(figsize=[16, 16])
            [[ax1, ax2], [ax3, ax4]] = fig.subplots(2, 2)
            ax1.imshow(IMG[i_v].squeeze(), origin='upper', cmap='Greys_r', vmin=0, vmax=1.1)
            ax2.imshow(GT[i_v].squeeze(), origin='upper', cmap='Greys_r')
            ax3.imshow(PRED[0], origin='upper', cmap='Greys_r', vmin=0, vmax=1)
            ax4.imshow(IMG[i_v].squeeze(), origin='upper', cmap="Greys_r")

            # Draw prediction/gt rings
            for x, y, r in pred_rings:
                circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=2, alpha=0.5)
                ax4.add_artist(circle)
            for x, y, r in gt_rings:
                circle = plt.Circle((x, y), r, color='blue', fill=False, linewidth=2, alpha=0.5)
                ax4.add_artist(circle)

            ax1.set_title('Moon DEM Image')
            ax2.set_title('Ground-Truth Target Mask')
            ax3.set_title('Prediction Masks')
            ax4.set_title('Ground Truth (Blue) Predictions (Red)')
            img_name = os.path.join(debug_path,"img_{}.png".format(i_v))
            print('=> saving {}'.format(img_name))
            plt.savefig(img_name)
            plt.close('all')


def get_heatmap(data, model, debug_freq, heatmap_path):
    pass

