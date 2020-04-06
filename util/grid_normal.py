import random
import numpy as np


def grid_mean_3d(image, grid_size=[4,4,4], norm="mean"):
    w, h, d = image.shape
    grid_x, grid_y, grid_z = grid_size
    assert w % grid_x == 0
    assert h % grid_y == 0
    assert d % grid_z == 0
    round1 = w // grid_x
    round2 = h // grid_y
    round3 = d // grid_z
    all_grid_patchs = []
    for indx in range(round1):
        for indy in range(round2):
            for indz in range(round3):
                patch = image[indx*grid_x:(indx+1)*grid_x, indy *
                              grid_y:(indy+1)*grid_y, indz*grid_z:(indz+1)*grid_z]
                all_grid_patchs.append(patch)

    num = 0
    grid_mean_image = np.zeros((w, h, d), np.float32)
    for indx in range(round1):
        for indy in range(round2):
            for indz in range(round3):
                if norm == "mean":
                    grid_mean_image[indx*grid_x:(indx+1)*grid_x, indy*grid_y:(
                        indy+1)*grid_y, indz*grid_z:(indz+1)*grid_z] = all_grid_patchs[num].mean()
                if norm == "max":
                    grid_mean_image[indx*grid_x:(indx+1)*grid_x, indy*grid_y:(
                        indy+1)*grid_y, indz*grid_z:(indz+1)*grid_z] = all_grid_patchs[num].max()
                if norm == "min":
                    grid_mean_image[indx*grid_x:(indx+1)*grid_x, indy*grid_y:(
                        indy+1)*grid_y, indz*grid_z:(indz+1)*grid_z] = all_grid_patchs[num].min()
                if norm == "random":
                    patch_value = [all_grid_patchs[num].min(
                    ), all_grid_patchs[num].max(), all_grid_patchs[num].mean()]
                    grid_mean_image[indx*grid_x:(indx+1)*grid_x, indy*grid_y:(
                        indy+1)*grid_y, indz*grid_z:(indz+1)*grid_z] = patch_value[random.randint(0, 2)]
                num += 1
    return grid_mean_image
