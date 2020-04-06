import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure,draw
CTpath = '/lyc/abdomen/data_zoom/test/btcv01/'
file_wanted = ['imgzoom.npy', 'labelzoom.npy',
               'Seg.npy', 'FSDis_combine.npy', 'ManualDis.npy']

# img = np.load(CTpath+file_wanted[0])
# label = np.load(CTpath+file_wanted[1])
# Seg = np.load(CTpath+file_wanted[2])
FSDis = np.float64(np.load(CTpath+file_wanted[3]))
label = np.load(CTpath+file_wanted[1])
plt.figure(dpi=800)
for i in np.arange(1, FSDis[0].shape[0]):
    f, plots = plt.subplots(2, 4, figsize=[60, 60])
    plots[0, 0].imshow(FSDis[0][i], cmap='gray')
    plots[0, 0].set_title('image{0:}'.format(i))
    plots[0, 1].imshow(FSDis[1][i], cmap='gray')
    plots[0, 2].imshow(label[i], cmap='gray')
    plots[0, 2].set_title('lab')
    plots[0, 3].imshow(FSDis[7][i], cmap='gray')
    plots[1, 0].imshow(FSDis[8][i], cmap='gray')
    plots[1, 1].imshow(FSDis[9][i], cmap='gray')
    plots[1, 2].imshow(FSDis[10][i], cmap='gray')
    plots[1, 3].imshow(FSDis[11][i], cmap='gray')
    plt.show()