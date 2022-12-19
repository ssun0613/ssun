import numpy as np
import glob


# data = np.load('/storage/mskim/WDM/Wafer_Map_Datasets.npz')
# label = data['arr_1']
#
# for i in range(label.shape[0]):
#     a = np.where(label[i])
#     if a[0].shape[0] == 0 or a[0].shape[0] ==1:
#         np.savez('/storage/mskim/WDM/single_data/single_label/' +str(i), label[i])


# data_image = sorted(glob.glob('/storage/mskim/WDM/single_data/single_image/*.npz'))
# data_label = sorted(glob.glob('/storage/mskim/WDM/single_data/single_label/*.npz'))
#
# for i in range(5000,5135,1):
#     out_image = np.load(data_image[i])
#     out_label = np.load(data_label[i])
#     print(out_label['arr_0'])
#     np.savez('/storage/mskim/WDM/single_data/nearfull/' + str('R') + str(i-4133), out_image['arr_0'])
#     np.savez('/storage/mskim/WDM/single_data/nearfull_label/' + str('R') + str(i-4133), out_label['arr_0'])


# data_image = sorted(glob.glob('/storage/mskim/WDM/single_data_1/single_image/*.npz'))
data_image = sorted(glob.glob('/storage/mskim/WDM/single_data_1/train/image/*.npz'))
data_label = sorted(glob.glob('/storage/mskim/WDM/single_data_1/single_label/*.npz'))

for i in range(7015,8015,1):
    out_image = np.load(data_image[i])
    out_label = np.load(data_label[i])
    print(out_label['arr_0'])
    if i < 7815:
        np.savez('/storage/mskim/WDM/single_data_1/train/image/' + str('S') + str(i-7014), out_image['arr_0'])
        np.savez('/storage/mskim/WDM/single_data_1/train/label/' + str('S') + str(i-7014), out_label['arr_0'])
    elif i >= 7815:
        np.savez('/storage/mskim/WDM/single_data_1/test/image/' + str('S') + str(i-7814), out_image['arr_0'])
        np.savez('/storage/mskim/WDM/single_data_1/test/label/' + str('S') + str(i-7814), out_label['arr_0'])



