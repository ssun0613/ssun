import sys
sys.path.append("..")

import numpy as np

p_1 = np.load('../p_r_data/p_r_data_capsnet/p_r_data_t_p_capsnet_mse_16_256_20.npy')
p_2 = np.load('../p_r_data/p_r_data_capsnet/p_r_data_t_p_capsnet_mse_8_256_200.npy')
p_3 = np.load('../p_r_data/p_r_data_capsnet/p_r_data_t_p_capsnet_mse_8_256_400.npy')
p_4 = np.load('../p_r_data/p_r_data_capsnet/p_r_data_t_p_capsnet_mse_8_256_600.npy')
p_5 = np.load('../p_r_data/p_r_data_capsnet/p_r_data_t_p_capsnet_mse_8_256_1000.npy')

a = np.append(p_1, p_2, axis=0)
a_1 = np.append(a, p_3, axis=0)
a_2 = np.append(a_1, p_4, axis=0)
a_3 = np.append(a_2, p_5, axis=0)

np.save('../p_r_data/p_r_data_capsnet/p_r_data_t_p_capsnet_mse(8,16,256).npy',a_3)
print('../p_r_data/p_r_data_capsnet/p_r_data_t_p_capsnet_mse.npy')


