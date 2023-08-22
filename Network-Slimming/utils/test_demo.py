import os

save_path = '/data2/myxu/tsr_tlr_dataset/tsr_data_for_tlwr/'

subfolders = os.listdir(save_path)
for folder in subfolders:
    imgs = os.listdir(save_path + folder)
    for img in imgs:
        if 'YUV_bt601V' in img or '.npy' in img:
            os.remove(os.path.join(save_path + folder, img))