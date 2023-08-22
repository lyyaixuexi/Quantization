import os

label_care = ['zlight_g_left',
              'zlight_g_right',
              'zlight_g_round',
              'zlight_g_straight',
              'zlight_r_left',
              'zlight_r_right',
              'zlight_r_round',
              'zlight_r_straight',
              'zlight_y_left',
              'zlight_y_right',
              'zlight_y_round',
              'zlight_y_straight']

# Get the all sub folders in the "path" folder
def get_sub_folders(path, sub_folder):

    if not type(path) == type(list()):
        path = [path]

    for each_path in path :
        current_files = os.listdir(each_path)
        for file_name in current_files:
            full_file_name = os.path.join(each_path, file_name)
            if os.path.isdir(full_file_name):
                get_sub_folders(full_file_name, sub_folder)
            else:
                return sub_folder.append(os.path.dirname(full_file_name))

data_path = '/data2/myxu/tsr-classify-training/home.bak/nv3070/ljj_project/DATA/'
save_path = '/data2/myxu/tsr_tlr_dataset/tsr_data_for_tlwr/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

sub_folders = []
get_sub_folders(data_path, sub_folders)

needed_folders = []
for folder in sub_folders:
    for label in label_care:
        if label in folder:
            needed_folders.append(folder)
            break

fail_dir_path = []
for class_path in needed_folders:
    cmd_tsr = 'cp -r ' + class_path + ' ' + save_path
    retval = os.system(cmd_tsr)
    if retval != 0:
        if class_path not in fail_dir_path:
            fail_dir_path.append(class_path)

subfolders = os.listdir(save_path)
for folder in subfolders:
    imgs = os.listdir(save_path + folder)
    for img in imgs:
        if 'YUV_bt601V' in img:
            os.remove(os.path.join(save_path + folder, img))

# cmd_str = 'cd ' + save_path[:28] + ' && tar -cvzf tsr_data_for_tlwr.tar.gz tsr_data_for_tlwr/'
# retval = os.system(cmd_tsr)
# if retval != 0:
#     print(cmd_str)