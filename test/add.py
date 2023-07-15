def openreadtxt(file_name):
    data = []
    file = open(file_name, 'r')  # 打开文件
    file_data = file.readlines()  # 读取所有行
    for row in file_data:
        tmp_list = row.split(' ')  # 按‘，’切分每行的数据
        tmp_list[-1] = tmp_list[-1].replace('\n', '') #去掉换行符
        data.append(tmp_list)  # 将每行数据插入data中
    return data


if __name__ == "__main__":
    data = openreadtxt('/mnt/cephfs/home/lyy/Quantization/test/test.txt')
    total = 0
    num = 0
    for i in data:
        total += float(i[0])
        print(float(i[0]))
        num += 1
    print(total)
    print(num)
    print(total/num)
