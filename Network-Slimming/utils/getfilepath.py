import sqlite3


##----用于拿到数据库所有的被称为可用数据的路径----#

def getDate1(database,datasheet):
    conn=sqlite3.connect(database)
    result = conn.execute('''SELECT * FROM {} WHERE 是否可用 = '{}'
                            '''.format(datasheet,True))
    output =  result.fetchall()
    conn.close()
    return output


def getDate2(database,datasheet,which_camera,path_like):
    conn=sqlite3.connect(database)
    result = conn.execute('''SELECT * FROM {} WHERE 是否可用 = '{}' AND 
                            WHICH_VIDEO = '{}' AND 
                            PATH LIKE "%{}%";
                            '''.format(datasheet,True,which_camera,path_like))
    output =  result.fetchall()
    conn.close()
    return output

def getdata_by_videoid(database,datasheet,video_id_list):
    '''
    输入video_id_list 
    输出所有相关视频词条
    '''
    conn=sqlite3.connect(database)
 
    if not type(video_id_list) == list:
        video_id_list = [video_id_list]

    output = []

    for video_id in video_id_list:
        total_size,md5_1000b = video_id.split('#')
        print(datasheet,True,total_size,md5_1000b)
        result = conn.execute('''SELECT * FROM {} WHERE 是否可用 = '{}' AND 
                                TOTAL_SIZE = {} AND 
                                MD5_1000B = '{}';
                                '''.format(datasheet,True,total_size,md5_1000b))
        temp =  result.fetchall()[0]
        output.append(temp)
    conn.close()
    return output

if __name__ == '__main__':
    database_path = "/mnt/nas/深度学习组/lvjunjie_datasets/数据库汇总/nas视频数据库.db"
    datasheet_name = "MDC视频数据库"
    which_camera = "front30"
    path_like = '00_tongxiang_nas'

    video_id = ['24174865#bf3218fe609d8e65f923e99175c06145','24174865#bf3218fe609d8e65f923e99175c06145']
    file_list = getdata_by_videoid(database_path, datasheet_name,video_id)

    print(file_list)

    for i in file_list:
        print(i)
    print(len(file_list))