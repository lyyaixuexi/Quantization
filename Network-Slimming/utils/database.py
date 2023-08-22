import sqlite3
import time
import datetime
 
class DB:
    def __init__(self,db_file,tablename):
        self.db_file = db_file
        self.Start()
        self.CreatTable(tablename)
        self.tablename = tablename
        #self.Close()

 
    def Start(self,):
        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()
 
    def CreatTable(self,tablename):
        try:
            sql = '''
            CREATE TABLE IF NOT EXISTS {}(
               `uuid` TEXT PRIMARY KEY NOT NULL,
               `imgBase64` TEXT NOT NULL,
               `SourceVideo` TEXT NOT NULL,
               `FrameID` INTEGE NOT NULL,
               `OrginalBox` TEXT NOT NULL,
               `BoxInCrop` TEXT NOT NULL,
               `label` TEXT
            )
            '''.format(tablename)
            self.cursor.execute(sql)
            return 1
        except Exception as e:
            print('>> Creat Error:', e)
            return 0
 
    def Insert(self, uuid, imgBase64, SourceVideo, FrameID ,OrginalBox,BoxInCrop,label = ''):
        try:
            sql = '''
            INSERT INTO {}
            VALUES
            (?, ?, ?, ?, ?, ?, ?);
           '''.format(self.tablename)
            #self.Start()
            self.cursor.execute(sql, ( uuid, imgBase64, SourceVideo, FrameID ,OrginalBox,BoxInCrop,label))
            self.conn.commit()
            #self.Close()
            return 1
        except Exception as e:
            print('>> Insert Error:', e)
            return 0
 
    def Select(self, id):
        self.Start()
        self.cursor.execute('''SELECT * from {} WHERE id=(?);'''.format(self.tablename), (id,))
        res = self.cursor.fetchall()
        self.Close()
        return res
 
    def Close(self):
        self.cursor.close()
        self.conn.close()
 
    def SelectALL(self):
        self.Start()
        sql = "SELECT * from {};".format(self.tablename)
        self.cursor.execute(sql)
        res = self.cursor.fetchall()
        self.Close()
        return res
 
 
if __name__ == '__main__':
    db = DB('test')
    db.Insert('8', '2','3','4','5','6')
    res = db.SelectALL()
    print(res)
