# -*- coding:utf-8 -*-

class data(object):
    def __init__(self,datas):
        self.datas = datas
    def printdata(self):
        print(self.datas)

library = [["a", 0], ["b", 1], ["z", 25]]
# dic = {u"姓名": "", u"年龄": 0}   this can't be here, should be in the iteration
lst = []
for info in library:
    dic = {u"姓名": "", u"年龄": 0}
    dic[u"姓名"] = info[0]
    dic[u"年龄"] = info[1]
    lst.append(data(dic))

print lst[1].printdata()