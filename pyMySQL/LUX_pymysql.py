import sys
from dateutil import parser
sys.path.insert(2, '//global/project/projectdirs/lux/Tools/anaconda/lib/python2.7/site-packages/')
sys.path.insert(2, '//global/project/projectdirs/lux/data/')
import pymysql
import xml2dict
import xml.etree.cElementTree as ET


db=pymysql.connect(host='151.159.226.141',port=3101,user='luxanalysis',passwd='bigdet',db='control')
cur = db.cursor()