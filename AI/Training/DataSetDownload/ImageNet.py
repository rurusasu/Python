from ImageNet_Downloader import downloader
import os

root_dir = os.getcwd() + r'\DataSet'
#print (root_dir)
wnid = 'n03388183'
api = downloader.ImageNet(root_dir)
api.download(wnid, verbose=True)