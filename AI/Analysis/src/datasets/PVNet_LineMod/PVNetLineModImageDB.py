import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')

import numpy as np

from config.config import cfg
from PVNetLineModModelDB import LineModModelDB
from src.utils.base_utils import read_pickle, Projector


class LineModImageDB(object):
    '''
    rgb_pth relative path to cfg.LINEMOD
    dpt_pth relative path to cfg.LINEMOD
    RT np.float32 [3,4]
    object_typ 'cat' ...
    rnd_typ 'real' or 'render'
    corner  np.float32 [8,2]
    '''
    def __init__(self,
                              object_name,
                              render_num: int = 10000,
                              fuse_num: int = 10000,
                              ms_num: int = 10000,
                              has_render_set=True,
                              has_fuse_set=True):

        self.linemod_dir = cfg.LINEMOD_DIR
        self.pvnet_linemod_dir = cfg.PVNET_LINEMOD_DIR

        self.object_name = object_name
        # some dirs for processing
        os.path.join(self.pvnet_linemod_dir ,
                                   'posedb',
                                   '{}_render.pkl'.format(object_name))
        self.render_dir = 'renders/{}'.format(object_name)
        self.rgb_dir = '{}/JPEGImages'.format(object_name)
        self.mask_dir='{}/mask'.format(object_name)
        self.rt_dir=os.path.join(cfg.DATA_DIR,'LINEMOD_ORIG',object_name,'data')
        self.render_num=render_num

        self.test_fn='{}/test.txt'.format(object_name)
        self.train_fn='{}/train.txt'.format(object_name)
        self.val_fn='{}/val.txt'.format(object_name)

        if has_render_set:
            self.render_pkl=os.path.join(self.linemod_dir,'posedb','{}_render.pkl'.format(object_name))
            # prepare dataset
            if os.path.exists(self.render_pkl):
                # read cached
                self.render_set=read_pickle(self.render_pkl)
            else:
                # process render set
                self.render_set=self.collect_render_set_info(self.render_pkl,self.render_dir)
        else:
            self.render_set=[]

        self.real_pkl=os.path.join(self.linemod_dir,'posedb','{}_real.pkl'.format(object_name))
        if os.path.exists(self.real_pkl):
            # read cached
            self.real_set=read_pickle(self.real_pkl)
        else:
            # process real set
            self.real_set=self.collect_real_set_info()

            # prepare train test split
            self.train_real_set=[]
            self.test_real_set=[]
            self.val_real_set=[]
            self.collect_train_val_test_info()

            self.fuse_set=[]
            self.fuse_dir='fuse'
            self.fuse_num=fuse_num
            self.object_idx=cfg.linemod_object_names.index(object_name)

            if has_fuse_set:
                self.fuse_pkl=os.path.join(cfg.LINEMOD,'posedb','{}_fuse.pkl'.format(object_name))
                # prepare dataset
                if os.path.exists(self.fuse_pkl):
                    # read cached
                    self.fuse_set=read_pickle(self.fuse_pkl)
                else:
                    # process render set
                    self.fuse_set=self.collect_fuse_info()
            else:
                self.fuse_set=[]


    def collect_render_set_info(self,
                                                               pkl_file,
                                                               render_dir,
                                                               format='jpg'):
        database=[]
        projector=Projector()
        modeldb=LineModModelDB()
        for k in range(self.render_num):
            data={}
            data['rgb_pth'] = os.path.join(render_dir, '{}.{}'.format(k, format))
            data['dpt_pth'] = os.path.join(render_dir, '{}_depth.png'.format(k))
            data['RT'] = read_pickle(os.path.join(self.linemod_dir,
                                                                                         render_dir,
                                                                                         '{}_RT.pkl'.format(k)))['RT']
            data['object_typ'] = self.object_name
            data['rnd_typ'] = 'render'
            data['corners'] = projector.project(modeldb.get_corners_3d(self.object_name),
                                                                                   data['RT'],
                                                                                   'blender')
            data['farthest'] = projector.project(modeldb.get_farthest_3d(self.object_name),
                                                                                    data['RT'],
                                                                                    'blender')
            data['center'] = projector.project(modeldb.get_centers_3d(self.object_name)[None,:],
                                                                                 data['RT'],
                                                                                 'blender')
            for num in [4,12,16,20]:
                data['farthest{}'.format(num)]=projector.project(modeldb.get_farthest_3d(self.object_name,num),data['RT'],'blender')
            data['small_bbox'] = projector.project(modeldb.get_small_bbox(self.object_name), data['RT'], 'blender')
            axis_direct=np.concatenate([np.identity(3), np.zeros([3, 1])], 1).astype(np.float32)
            data['van_pts']=projector.project_h(axis_direct, data['RT'], 'blender')
            database.append(data)

        save_pickle(database,pkl_file)
        return database

if __name__=='__main__':
    object_name = "ape"

    db = LineModImageDB(object_name=object_name)