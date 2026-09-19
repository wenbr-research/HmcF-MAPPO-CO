import numpy as np
import random
from .missile import Missile
from .utils import geo as Geo

# 怪兽巢穴类（塔防术语版本，与 monster_spawn.py 等价）

class MissileLaunch():
    """
    怪兽巢穴：控制怪兽的生成波次、位置和类型
    """
    def __init__(self, params):
        '''
        初始化怪兽巢穴。
        :param params:
        '''
        self.monster_spawn_id = params['monster_spawn_id']
        self.wave_count = params['wave_count']
        self.location = params['location']
        self.spawn_interval = params['spawn_interval']
        self.last_spawn_time = 0
        self.location = params['location']
        self.monster_type = params['monster_type']
        self.dt = params['dt']
        self.max_batch_size = 2
        self.target_num = params['target_num']

    def ready_to_spawn(self):
        '''判断是否可以生成怪兽'''
        if self.wave_count > 0 and self.last_spawn_time == 0:
            return True
        return False

    def update_spawn_time(self, dt):
        '''更新生成冷却时间'''
        self.last_spawn_time -= dt
        self.last_spawn_time = max(0, self.last_spawn_time)



    def spawn_monsters(self, monster_id, target):
        '''
        生成一批怪兽，通常1~2只，设置目标、位置等信息
        :param monster_id: 已有怪兽ID，新怪兽ID递增
        :param target:
        :return:
        '''
        batch_size = random.randint(1, min(1, self.max_batch_size))

        self.wave_count -= 1

        batch_monsters = []
        for i in range(batch_size):
            monster_params = dict()
            monster_params['quantity'] = batch_size
            monster_params['monster_spawn_id'] = self.monster_spawn_id
            target_id = random.randint(0, self.target_num - 1)
            monster_params['stop_loc'] = target[target_id].location
            monster_params['targetId'] = target_id
            monster_params['start_loc'] = self.location
            monster_params['type'] = random.choice(self.monster_type)
            m = Missile(monster_params, i + monster_id)

            angle = Geo.get_azimuth(m.start_loc, m.stop_loc)
            m.set_angle(angle)
            batch_monsters.append(m)

        self.last_spawn_time = self.spawn_interval

        return batch_monsters

    def select_end(self, targets):
        target = random.choice(targets)
        return target


    # def update_attacked_time(self, dt):
    #     '''
    #     更新导弹状态
    #     :return:
    #     '''
    #     self.last_attacked_time = max(0, self.last_attacked_time - dt)

    # def return_latest_state(self):
    #     state_now = []
    #     # state是一个列表，包含导弹的ID，目标ID，数量，速度，TNT，剩余数量，最近一次被拦截的时间
    #     state_now.append(self.missile_launchId)
    #     state_now.append(self.quantity)
    #     state_now.append(self.speed)
    #     state_now.append(self.TNT)
    #     state_now.append(self.quantity)
    #     state_now.append(self.last_attacked_time)
    #     return state_now