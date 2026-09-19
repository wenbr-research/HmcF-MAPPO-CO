import numpy as np
import random
from .utils import geo

# 导弹类

class Monster():
    """
    定义导弹类型
    speed:速度 m/s
    TNT:爆炸当量
    quantity:一批导弹的数量
    """
    def __init__(self, params, Id):
        '''
        根据参数设置导弹批次数据，包括速度、爆炸当量、数量
        如果被击落，self.quantity == 0， 如果击中目标，self.now_loc为None.
        :param params:
        '''
        self.speed = params['type']['speed']
        self.damage = params['type']['damage']
        self.quantity = params['type']['quantity']
        #self.target = params['target']
        self.start_loc = params['start_loc']
        self.now_loc = params['start_loc'] # 当前位置
        self.stop_loc = params['stop_loc']
        self.stronghold_id = params['stronghold_id']
        self.nest_id = params['nest_id']
        self.monster_id = Id # 导弹批次ID
        self.angle = None
        self.last_attacked_time = 0 # 最近一次要被拦截的时间（未来），用于计算拦截时间间隔
        self.attack_times = 0 # 被拦截次数
        self.loss = self.damage # 损失一枚，对方的奖励，默认为1
        self.init_quantity = params['type']['quantity']
        self.monster_type = params['type']['monster_type'] # 导弹类型

    def reset_monster_id(self, id):
        self.monster_id = id


    @property
    def hp(self):
        return self.quantity

    @property
    def max_hp(self):
        return self.init_quantity

    @property
    def cooldown_remaining(self):
        return self.last_attacked_time

    @property
    def times_hit(self):
        return self.attack_times

    def remove_one_monster(self):
        '''
        删除导弹
        :param monster:
        :return:
        '''
        self.quantity -= 1
        if self.quantity <= 0:
            self.quantity = 0


    def update_angle(self):
        '''
        更新导弹角度
        :param angle:
        :return:
        '''

        self.angle = geo.get_azimuth(self.now_loc, self.stop_loc)

    def set_angle(self, angle):
        '''
        设置导弹的角度
        :param angle:
        :return:
        '''
        self.angle = angle


    # 随机返回一个导弹类型
    def select_type(self, monster_type):
        '''
        随机选择导弹类型
        :param monster_type:
        :return:
        '''
        selected_type = random.choice(monster_type)
        return selected_type

    # 随机设置导弹坐标
    def select_start(self):
        '''
            根据高斯分布随机选择导弹起点
            经度 30 纬度在100和110之间随机
        '''

        # 生成高斯分布的随机点
        normal_number = np.random.normal(0, 1)
        # 将高斯分布的随机点映射到指定范围内
        mapped_number = 100 + (normal_number * (110 - 100))
        # 精度为小数后两位
        mapped_number_rounded = round(mapped_number, 2)
        return (30.0,mapped_number_rounded)

    #随机选取导弹的目标
    def select_end(self, targets):
        target = random.choice(targets)

        return target


    def update_position(self, dt):
        '''
        导弹移动到目标位置
        :return:
        '''
        dist = geo.get_horizontal_distance(self.now_loc, self.stop_loc) # km
        if dist * 1000 < self.speed * dt: # 击中目标
            self.now_loc = None
        else:
            self.now_loc = geo.get_geopoint_from_distance(self.now_loc, self.angle, self.speed * dt)


    def update_attacked_time(self, dt):
        '''
        更新导弹状态
        :return:
        '''
        self.last_attacked_time = max(0, self.last_attacked_time - dt)

    # obs信息较为简略
    def return_latest_obs(self, sight_range, position):
        obs_now = []
        # (6,)大小
        obs_now.append((self.now_loc[0] - position[0]) / sight_range) # 相对位置
        obs_now.append((self.now_loc[1] - position[1]) / sight_range) # 相对位置
        obs_now.append(self.quantity/self.init_quantity)
        
        # 在外部进行归一化，这里不处理
        obs_now.append(self.speed)
        obs_now.append(self.damage)
        
        # sigmoid函数归一化
        obs_now.append(1.0 / (1.0 + np.exp(-self.last_attacked_time)))
        
        return obs_now
    
    # state信息非常详细
    def return_latest_state(self):
        state_now = []
        # (7,)
        # state_now.append(self.monster_id)
        state_now.append(self.stronghold_id)
        state_now.append(self.now_loc[0])
        state_now.append(self.now_loc[1])
        state_now.append(self.quantity)
        state_now.append(self.speed)
        state_now.append(self.damage)
        state_now.append(self.last_attacked_time)
        return np.array(state_now)
    
    @staticmethod
    def get_enemy_num_attributes():
        enemy_state_attr_names = [
            "targetId",
            "now_loc[0]",
            "now_loc[1]",
            "quantity",
            "speed",
            "damage",
            "last_attacked_time",
            "quantity/init_quantity",
        ]
        return len(enemy_state_attr_names)
    
    def set_norm_scale(self, scale):
        self.norm_scale = scale