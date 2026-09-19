import numpy as np
import random

from .utils import geo as Geo

# 怪兽类

class Monster():
    """
    定义怪兽类型
    speed:速度 m/s
    TNT:爆炸当量
    quantity:一批怪兽的数量
    """
    def __init__(self, params, Id):
        '''
        根据参数设置怪兽批次数据，包括速度、爆炸当量、数量
        如果被击落，self.quantity == 0， 如果击中目标，self.now_loc为None.
        :param params:
        '''

        self.speed = params['type']['speed']  # 怪兽移动速度
        self.HP = params['type']['HP']  # 怪兽血量
        self.quantity = params['type']['quantity']  # 当前批次中怪兽的个体数量
        self.init_quantity = params['type']['quantity']  # 初始数量，用于归一化
        self.mtype = params['type']['mtype']  # 怪兽类型标签

        self.start_loc = params['start_loc']  # 出生点
        self.now_loc = params['start_loc']  # 当前坐标
        self.stop_loc = params['stop_loc']  # 终点位置（塔防目标）

        self.targetId = params['targetId']  # 攻击目标编号（如目标塔或基地）
        self.monster_spawn_id = params['monster_spawn_id']  # 出生点ID（原导弹发射点）
        self.monster_id = Id  # 怪兽编号

        self.angle = None  # 朝向角度
        self.last_attacked_time = 0  # 最近一次被攻击的冷却时间
        self.attack_times = 0  # 被攻击次数
        self.attacked_prob_list = []  # 被攻击时命中的概率记录列表

        self.loss = self.HP  # 被消灭带来的奖励值（可与HP一致）


    def reset_monsterId(self, id):
        self.monsterId = id

    def remove_one_monster(self):
        '''
        删除怪兽
        :param monster:
        :return:
        '''
        self.quantity -= 1
        if self.quantity <= 0:
            self.quantity = 0

    def update_angle(self):
        '''
        更新怪兽角度
        :param angle:
        :return:
        '''

        self.angle = Geo.get_azimuth(self.now_loc, self.stop_loc)

    def set_angle(self, angle):
        '''
        设置怪兽的角度
        :param angle:
        :return:
        '''
        self.angle = angle

    def select_type(self, monster_type):
        '''
        随机选择怪兽类型
        :param monster_type:
        :return:
        '''
        selected_type = random.choice(monster_type)
        return selected_type

    def select_start(self):
        '''
            根据高斯分布随机选择怪兽起点
            经度 30 纬度在100和110之间随机
        '''

        # 生成高斯分布的随机点
        normal_number = np.random.normal(0, 1)
        # 将高斯分布的随机点映射到指定范围内
        mapped_number = 100 + (normal_number * (110 - 100))
        # 精度为小数后两位
        mapped_number_rounded = round(mapped_number, 2)
        return (30.0,mapped_number_rounded)

    def select_end(self, targets):
        target = random.choice(targets)

        return target

    def update_position(self, dt):
        '''
        怪兽移动到目标位置
        :return:
        '''
        dist = Geo.get_horizontal_distance(self.now_loc, self.stop_loc) # km
        if dist * 1000 < self.speed * dt: # 击中目标
            self.now_loc = None
        else:
            self.now_loc = Geo.get_geopoint_from_distance(self.now_loc, self.angle, self.speed * dt)

    def update_attacked_time(self, dt):
        '''
        更新怪兽状态
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
        obs_now.append(self.HP)
        
        # sigmoid函数归一化
        obs_now.append(1.0 / (1.0 + np.exp(-self.last_attacked_time)))
        
        return obs_now
    
    # state信息非常详细
    def return_latest_state(self):
        state_now = []
        # (7,)
        # state_now.append(self.monsterId)
        state_now.append(self.targetId)
        state_now.append(self.now_loc[0])
        state_now.append(self.now_loc[1])
        state_now.append(self.quantity)
        state_now.append(self.speed)
        state_now.append(self.HP)
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
            "HP",
            "last_attacked_time",
            "quantity/init_quantity",
        ]
        return len(enemy_state_attr_names)
    
    def set_norm_scale(self, scale):
        self.norm_scale = scale