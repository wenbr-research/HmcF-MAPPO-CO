#目标类

'''
在经纬度坐标系统中，使用正数和负数是为了区分地球上的不同半球：
纬度（Latitude）：
    北纬（North）用正数表示，范围从0°到90°。
    南纬（South）用负数表示，范围从0°到-90°。
经度（Longitude）：
    东经（East）用正数表示，范围从0°到180°。
    西经（West）用负数表示，范围从0°到-180°。
'''
import numpy as np

class Stronghold():
    def __init__(self, parameters, Id):
        self.stronghold_id = Id
        self.location = parameters['location']
        self.hp = parameters['hp'] # 血量，即抗击打能力，以当量定义
        self.importance = parameters['importance'] # 重要程度
        self.last_hp = self.hp # 记录上一次的血量
        self.init_hp = self.hp # 初始血量

    @property
    def max_hp(self):
        return self.init_hp

    def check_available(self):
        if self.hp > 0:
            return True
        else:
            return False

    def take_damage(self, damage):
        self.last_hp = self.hp
        self.hp -= damage

    def reset_stronghold(self):
        self.hp = 0 # 受打击TNT量，即血量，为负值
        self.last_hp = self.hp

    def return_latest_state(self):
        state_now = []
        # (5,)
        state_now.append(self.stronghold_id)
        state_now.append(self.location[0])
        state_now.append(self.location[1])
        state_now.append(self.importance)
        state_now.append(self.hp)
        return np.array(state_now)
    
    def return_latest_obs(self, sight_range, position):
        obs_now = []
        # (5,) 大小
        obs_now.append(1) # 1表示可见
        obs_now.append((self.location[0] - position[0]) / sight_range) # 相对位置
        obs_now.append((self.location[1] - position[1]) / sight_range) # 相对位置
        obs_now.append(self.hp/self.init_hp)
        
        # importance是一个0～1之间的数，表示目标的重要程度
        obs_now.append(self.importance)
        
        return np.array(obs_now)
    
    @staticmethod
    def get_stronghold_num_attributes():
        tar_state_attr_names = [
            "stronghold_id",
            "location[0]",
            "location[1]",
            "importance",
            "hp",
            "hp/init_hp"
        ]
        return len(tar_state_attr_names)