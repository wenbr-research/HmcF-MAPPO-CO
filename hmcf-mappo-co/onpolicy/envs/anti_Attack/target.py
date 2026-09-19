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

class Target():
    def __init__(self, parameters, Id):
        self.targetId = Id
        self.location = parameters['location']
        self.HP = parameters['HP'] # 血量，即抗击打能力，以当量定义
        self.importance = parameters['importance'] # 重要程度
        self.last_HP = self.HP # 记录上一次的血量
        self.init_HP = self.HP # 初始血量

    def check_available(self):
        if self.HP > 0:
            return True
        else:
            return False

    def attacked(self, damage):
        self.last_HP = self.HP
        self.HP -= damage

    def reset_Target(self):
        self.HP = 0 # 受打击TNT量，即血量，为负值
        self.last_HP = self.HP

    def return_latest_state(self):
        state_now = []
        # (5,)
        state_now.append(self.targetId)
        state_now.append(self.location[0])
        state_now.append(self.location[1])
        state_now.append(self.importance)
        state_now.append(self.HP)
        return np.array(state_now)
    
    def return_latest_obs(self, sight_range, position):
        obs_now = []
        # (5,) 大小
        obs_now.append(1) # 1表示可见
        obs_now.append((self.location[0] - position[0]) / sight_range) # 相对位置
        obs_now.append((self.location[1] - position[1]) / sight_range) # 相对位置
        obs_now.append(self.HP/self.init_HP)
        
        # importance是一个0～1之间的数，表示目标的重要程度
        obs_now.append(self.importance)
        
        return np.array(obs_now)
    
    @staticmethod
    def get_tar_num_attributes():
        tar_state_attr_names = [
            "targetId",
            "location[0]",
            "location[1]",
            "importance",
            "HP",
            "HP/init_HP"
        ]
        return len(tar_state_attr_names)