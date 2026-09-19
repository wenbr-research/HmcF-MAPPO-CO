import random
from .monster import Monster
from .utils import geo

# 导弹类

class MonsterNest():
    """
    定义导弹类型
    speed:速度 m/s
    TNT:爆炸当量
    quantity:一批导弹的数量
    """
    def __init__(self, params):
        '''
        根据参数设置导弹批次数据，包括速度、爆炸当量、数量
        如果被击落，self.quantity == 0， 如果击中目标，self.now_loc为None.
        :param params:
        '''
        self.nest_id = params['nest_id']
        self.quantity = params['quantity'] # 剩余弹药批次，每次发射一批
        self.location = params['location']
        self.spawn_interval = params['spawn_interval']
        self.last_launch_time = 0
        self.location = params['location']
        self.monster_type = params['monster_types']
        self.dt = params['dt']
        self.max_quantity_batch = 2 # 一批导弹的最大数量
        self.target_num = params.get('target_num', params.get('stronghold_num', 5))

    def ready_to_spawn(self):
        '''
        判断是否可以发射导弹
        :return:
        '''
        if self.quantity > 0 and self.last_launch_time == 0:
            return True
        return False

    def update_spawn_cooldown(self, dt):
        '''
        更新导弹发射时间
        :return:
        '''
        self.last_launch_time -= dt
        self.last_launch_time = max(0, self.last_launch_time)



    def spawn_monsters(self, monsterId, target):
        '''
        发射导弹, 一次发射一批，通常为1、2枚，需要设置导弹目标，位置等信息
        :param monsterId: 已经有的导弹Id，之后的导弹Id需要递增
        :param target:
        :return:
        '''

        # 可以发射导弹，随机产生一批导弹个数，并构造导弹

        quantity_batch = random.randint(1, min(1, self.max_quantity_batch)) # 一批导弹的数量，最大为2，最小为1

        self.quantity -= 1 # 更新剩余弹药批量

        batch_monsters = []
        for i in range(quantity_batch):
            monster_params = dict()
            # monster_params['monsterId'] = self.nest_id
            monster_params['quantity'] = quantity_batch
            monster_params['nest_id'] = self.nest_id
            target_id = random.randint(0, self.target_num - 1) # 随机选择一个目标
            monster_params['stop_loc'] = target[target_id].location
            monster_params['stronghold_id'] = target_id
            monster_params['start_loc'] = self.location
            monster_params['type'] = random.choice(self.monster_type)
            m = Monster(monster_params, i + monsterId)

            angle = geo.get_azimuth(m.start_loc, m.stop_loc)
            m.set_angle(angle)
            batch_monsters.append(m)

        # 更新剩余发射时间
        self.last_launch_time = self.spawn_interval

        return batch_monsters

    #随机选取导弹的目标
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
    #     state_now.append(self.nest_id)
    #     state_now.append(self.quantity)
    #     state_now.append(self.speed)
    #     state_now.append(self.damage)
    #     state_now.append(self.quantity)
    #     state_now.append(self.last_attacked_time)
    #     return state_now