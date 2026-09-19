import numpy as np
import random
from .monster import Monster
from .utils import geo as Geo

# 怪兽类

class MonsterSpawn():
    """
    定义怪兽类型
    """
    def __init__(self, params):
        """
        初始化怪兽出生点 MonsterSpawn。

        每个出生点控制若干波怪兽的生成，定义生成频率、起始位置、怪兽种类等。

        参数 params 包含以下字段：
        - 'monster_spawn_id'：出生点编号（原 missile_launchId）
        - 'wave_count'：剩余怪兽波数（每波可生成多个怪兽）
        - 'location'：出生点坐标（经纬度）
        - 'spawn_interval'：每波怪兽之间的最小生成间隔
        - 'monster_type'：怪兽类型列表（每个类型包含属性如 HP、速度等）
        - 'dt'：时间步长，用于模拟推进
        - 'target_num'：目标数量，用于随机选择怪兽攻击目标
        """

        self.monster_spawn_id = params['monster_spawn_id']  # 出生点编号
        self.wave_count = params['wave_count']  # 剩余待生成的怪兽波数
        self.location = params['location']  # 出生点坐标（如地图坐标）

        self.spawn_interval = params['spawn_interval']  # 两波怪兽之间的生成间隔时间
        self.last_spawn_time = 0  # 上一次生成后的冷却计时器

        self.monster_type = params['monster_type']  # 可供生成的怪兽类型（每种包含 speed/HP 等）
        self.dt = params['dt']  # 每步仿真时间（外部驱动）

        self.max_batch_size = 2  # 每波最多生成怪兽的数量（随机 1~2）
        self.target_num = params['target_num']  # 可供怪兽攻击的目标总数


    def ready_to_spawn(self):
        """
        判断当前是否满足生成怪兽的条件。

        满足以下两个条件时可生成怪兽：
        1. self.wave_count > 0：还有剩余波次可生成；
        2. self.last_spawn_time == 0：生成间隔冷却时间已到。

        :return: True 表示可以生成怪兽，False 表示仍在冷却或已无怪兽波次可生成。
        """
        if self.wave_count > 0 and self.last_spawn_time == 0:
            return True
        return False


    def update_spawn_time(self, dt):
        """
        更新怪兽出生点的冷却计时器。

        每次调用该函数，将上一波怪兽生成后的冷却时间 `last_spawn_time` 递减 dt。
        若冷却时间减至 0，表示可以开始生成下一波怪兽。

        参数：
            dt (float)：仿真步长（秒），每帧推进的时间间隔。

        示例：
            假设 `spawn_interval = 10`，表示两波怪兽之间间隔 10 秒。
            每次 tick 调用本函数，会减去 dt（如 0.5），直至冷却归零。
        """
        self.last_spawn_time -= dt
        self.last_spawn_time = max(0, self.last_spawn_time)  # 冷却时间不能为负


    def spawn_monsters(self, monster_id, targets):
        """
        生成一批怪兽（模拟塔防中从出生点生成怪物的行为）。

        每次生成时，从当前出生点出发，创建 1~max_batch_size 个怪兽，
        并为它们分配目标坐标、角度、起点等属性。

        参数：
            monster_id (int): 怪兽编号起始值，用于确保编号唯一。
            targets (list): 可选目标点对象列表（每个对象应有 .location 属性）。

        返回：
            list: 一个包含生成的 Monster 实例的列表。
        """
        # 随机决定本次生成几只怪兽（1 ~ max_batch_size）
        batch_size = random.randint(1, min(1, self.max_batch_size))
        # batch_size = random.randint(1, self.max_batch_size)
        self.wave_count -= 1  # 减少剩余生成波次数

        batch_monsters = []
        for i in range(batch_size):
            monster_params = {
                'quantity': batch_size,
                'monster_spawn_id': self.monster_spawn_id,
                'start_loc': self.location,
                'type': random.choice(self.monster_type),
            }

            # 为该怪兽分配一个目标（位置 + ID）
            target_id = random.randint(0, self.target_num - 1)
            monster_params['targetId'] = target_id
            monster_params['stop_loc'] = targets[target_id].location

            # 构造怪兽对象
            m = Monster(monster_params, monster_id + i)
            # m.set_angle(Geo.get_azimuth(m.start_loc, m.stop_loc))
            m.set_angle(Geo.get_azimuth(m.start_loc, m.stop_loc))
            batch_monsters.append(m)

        # 重置冷却计时器
        self.last_spawn_time = self.spawn_interval

        return batch_monsters


    def select_end(self, targets):
        """
        随机选择一个目标作为怪兽的终点。

        怪兽生成后会从出生点移动到该目标位置，模拟塔防中“攻击基地/塔”的路径行为。

        参数：
            targets (list): 目标对象列表，每个对象应包含 location 属性（如位置坐标）

        返回：
            object: 被选中的目标对象
        """
        return random.choice(targets)
