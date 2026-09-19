import random
import numpy as np
from .utils import geo as Geo


# class FireBattalion():
class Tower():
    """
    塔类（原火力营 FireBattalion）：
    模拟塔防场景中用于攻击怪兽的防御塔，具有冷却通道、攻击范围、命中概率等属性。
    """
    def __init__(self, params, Id, location):
        """
        初始化塔的属性。

        Args:
            params: dict, 塔的参数配置：包括攻击范围、速度、命中率、弹药量等。
            Id: int, 当前塔的唯一编号。
            location: list/tuple, 当前塔的坐标位置。
        """
        self.tower_id = Id
        self.range = params['range']  # 攻击范围（单位：km）
        self.projectile_speed = params['projectile_speed']  # 发射子弹的速度（单位：m/s）
        self.hit_probability = params['hit_probability']  # 命中概率
        self.reaction_time = params['reaction_time']  # 塔的固定反应时间（单位：秒）
        self.ammo_capacity = params['ammo_capacity']  # 初始弹药量
        self.remaining_ammo = self.ammo_capacity  # 当前剩余弹药
        self.estimated_ammo = self.ammo_capacity  # 估计弹药（用于动作空间限制）

        self.max_attack_per_channel = params['max_attack_per_channel'] if params['max_attack_per_channel'] > 2 else 2  # 每次最多攻击数量
        self.num_channels = params['num_channels']  # 并行攻击通道数
        self.strategy = params['strategy']  # 目标选择策略
        self.projectile_id_offset = params['ammo_per_launcher']  # 初始投射编号偏移（用于编码）

        self.location = location  # 塔的位置坐标

        self.reward = 0  # 当前累计奖励值
        self.visible_monsters = []  # 当前视野内可见怪兽 ID 列表

        self.action_space_shape = (self.num_channels, self.max_attack_per_channel)  # 动作空间尺寸
        self.action_mask = np.zeros(self.action_space_shape)  # 动作 mask，用于掩码合法动作
        self.action_mask[:, 0] = 1  # 默认每通道最小动作为“不攻击”

        self.attack_targets = [None for _ in range(self.num_channels)]  # 每个通道锁定的怪兽 ID
        self.killed_monsters = []  # 成功击杀的怪兽 ID 记录


    def reset_action_mask(self):
        """ 重置动作掩码，仅允许“不攻击”动作为合法动作 """
        self.action_mask = np.zeros(self.action_space_shape)
        self.action_mask[:, 0] = 1

    def reset_tower(self):
        """ 重置塔的所有战斗状态，包括弹药、通道锁定和奖励 """
        self.remaining_ammo = self.ammo_capacity
        self.estimated_ammo = self.ammo_capacity
        self.visible_monsters.clear()
        self.attack_targets = [None for _ in range(self.num_channels)]
        self.killed_monsters.clear()

    def get_channel_index_for_target(self, monster_id):
        """
        获取指定怪兽当前被锁定在哪个通道上。

        Args:
            monster_id (int): 怪兽 ID

        Returns:
            list: 匹配的通道索引列表（通常最多一个）
        """
        return np.where(np.array(self.attack_targets) == monster_id)[0]

    def preprocess_action(self, net_action):
        """
        将网络输出的动作索引转换为 one-hot 编码格式。
        Args:
            net_action: list[int], 每个通道对应的离散动作索引
        Returns:
            actions: np.ndarray, one-hot 编码动作矩阵
        """
        actions = np.zeros((self.num_channels, self.max_attack_per_channel))
        for i in range(self.num_channels):
            actions[i, net_action[i]] = 1
        return actions

    def reward_adding(self, reward):
        """
        增加当前塔累计的击杀奖励。

        Args:
            reward (float): 本次攻击所获得的奖励值
        """
        self.reward += reward

    def perform_action(self, action, monsters):
        """
        根据决策执行攻击，更新怪兽状态并计算奖励。

        Args:
            action (list[int]): 每个通道选择攻击的怪兽数量（0 表示不攻击）
            monsters (dict[int, Monster]): 所有怪兽对象，以 ID 为 key
        """
        self.update_remaining_ammo(action)

        # 若所有通道都未锁定目标，动作也必须是全 0
        if all(t is None for t in self.attack_targets):
            assert np.sum(action) == 0, "无攻击目标却执行了攻击动作"
            return

        if np.sum(action) == 0:
            return  # 所有通道均选择不攻击

        for i in range(len(action)):
            if action[i] == 0:
                continue  # 当前通道未执行攻击

            num_attacks = action[i]
            target_id = self.attack_targets[i]
            monster = monsters[target_id]

            # 记录被攻击次数与命中概率
            monster.attack_times += num_attacks
            monster.attacked_prob_list.extend([self.hit_probability] * int(num_attacks))

            # 判定命中，每命中一次计算奖励与怪兽状态更新
            for _ in range(int(num_attacks)):
                if np.random.rand() < self.hit_probability and monster.quantity > 0:
                    self.reward_adding(monster.loss)
                    self.killed_monsters.append(monster.monster_id)
                    monster.remove_one_monster()

            # 若怪兽死亡，释放通道并移出视野
            if monster.quantity == 0:
                self.visible_monsters.remove(monster.monster_id)
                self.remove_attack_target(monster.monster_id)
            else:
                # 仍存活，更新其“下次可被攻击”的时间
                dynamic_time = Geo.get_horizontal_distance(self.location, monster.now_loc) \
                               / Geo.speed_ms2kmh(self.projectile_speed) * 3600
                monster.last_attacked_time = self.reaction_time + dynamic_time


    def damage_assess(self):
        '''
        损伤评估，击毁导弹数量，发射导弹数量，需要根据论文指标进行评估
        :return:
        计算总数 or 分别结算每个火力营
        '''
        pass

    def get_available_channels(self):
        """
        获取当前未锁定目标的攻击通道编号列表。

        Returns:
            list[int]: 所有空闲攻击通道的索引
        """
        return [i for i, target in enumerate(self.attack_targets) if target is None]

    def add_visible_monster(self, monster_id):
        """ 添加一个进入视野的怪兽 """
        self.visible_monsters.append(monster_id)

    def remove_visible_monster(self, monster_id):
        """ 移除一个离开视野的怪兽 """
        self.visible_monsters.remove(monster_id)

    def remove_attack_target(self, monster_id):
        """ 移除已死亡或不可攻击的目标 """
        if monster_id in self.attack_targets:
            self.attack_targets[self.attack_targets.index(monster_id)] = None

    def sort_missile(self, missiles):
        '''
        对视野内导弹按照某种策略排序来袭导弹顺序，可以根据距离、命中概率、重要性等排序或者训练一个排序网络
        :return:
        '''
        pass

    def select_targets(self, monsters, policy):
        """
        根据策略从视野内怪兽中选择目标分配给各通道。
        Args:
            monsters: dict[int, Monster], 所有怪兽对象
            policy: str, 策略类型（如 'distance', 'TNT', 'speed', 'comprehensive'）
        """
        if policy == 'distance':
            self.visible_monsters.sort(key=lambda x: Geo.get_horizontal_distance(monsters[x].now_loc, self.location))
        elif policy == 'TNT':
            self.visible_monsters.sort(key=lambda x: -monsters[x].HP)
        elif policy == 'speed':
            self.visible_monsters.sort(key=lambda x: -monsters[x].speed)
        elif policy == 'comprehensive':
            self.visible_monsters.sort(
                key=lambda x: -Geo.get_horizontal_distance(monsters[x].now_loc, self.location) * monsters[x].HP *
                              monsters[x].speed)

        available = self.get_available_channels()
        for m_id in self.visible_monsters:
            if len(available) == 0 or m_id in self.attack_targets:
                continue
            idx = available.pop(0)
            self.attack_targets[idx] = m_id

        for i in range(self.num_channels):
            m_id = self.attack_targets[i]
            if m_id is None or monsters[m_id].last_attacked_time > 0:
                self.action_mask[i, :] = 0
                self.action_mask[i, 0] = 1
                continue

            max_fire = min(self.estimated_ammo, self.max_attack_per_channel - 1)
            if max_fire == 0:
                self.action_mask[i, :] = 0
                self.action_mask[i, 0] = 1
            else:
                self.action_mask[i, :(max_fire + 1)] = 1
                self.action_mask[i, (max_fire + 1):] = 0
            self.estimated_ammo -= max_fire

    def get_remaining_ammo(self):
        return self.remaining_ammo

    def update_remaining_ammo(self, action):
        """ 根据动作减少对应的弹药量，并同步估计值 """
        self.remaining_ammo -= np.sum(action)
        assert self.remaining_ammo >= 0, '剩余弹药不能为负'
        self.estimated_ammo = self.remaining_ammo

    def return_latest_obs(self, sight_range, position):
        """
        返回当前塔的归一化观测信息。
        Args:
            sight_range: float, 观察范围
            position: tuple, 全局参考位置
        Returns:
            np.ndarray: 观测向量
        """
        obs = [
            1,
            (self.position[0] - position[0]) / sight_range,
            (self.position[1] - position[1]) / sight_range,
            self.remaining_ammo / self.ammo_capacity,
            self.range / self.norm_scale["range"],
            self.projectile_speed / self.norm_scale["projectile_speed"],
            self.hit_probability / self.norm_scale["hit_probability"],
            self.reaction_time / self.norm_scale["reaction_time"],
            self.max_attack_per_channel / self.norm_scale["max_attack_per_channel"],
            self.num_channels / self.norm_scale["num_channels"]
        ]
        return np.array(obs)

    def return_latest_state(self):
        """ 返回当前塔的完整状态向量 """
        return np.array([
            self.tower_id,
            self.location[0],
            self.location[1],
            self.range,
            self.projectile_speed,
            self.hit_probability,
            self.reaction_time,
            self.ammo_capacity,
            self.max_attack_per_channel,
            self.remaining_ammo,
            self.num_channels
        ])


    @staticmethod
    def get_ally_num_attributes():
        """ 返回塔状态向量的属性数量（用于神经网络输入定义） """
        return 12

    def reset_reward(self):
        """ 清空当前累计奖励 """
        self.reward = 0

    def get_reward(self):
        """ 获取当前累计奖励 """
        return self.reward

    def set_norm_scale(self, scale_dict):
        """ 设置用于归一化的最大值字典 """
        self.norm_scale = scale_dict