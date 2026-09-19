# 塔类（塔防术语版本，与 tower.py 等价）
import random
import numpy as np
from .utils import geo as Geo

class FireBattalion():
    def __init__(self, params, Id, Location):
        """
        初始化塔的属性（塔防术语版本）。
        Args:
            params: dict of parameters
            range: 攻击范围 km， 例如30
            projectile_speed: 弹丸速度 m/s， 例如20
            hit_probability: 命中概率，不同塔类型不同， 例如0.7
            reaction_time: 固定反应时间 s， 例如20
            ammo_capacity: 弹药数量， 例如32
            remaining_ammo: 剩余弹药数量， 例如32
            max_attack_per_channel: 单次最大攻击数量， 通常一批几枚，例如3
            num_channels: 攻击通道数， 例如6
        """
        self.tower_id = Id
        self.range = params['range']
        self.projectile_speed = params['projectile_speed']
        self.hit_probability = params['hit_probability']
        self.reaction_time = params['reaction_time']
        self.ammo_capacity = params['ammo_capacity']
        self.remaining_ammo = self.ammo_capacity
        self.estimated_ammo = self.ammo_capacity
        self.max_attack_per_channel = params['max_attack_per_channel'] if params['max_attack_per_channel'] > 2 else 2

        self.num_channels = params['num_channels']
        self.strategy = params['strategy']
        self.ammo_per_launcher = params['ammo_per_launcher']

        '''塔属性'''
        self.location = Location

        '''r, 击杀怪兽奖励'''
        self.r = 0

        '''可视范围内怪兽, 构建observation'''
        self.visible_monsters = []

        '''
        构建动作空间 onehot vector, 0-不动作，1-拦截
        '''
        self.action_space_shape = (self.firepower_channels, self.max_fire_onebatch) # 只选择拦截目标对象；
        self.action_mask = np.zeros(self.action_space_shape) # 基于反应时间等因素，构建动作空间mask
        self.action_mask[:, 0] = 1  # 0-不动作

        '''
        选择拦截目标， 
        '''
        #interception_target中是missile对象
        self.interception_target = [None for _ in range(self.firepower_channels)] # 拦截目标，与firepower_channels对应，放导弹Id，None表示无拦截目标。固定大小

        '''
        记录已成功拦截的导弹
        '''
        self.intercepted_missile = [] # 记录已成功拦截的导弹批次弹量 [missilesId_i], Id出现了次数即为拦截枚数。

    def reset_action_mask(self):
        '''
        重置动作空间mask
        :return:
        '''
        self.action_mask = np.zeros(self.action_space_shape)
        self.action_mask[:, 0] = 1  # 0-不动作


    def reset_fire_battalion(self):
        '''
        重置火力营状态,重置剩余弹药数量,清除视野内来袭导弹和拦截目标
        :return:
        '''

        self.rest_ammo = self.ammonition_quantity
        # todo: 同时需要重置 rest_est_ammo
        self.rest_est_ammo = self.ammonition_quantity
        self.insight_missile = []
        self.interception_target = [None for _ in range(self.firepower_channels)]
        self.intercepted_missile = []


    def get_loc_from_interception_target(self, missilesId):
        '''
        获取拦截目标的位置
        :return:
        '''

        return np.where(self.interception_target == missilesId)[0]
    
    def preprocess_action(self, net_action):
        '''
        网络输出action转换为真实action
        输入：action is [0 0 2 1 2 0]

        输出：[[1 0 0]
              [1 0 0]
              [0 0 1]
              [0 1 0]
              [0 0 1]
              [1 0 0]]
        '''
        actions = np.zeros((self.firepower_channels, self.max_fire_onebatch))
        # print("shape of net_action:", net_action.shape)
        # print("shape of actions:", actions.shape)
        for i in range(self.firepower_channels):
            actions[i, net_action[i]] = 1
        return actions
        
    def reward_adding(self, reward):
        '''
        ��加��截导弹的奖��
        :param reward: ��加的��截导弹的奖��
        :return:
        '''
        self.r += reward

    def perform_action(self, action, missiles):
        '''
        拦截导弹，
        :param missile: 输入批次导弹信息
        :param action: 动作空间, shape=(num_fire_onebatch, )
        :return:
        '''
        
        # action = self.preprocess_action(action)

        # todo 拦截策略可能不同，根据不同策略进行拦截，例如：最近导弹优先、最大威胁导弹优先。预先无法确定拦截目标，这里需要重点思考。

        self.update_rest_ammo(action)

        # assert action
        if all(inter is None for inter in self.interception_target): # 无拦截目标
            assert np.sum(action) == 0, '无拦截目标，不应该发射导弹'
            return

        if np.sum(action) == 0: # 不进行任何操作
            return
        else: # 拦截导弹, 有有效操作
            for i in range(len(action)):
                # action 为shape=(num_fire_onebatch,)，每个元素为0-max_fire_onebatch，对每个火力通道操作。
                if action[i] == 0: # 不进行任何操作
                    continue

                num_missiles = action[i] # 发射导弹数量
                index = self.interception_target[i]
                missile = missiles[index]
                missile.attack_times += num_missiles # 记录导弹被拦截次数
                ## todo: 保存拦截导弹的概率，用于计算下次的拦截弹分配。
                missile.attacked_list.extend([self.hit_probability] * int(num_missiles))
                for _ in range(int(num_missiles)): # 产生随机数，判断是否命中
                    if np.random.rand() < self.hit_probability and missile.quantity > 0:
                        # self.r += missile.loss
                        self.reward_adding(missile.loss) # 奖励,涉及奖励的设置，拦截一枚的奖励
                        self.intercepted_missile.append([missile.missileId]) # 记录拦截导弹
                        missile.remove_one_missile()# quantity值减1

                #todo 火力通道和打击一一对应
                #从视野删除命中目标信息，释放火力通道,不需要记录冷却时间
                if missile.quantity == 0:
                    self.insight_missile.remove(missile.missileId)
                    self.remove_interception_target(missile.missileId)
                else:
                    # 记录观察需要的时间
                    dynamic_time = (Geo.get_horizontal_distance(self.battalion_location, missile.now_loc)
                                    / Geo.speed_ms2kmh(self.speed_of_artillery) * 3600) # 动态时间,拦截速度为m/s, 转换后计算时间
                    missile.last_attacked_time = self.fixed_reaction_time + dynamic_time

    def damage_assess(self):
        '''
        损伤评估，击毁导弹数量，发射导弹数量，需要根据论文指标进行评估
        :return:
        计算总数 or 分别结算每个火力营
        '''
        pass

    # def update_interception_target_from_insight(self, missiles):
    #     '''
    #     更新拦截目标
    #     :return:
    #     '''
    #     # 获取空火力通道
    #     empty_channels = self.get_available_interception_target()
    #     # 从视野中获取导弹放置到空火力通道，
    #     # todo 这里需要修改，根据拦截策略选择拦截目标！！！！！！！
    #     for missileId in self.insight_missile:
    #         if missiles[missileId] is None: continue
    #         missile = missiles[missileId]
    #         mask_space_for_one_channels = min(self.rest_ammo, self.max_fire_onebatch)
    #
    #         # 从insight这更新, 未被拦截的导弹，且火力通道有空余，按照某策略选择拦截目标
    #
    #         if missileId not in self.interception_target: # 从insight这更新
    #             if len(empty_channels) > 0 and missile.last_attacked_time == 0 and self.rest_ammo > 0:
    #                 empty_loc = empty_channels.pop(0)
    #                 self.interception_target[empty_loc] = missileId
    #                 self.action_mask[empty_loc, :mask_space_for_one_channels] = 1
    #                 self.action_mask[empty_loc, mask_space_for_one_channels:] = 0
    #                 self.rest_ammo -= mask_space_for_one_channels # 更新剩余弹药数量
    #
    #         # 对于已经有的火力通道，需要更新action space（例如，已经被打击过的导弹，action space会收rest_ammo影响）
    #         # 若火力通道锁定，完全打击完之前，不能释放火力通道。
    #         # todo（可能出现多个火力营同时锁定一批导弹，在未打击前都无法释放，需要确认）
    #         if missileId in self.interception_target and missile.last_attacked_time == 0:  ## 什么时候将导弹列入到拦截目标中的？
    #             miss_Loc_InInterception = self.interception_target.index(missileId)
    #             self.action_mask[miss_Loc_InInterception, :mask_space_for_one_channels] = 1
    #             self.action_mask[miss_Loc_InInterception, mask_space_for_one_channels:] = 0
    #             self.rest_ammo -= mask_space_for_one_channels
    #         # list中索引某元素的位置



    def get_available_interception_target(self):
        '''
        获取可拦截目标在火力通道上的位置
        :return:
        '''

        locs = [loc for loc in range(self.firepower_channels) if self.interception_target[loc] is None]
        return locs


    def add_insight_missile(self, missileId):
        '''
        增加新导弹（进入视野）
        :param missile:
        :return:
        '''
        self.insight_missile.append(missileId)


    def remove_insight_missile(self, missile):
        '''
        删除旧导弹（出视野）
        :param missile:
        :return:
        '''
        self.insight_missile.remove(missile)

    def remove_interception_target(self, missileId):
        '''
        删除拦截目标
        :param missile:
        :return:
        '''
        loc = self.interception_target.index(missileId)
        self.interception_target[loc] = None # 置空
        # self.action_mask[loc] = 0  # 置mask为0, 整个vector为0 [0 ~ max_fire_onebatch-1]

    def sort_missile(self, missiles):
        '''
        对视野内导弹按照某种策略排序来袭导弹顺序，可以根据距离、命中概率、重要性等排序或者训练一个排序网络
        :return:
        '''
        pass


    def select_target(self, missiles, assign_model_policy):
        # 若有多架导弹，只取命中概率最高的max_fire_onebatch架，
        # todo 拦截策略，按照概率 距离 或者攻击目标重要性等进行拦截，需要讨论
        if assign_model_policy == 'distance':
            self.insight_missile.sort(key=lambda x: Geo.get_horizontal_distance(missiles[x].now_loc, self.battalion_location))# todo 按照距离由小到大排序
        elif assign_model_policy == 'TNT': # 'TNT', 'speed', 'comprehensive'
            self.insight_missile.sort(key=lambda x: -missiles[x].TNT) # todo 按照TNT由大到小排序
        elif assign_model_policy == 'speed':
            self.insight_missile.sort(key=lambda x: -missiles[x].speed) # todo 按照速度由大到小排序
        elif assign_model_policy == 'comprehensive':
            self.insight_missile.sort(key=lambda x: -Geo.get_horizontal_distance(missiles[x].now_loc,
                                                self.battalion_location) * missiles[x].TNT * missiles[x].speed) # todo 综合排序. 或者离重要目标距离排序

        locs = self.get_available_interception_target() # 获取空的火力通道，不改变已有的拦截目标
        for ins in self.insight_missile: # 按照打击概率放入火力通道# ?这是按照距离加入的
            if len(locs) == 0 or ins in self.interception_target: continue # 火力通道已满，或者目标已在拦截目标中
            loc = locs.pop(0)
            self.interception_target[loc] = ins

        # 但是受制于弹药量，可能无法全部打击，需要mask. todo: 注意，这里的mask是根据剩余弹药量计算的，如果是场景二（是否开火），mask表示是否可发射。
        ## todo:                                具体执行还需要根据action完成，并更新剩余弹药量！！  函数下面的代码未改。
        for loc in range(self.firepower_channels):
            missileId = self.interception_target[loc]
            
            if missileId is None:  # 无拦截目标,进行下一次循环
                self.action_mask[loc, :] = 0
                self.action_mask[loc, 0] = 1  # 0-不动作
                continue
            if missiles[missileId].last_attacked_time > 0:  # 未到冷却时间，不能打击
                self.action_mask[loc, :] = 0
                self.action_mask[loc, 0] = 1  # 0-不动作
                continue

            mask_space_for_one_channels = min(self.rest_est_ammo, self.max_fire_onebatch-1) # 0-不动作，max-最大发射数量[0-4],5个
            if mask_space_for_one_channels > 0: #
                assert self.rest_ammo > 0, '剩余弹药数量不应该为负数'

            if len(self.action_mask[0])-1 == mask_space_for_one_channels:  # 均可发射，一次发射量等于最大发射数量
                self.action_mask[loc, :] = 1
            elif mask_space_for_one_channels == 0:  # 无法发射
                self.action_mask[loc, :] = 0
                self.action_mask[loc, 0] = 1
            else:
                self.action_mask[loc, :(mask_space_for_one_channels+1)] = 1
                self.action_mask[loc, (mask_space_for_one_channels+1):] = 0

            self.rest_est_ammo -= mask_space_for_one_channels  # 更新剩余弹药数量
            assert self.rest_est_ammo >= 0, '剩余弹药数量不应该为负数'
            self.rest_est_ammo = max(0, self.rest_est_ammo) # 保证不为负数, 但这里不应该为负数，因为mask_space_for_one_channels是根据剩余弹药数量计算的

        # print("after action_mask:", self.action_mask)


    def get_rest_ammo(self):
        '''
        获取剩余弹药数量
        :return:
        '''
        return self.ammonition_quantity

    def update_rest_ammo(self, action):
        '''
        更新剩余弹药数量, action-onehot vector (fire_channels x max_fire_onebatch) 0-1
        实际弹量消耗
        :param action: 该agent的动作空间，shape=(num_fire_onebatch, ), 每个元素为0-max_fire_onebatch (例如0，1，2，3，4)
        :return:
        '''
        #遍历所有火力通道的非0值，每遇到一个弹药量减1
        self.rest_ammo -= np.sum(action)
        assert self.rest_ammo >= 0, '剩余弹药数量不应该为负数'
        # 更新剩余估计弹药数量
        self.rest_est_ammo = self.rest_ammo
    
    # 获取当前观察
    def return_latest_obs(self, sight_range, position):
        obs_now = []
        
        # 大小(10,)
        obs_now.append(1) # 代表可见
        obs_now.append((self.battalion_location[0]-position[0])/sight_range) # 归一化
        obs_now.append((self.battalion_location[1]-position[1])/sight_range) # 归一化
        obs_now.append(self.rest_ammo/self.ammonition_quantity) # 归一化
        
        # 每一个属性值需归一化
        obs_now.append(self.range_of_attack / self.norm_scale["range_of_attack"])
        obs_now.append(self.speed_of_artillery / self.norm_scale["speed_of_artillery"])
        obs_now.append(self.hit_probability / self.norm_scale["hit_probability"])
        
        obs_now.append(self.fixed_reaction_time / self.norm_scale["fixed_reaction_time"])
        obs_now.append(self.max_fire_onebatch / self.norm_scale["max_fire_onebatch"])
        
        obs_now.append(self.firepower_channels / self.norm_scale["firepower_channels"])
        
        return np.array(obs_now)

    # 获取当前状态,todo 考虑更新为字典形式，，方便查找
    def return_latest_state(self):
        state_now = []
        # (11,)
        state_now.append(self.battalionId)
        state_now.append(self.battalion_location[0])
        state_now.append(self.battalion_location[1])
        state_now.append(self.range_of_attack)
        state_now.append(self.speed_of_artillery)
        state_now.append(self.hit_probability)
        state_now.append(self.fixed_reaction_time)
        state_now.append(self.ammonition_quantity)
        state_now.append(self.max_fire_onebatch)
        state_now.append(self.rest_ammo)
        state_now.append(self.firepower_channels)
        return np.array(state_now)
    
    @staticmethod
    def get_ally_num_attributes():
        ally_state_attr_names = [
            "battalionId",
            "battalion_location[0]",
            "battalion_location[1]",
            "range_of_attack",
            "speed_of_artillery",
            "hit_probability",
            "fixed_reaction_time",
            "ammonition_quantity",
            "max_fire_onebatch",
            "rest_ammo",
            "firepower_channels",
            "rest_ammo/ammonition_quantity",
        ]
        return len(ally_state_attr_names)

    # 奖励清零
    def reset_reward(self):
        self.r = 0
        return

    # 获取奖励
    def get_reward(self):
        return self.r
    
    # 归一化最大值
    def set_norm_scale(self, max_value):
        self.norm_scale = max_value