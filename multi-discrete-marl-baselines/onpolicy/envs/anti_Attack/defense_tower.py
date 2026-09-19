#火力营类
import numpy as np
from .utils import geo

class DefenseTower():
    def __init__(self, params, Id, Location):
        """
        Initialize fire battalion with given parameters. 假设一个发射架一枚导弹，N个发射架N枚导弹。只要有火力通道，就可以发射。
        Args:
            params: dict of parameters
            range_of_attack: 打击范围 km， 例如30
            speed_of_artillery: 导弹速度 m/s， 例如20
            hit_probability: 命中概率，不同火力营类型不同， 例如0.7
            fixed_reaction_time: 固定反应时间 s， 例如20
            ammunition_quantity: 弹药数量， 例如32
            rest_ammo: 剩余弹药数量， 例如32
            max_fire_onebatch: 单次最大发射数量， 通常一批几枚，例如3
            firepower_channels: 火力通道，指雷达照射批次 例如6
        """
        self.tower_id = Id
        self.attack_range = params['attack_range']
        self.projectile_speed = params['projectile_speed']
        self.hit_rate = params['hit_rate']
        self.fixed_reaction_time = params['fixed_reaction_time']
        self.max_ammo = params['max_ammo']
        self.current_ammo = self.max_ammo # 剩余弹药数量 = 弹药数量， 实际剩余弹量
        self.estimated_ammo = self.max_ammo # 剩余估计弹药数量 = 弹药数量，用于估计action space
        self.max_attack_per_channel = params['max_attack_per_channel']
        self.attack_channels = params['attack_channels']
        self.defense_strategy = params['defense_strategy']
        self.projectile_count = params['projectile_count']

        '''
        火力营属性
        '''
        # 火力营位置
        self.tower_location = Location

        '''
        r, 拦截导弹奖励
        '''
        self.r = 0

        '''
        来袭目标属性,范围内导弹, 构建observation
        '''
        self.visible_monsters = []

        '''
        构建动作空间 onehot vector, 0-不动作，1-拦截
        '''
        self.action_space_shape = (self.attack_channels, self.max_attack_per_channel) # 只选择拦截目标对象；
        self.action_mask = np.zeros(self.action_space_shape) # 基于反应时间等因素，构建动作空间mask
        self.action_mask[:, 0] = 1  # 0-不动作

        '''
        选择拦截目标， 
        '''
        #locked_targets中是monster对象
        self.locked_targets = [None for _ in range(self.attack_channels)] # 拦截目标，与firepower_channels对应，放导弹Id，None表示无拦截目标。固定大小

        '''
        记录已成功拦截的导弹
        '''
        self.intercepted_monsters = [] # 记录已成功拦截的导弹批次弹量 [monster_id_i], Id出现了次数即为拦截枚数。

    def reset_action_mask(self):
        '''
        重置动作空间mask
        :return:
        '''
        self.action_mask = np.zeros(self.action_space_shape)
        self.action_mask[:, 0] = 1  # 0-不动作


    def reset_defense_tower(self):
        '''
        重置火力营状态,重置剩余弹药数量,清除视野内来袭导弹和拦截目标
        :return:
        '''

        self.current_ammo = self.max_ammo
        self.visible_monsters = []
        self.locked_targets = [None for _ in range(self.attack_channels)]

    def get_loc_from_locked_target(self, monster_id):
        '''
        获取拦截目标的位置
        :return:
        '''

        return np.where(self.locked_targets == monster_id)[0]
    
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
        actions = np.zeros((self.attack_channels, self.max_attack_per_channel))
        # print("shape of net_action:", net_action.shape)
        # print("shape of actions:", actions.shape)
        for i in range(self.attack_channels):
            actions[i, net_action[i]] = 1
        return actions
        
    def reward_adding(self, reward):
        '''
        ��加��截导弹的奖��
        :param reward: ��加的��截导弹的奖��
        :return:
        '''
        self.r += reward

    def perform_action(self, action, monsters):
        '''
        拦截导弹，
        :param monster: 输入批次导弹信息
        :param action: 动作空间, shape=(num_fire_onebatch, )
        :return:
        '''
        
        # action = self.preprocess_action(action)

        # todo 拦截策略可能不同，根据不同策略进行拦截，例如：最近导弹优先、最大威胁导弹优先。预先无法确定拦截目标，这里需要重点思考。

        self.update_rest_ammo(action)

        # assert action
        if all(inter is None for inter in self.locked_targets): # 无拦截目标
            assert np.sum(action) == 0, '无拦截目标，不应该发射导弹'
            return

        if np.sum(action) == 0: # 不进行任何操作
            return
        else: # 拦截导弹, 有有效操作
            for i in range(len(action)):
                # action 为shape=(num_fire_onebatch,)，每个元素为0-max_fire_onebatch，对每个火力通道操作。
                if action[i] == 0: # 不进行任何操作
                    continue

        #         if np.sum(action[i]) == 0:
        #
        # for i in range(len(action)): # 遍历每个火力通道 action: (num_fire_onebatch x max_fire_onebatch)
        #     assert np.sum(action[i]) == 1, '每个火力通道只能发射一次, 或者不发射'
        #
        #     if action[i][0] == 1: # 不进行任何操作
        #         continue
        #
        #     if self.locked_targets[i] is None: # 无拦截目标,不执行拦截，进行下一次循环
        #         assert np.argmax(action[i]) == 0, '无拦截目标，不应该发射导弹'
        #         continue
        #
        #     ## 有拦截目标, 产生随机数，判断是否命中,
        #     num_missiles = np.argmax(action[i]) #开火次数
        #     if num_missiles == 0: # 不开火,决定此次动作不开火，则无需进行任务操作，也不需要设置被打击时间。
        #         continue
                num_missiles = action[i] # 发射导弹数量
                index = self.locked_targets[i]
                monster = monsters[index]
                monster.attack_times += num_missiles # 记录导弹被拦截次数
                for _ in range(num_missiles): # 产生随机数，判断是否命中
                    if np.random.rand() < self.hit_rate and monster.quantity > 0:
                        # self.r += monster.loss
                        self.reward_adding(monster.loss) # 奖励,涉及奖励的设置，拦截一枚的奖励
                        self.intercepted_monsters.append([monster.monster_id]) # 记录拦截导弹
                        monster.remove_one_monster()# quantity值减1

                #todo 火力通道和打击一一对应
                #从视野删除命中目标信息，释放火力通道,不需要记录冷却时间
                if monster.quantity == 0:
                    self.visible_monsters.remove(monster.monster_id)
                    self.release_locked_target(monster.monster_id)
                else:
                    # 记录观察需要的时间
                    dynamic_time = (geo.get_horizontal_distance(self.tower_location, monster.now_loc)
                                    / geo.speed_ms2kmh(self.projectile_speed) * 3600) # 动态时间,拦截速度为m/s, 转换后计算时间
                    monster.last_attacked_time = self.fixed_reaction_time + dynamic_time

    def damage_assess(self):
        '''
        损伤评估，击毁导弹数量，发射导弹数量，需要根据论文指标进行评估
        :return:
        计算总数 or 分别结算每个火力营
        '''
        pass

    # def update_locked_targets_from_insight(self, monsters):
    #     '''
    #     更新拦截目标
    #     :return:
    #     '''
    #     # 获取空火力通道
    #     empty_channels = self.get_free_channels()
    #     # 从视野中获取导弹放置到空火力通道，
    #     # todo 这里需要修改，根据拦截策略选择拦截目标！！！！！！！
    #     for monster_id in self.visible_monsters:
    #         if monsters[monster_id] is None: continue
    #         monster = monsters[monster_id]
    #         mask_space_for_one_channels = min(self.current_ammo, self.max_attack_per_channel)
    #
    #         # 从insight这更新, 未被拦截的导弹，且火力通道有空余，按照某策略选择拦截目标
    #
    #         if monster_id not in self.locked_targets: # 从insight这更新
    #             if len(empty_channels) > 0 and monster.last_attacked_time == 0 and self.current_ammo > 0:
    #                 empty_loc = empty_channels.pop(0)
    #                 self.locked_targets[empty_loc] = monster_id
    #                 self.action_mask[empty_loc, :mask_space_for_one_channels] = 1
    #                 self.action_mask[empty_loc, mask_space_for_one_channels:] = 0
    #                 self.current_ammo -= mask_space_for_one_channels # 更新剩余弹药数量
    #
    #         # 对于已经有的火力通道，需要更新action space（例如，已经被打击过的导弹，action space会收rest_ammo影响）
    #         # 若火力通道锁定，完全打击完之前，不能释放火力通道。
    #         # todo（可能出现多个火力营同时锁定一批导弹，在未打击前都无法释放，需要确认）
    #         if monster_id in self.locked_targets and monster.last_attacked_time == 0:  ## 什么时候将导弹列入到拦截目标中的？
    #             miss_Loc_InInterception = self.locked_targets.index(monster_id)
    #             self.action_mask[miss_Loc_InInterception, :mask_space_for_one_channels] = 1
    #             self.action_mask[miss_Loc_InInterception, mask_space_for_one_channels:] = 0
    #             self.current_ammo -= mask_space_for_one_channels
    #         # list中索引某元素的位置



    def get_free_channels(self):
        '''
        获取可拦截目标在火力通道上的位置
        :return:
        '''

        locs = [loc for loc in range(self.attack_channels) if self.locked_targets[loc] is None]
        return locs


    def add_visible_monster(self, monster_id):
        '''
        增加新导弹（进入视野）
        :param monster:
        :return:
        '''
        self.visible_monsters.append(monster_id)


    def remove_visible_monster(self, monster):
        '''
        删除旧导弹（出视野）
        :param monster:
        :return:
        '''
        self.visible_monsters.remove(monster)

    def release_locked_target(self, monster_id):
        '''
        删除拦截目标
        :param monster:
        :return:
        '''
        loc = self.locked_targets.index(monster_id)
        self.locked_targets[loc] = None # 置空
        # self.action_mask[loc] = 0  # 置mask为0, 整个vector为0 [0 ~ max_fire_onebatch-1]

    def sort_monster(self, monsters):
        '''
        对视野内导弹按照某种策略排序来袭导弹顺序，可以根据距离、命中概率、重要性等排序或者训练一个排序网络
        :return:
        '''
        pass


    def select_target(self, monsters):
        # 若有多架导弹，只取命中概率最高的max_fire_onebatch架，
        # todo 拦截策略，按照概率 距离 或者攻击目标重要性等进行拦截，需要讨论

        self.visible_monsters.sort(key=lambda x: geo.get_horizontal_distance(monsters[x].now_loc, self.tower_location))# todo 按照距离排序
        locs = self.get_free_channels() # 获取空的火力通道，不改变已有的拦截目标
        for ins in self.visible_monsters: # 按照打击概率放入火力通道# ?这是按照距离加入的
            if len(locs) == 0 or ins in self.locked_targets: continue # 火力通道已满，或者目标已在拦截目标中
            loc = locs.pop(0)
            self.locked_targets[loc] = ins

        # 但是受制于弹药量，可能无法全部打击，需要mask
        for loc in range(self.attack_channels):
            monster_id = self.locked_targets[loc]
            
            if monster_id is None:  # 无拦截目标,进行下一次循环
                self.action_mask[loc, :] = 0
                self.action_mask[loc, 0] = 1  # 0-不动作
                continue
            if monsters[monster_id].last_attacked_time > 0:  # 未到冷却时间，不能打击
                self.action_mask[loc, :] = 0
                self.action_mask[loc, 0] = 1  # 0-不动作
                continue

            mask_space_for_one_channels = min(self.estimated_ammo, self.max_attack_per_channel-1) # 0-不动作，max-最大发射数量[0-4],5个
            if mask_space_for_one_channels > 0: #
                assert self.current_ammo > 0, '剩余弹药数量不应该为负数'

            if len(self.action_mask[0])-1 == mask_space_for_one_channels:  # 均可发射，一次发射量等于最大发射数量
                self.action_mask[loc, :] = 1
            elif mask_space_for_one_channels == 0:  # 无法发射
                self.action_mask[loc, :] = 0
                self.action_mask[loc, 0] = 1
            else:
                self.action_mask[loc, :(mask_space_for_one_channels+1)] = 1
                self.action_mask[loc, (mask_space_for_one_channels+1):] = 0

            self.estimated_ammo -= mask_space_for_one_channels  # 更新剩余弹药数量
            assert self.estimated_ammo >= 0, '剩余弹药数量不应该为负数'
            self.estimated_ammo = max(0, self.estimated_ammo) # 保证不为负数, 但这里不应该为负数，因为mask_space_for_one_channels是根据剩余弹药数量计算的

        # print("after action_mask:", self.action_mask)


    def get_rest_ammo(self):
        '''
        获取剩余弹药数量
        :return:
        '''
        return self.max_ammo

    def update_rest_ammo(self, action):
        '''
        更新剩余弹药数量, action-onehot vector (fire_channels x max_fire_onebatch) 0-1
        实际弹量消耗
        :param action: 该agent的动作空间，shape=(num_fire_onebatch, ), 每个元素为0-max_fire_onebatch (例如0，1，2，3，4)
        :return:
        '''
        #遍历所有火力通道的非0值，每遇到一个弹药量减1
        self.current_ammo -= np.sum(action)
        assert self.current_ammo >= 0, '剩余弹药数量不应该为负数'

        # for i in range(len(action)):
        #     assert np.sum(action[i]) == 1, '每个火力通道只能发射一次'
        #     self.current_ammo -= np.argmax(action[i])
        #     assert self.current_ammo >= 0, '剩余弹药数量不应该为负数'

        # 更新剩余估计弹药数量
        self.estimated_ammo = self.current_ammo
    
    # 获取当前观察
    def return_latest_obs(self, sight_range, position):
        obs_now = []
        
        # 大小(10,)
        obs_now.append(1) # 代表可见
        obs_now.append((self.tower_location[0]-position[0])/sight_range) # 归一化
        obs_now.append((self.tower_location[1]-position[1])/sight_range) # 归一化
        obs_now.append(self.current_ammo/self.max_ammo) # 归一化
        
        # 每一个属性值需归一化
        obs_now.append(self.attack_range / self.norm_scale["attack_range"])
        obs_now.append(self.projectile_speed / self.norm_scale["speed_of_artillery"])
        obs_now.append(self.hit_rate / self.norm_scale["hit_rate"])
        
        obs_now.append(self.fixed_reaction_time / self.norm_scale["fixed_reaction_time"])
        obs_now.append(self.max_attack_per_channel / self.norm_scale["max_fire_onebatch"])
        
        obs_now.append(self.attack_channels / self.norm_scale["firepower_channels"])
        
        return np.array(obs_now)

    # 获取当前状态,todo 考虑更新为字典形式，，方便查找
    def return_latest_state(self):
        state_now = []
        # (11,)
        state_now.append(self.tower_id)
        state_now.append(self.tower_location[0])
        state_now.append(self.tower_location[1])
        state_now.append(self.attack_range)
        state_now.append(self.projectile_speed)
        state_now.append(self.hit_rate)
        state_now.append(self.fixed_reaction_time)
        state_now.append(self.max_ammo)
        state_now.append(self.max_attack_per_channel)
        state_now.append(self.current_ammo)
        state_now.append(self.attack_channels)
        return np.array(state_now)
    
    @staticmethod
    def get_ally_num_attributes():
        ally_state_attr_names = [
            "tower_id",
            "tower_location[0]",
            "tower_location[1]",
            "attack_range",
            "speed_of_artillery",
            "hit_rate",
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