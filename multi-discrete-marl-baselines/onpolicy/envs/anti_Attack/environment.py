import math

from .defense_tower import DefenseTower
from .monster import Monster
from .monster_nest import MonsterNest
from .stronghold import Stronghold
import numpy as np
import random
from .multi_discrete import MultiDiscrete
from .utils import geo as Geo
# todo: change to gym
from gym.spaces import Discrete
import json5
import os
import matplotlib.pyplot as plt


def km_to_lat_lon_radius(lat, radius_km):
    """
    将以千米为单位的距离转换为经纬度坐标系中的差异。
    lat: 纬度
    radius_km: 半径（千米）
    """
    # 纬度每度大约对应的千米数（固定值）
    lat_per_km = 1 / 111  # 每千米对应的纬度差异
    # 经度每度对应的千米数随着纬度不同而变化
    lon_per_km = 1 / (111 * np.cos(np.radians(lat)))  # 每千米对应的经度差异
    return lat_per_km * radius_km, lon_per_km * radius_km

#环境类
class Environment():
    def __init__(self, args = None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.environ.get(
            "ANTIATTACK_ENV_CONFIG",
            os.path.join(current_dir, "env_config.json5"),
        )
        with open(config_path, "r", encoding="utf-8") as f:
            parameters = json5.load(f)

        # 默认的参数
        self.debug = True
        self.obs_agent_id = True
        self.obs_timestep_number = False
        # self.stacked_frames = getattr(args, 'stacked_frames', 1)
        self.obs_last_action = False # 是否使用上一步的动作作为观察,  todo:后续如果需要，添加
        self.add_center_xy = getattr(args, 'add_center_xy', True) # 是否添加中心坐标
        self.state_last_action = False # 是否使用上一步的动作作为状态 todo:后续如果需要，添加

        self.obs_instead_of_state = getattr(args, 'obs_instead_of_state', True)# 是否使用观察代替状态
        self.use_state_agent = getattr(args, 'use_state_agent', True)# 是否使用状态代替agent
        self.add_local_obs = getattr(args, 'add_local_obs', False)# 是否添加局部观察
        # self.add_visible_state = getattr(args, 'add_visible_state', False)# 是否添加可见状态
        # self.use_stacked_frames = getattr(args, 'use_stacked_frames', False)# 是否使用堆叠帧, 默认为False
        self.show_ui = getattr(args, 'show_ui', False)# 是否显示UI界面

        # todo
        self.episode_limit = getattr(args, 'episode_limit', 200)  # 最大step数

        self._episode_steps = 0
        self._total_steps = 0
        self.reward_negative_scale = 0.5
        # 如果reward_only_positive为False，设置负奖励的比例, 默认为0.5
        # 如果reward_only_positive为True，则忽略reward_negative_scale参数
        self.reward_only_positive = True  # 是否只有正奖励, 默认为True
        self.reward_scale = 10  # 奖励缩放因子, 默认为10

        self._seed = 2024
        # 初始化环境边界（单位：经纬度）
        # 西南角 (SW)：30.0° N, 100.0° E 西北角 (NW)：40.0° N, 100.0° E
        # 东北角 (NE)：40.0° N, 110.0° E 东南角 (SE)：30.0° N, 110.0° E
        # 一个纬度差为10度、经度差为10度的区域。
        self.border = parameters['border']

        # 设置采样时间（单位：秒）1/2/3
        self.dt = parameters['dt']

        # sum_damage_monsters: 用于归一化奖励,敌方导弹的最大数量
        self.sum_damage_monsters = 0
        # 上一时刻导弹总tnt
        self.prev_damage_monsters = 0
        # sum_ammo_towers: 用于归一化奖励,敌方火力营的最大数量
        self.sum_ammo_towers = 0
        # 上一时刻火力营总tnt
        self.prev_ammo_towers = 0

        # 标记是否赢得比赛
        self.win_counted = False
        # 标记是否输掉比赛
        self.defeat_counted = False
        self._episode_count = 0
        self._ratio_save_dir = None

        '''
        重要目标设置
        '''
        self.target_num = parameters['target_num']  # 设置重要目标数量
        self.target_points = []
        self.construct_stronghold(parameters['target_points_parems'])

        '''
        火力营设置
        '''
        self.tower_num = parameters['tower_num']  # 设置火力营数量
        self.n_agents = self.tower_num
        self.defense_towers = []
        self.construct_defense_tower(parameters['tower_params'])
        self.set_norm_scale_battalion() # 设置火力营的归一化参数
        # 计算max_firepower_channels，是所有火力营中最大的火力通道数量
        max_firepower_channels = 0 # 这个数值的计算应该放在初始化，计算一次即可
        for battalion in self.defense_towers: # 这里存疑，是否要求每个火力营的火力通道数量一样？多智能体是否可以不同，网络输出的维度不一样，需要讨论，暂时认为一样
            if battalion.attack_channels > max_firepower_channels:
                max_firepower_channels = battalion.attack_channels
        self.max_attack_channels = max_firepower_channels

        '''
        来袭导弹批次设置
        '''
                # 导弹发射位
        self.nest_num = parameters['nest_num']  # 怪兽母巢数量
        self.spawn_interval = parameters['spawn_interval']  # 导弹发射间隔
        self.missile_max_values = None
        self.monster_nests = []
        # 怪兽巢穴预设位置（从JSON5读取，训练时围绕这些坐标做高斯扰动）
        self.monster_spawn_locations = parameters.get('monster_spawn_locations', [
            [-34.7, 74], [-34.7, 75], [-32.9, 76.4], [-33, 76.6],
            [-33.1, 76.8], [-34.9, 77.3], [-35.66, 76.9]
        ])

        # 发射的导弹
        self.max_monster_batches = parameters['max_monster_batches']  # 设置来袭导弹批次数量，会设置几百枚导弹
        self.monster_types = parameters['monster_types']
        self.monsters = []

        '''
        基于每个火力营构建观察observation，参考灵异系统代码
        '''
        self.observations = [] # 由各自的火力营构建观察observation

        '''
        遍历每批导弹、火力营、重要目标构建全局状态，参考灵异系统代码
        '''
        self.state = [] # 全局状态, 由每批导弹、火力营、重要目标构建

        '''
        初始化action_mask, []
        '''
        self.action_mask = [] #
        for fire_battalion in self.defense_towers:
            self.action_mask.append(fire_battalion.action_mask[np.newaxis, :, :])
        self.action_mask = np.vstack(self.action_mask)
        # 发射0枚时，mask为1
        self.action_mask[:, :, 0] = 1
        # self.action_mask_expanded =  # 将action扩展。
        self.action_space_agent = self.max_attack_channels
        # self.defense_towers[0].max_attack_per_channel # self.max_monster_batches  # todo: 这里应该表示导弹批次，不是导弹数量，后续需要修改
        # self.action_space =  # self.max_monster_batches #  [self.max_monster_batches, self.action_space_agent]
        '''
        获取环境中各个状态维度信息
        '''
        ## enemy obs预设大小 missile_batch_num * missle_feature_dim
        # window * 最大通道数
        # self.see_missile_window = 4 # todo: 这里应该表示导弹批次，不是导弹数量，后续需要修改; 这里后续可能没用到，确认后可能会删掉。
        # self.missile_feature_dim = 8
        # ally obs预设大小 (battalion_num-1) * battalion_feature_dim
        self.ally_num = self.tower_num - 1

        if self.obs_instead_of_state:
            self.missile_feature_dim = 4 + self.target_num + 6 + len(self.monster_types)
            # 分别代表距离，相对距离x，y，是否可见，打击目标(one-hot)， 速度，剩余导弹，TNT，导弹角度，上次打击时间，被打击次数, 导弹类型(one-hot)

            # self obs预设大小 1 * self_feature_dim
            self.self_feature_dim = 9 + self.max_attack_channels + 2 * self.max_monster_batches
            # 距离，相对距离x，y， 剩余导弹，速度，射程，打击概率，固定反应时间，上一时刻action（火力通道），...,
            # 拦截列表（missile_batch_num）， 可见目标（missile_batch_num）

            self.battalion_feature_dim = 9 + self.max_attack_channels + self.max_monster_batches
            ## 距离，相对距离x，y， 剩余导弹，速度，射程，打击概率，固定反应时间，拦截列表,
            # self.max_monster_batches表示视野one-hot； 上一时刻action（火力通道）【可选】;

            # target obs预设大小 target_num * target_feature_dim
            self.target_feature_dim = 6
            # 距离，相对距离x，y，是否可见，HP（未归一化），HP归一化，重要性,
        else:
            self.missile_feature_dim = self.target_num + 6 + len(self.monster_types) + self.max_monster_batches
            # 没有了距离，相对距离x,y, 和是否可见
            self.self_feature_dim = self.max_attack_channels + 2 * self.max_monster_batches
            self.target_feature_dim = 2 # 没有了距离，相对距离x,y, 和是否可见
            self.battalion_feature_dim = 9 + self.max_attack_channels + self.max_monster_batches

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        self.state_space = []
        for i in range(self.n_agents):
            self.action_space.append(Discrete(self.max_attack_channels)) # todo: 只记录是否开火，火力通道个数
            self.observation_space.append(self.get_obs_size())
            self.share_observation_space.append(self.get_state_size())
            if self.obs_instead_of_state:
                self.state_space.append(self.get_obs_size()) # todo: 这里需要修改，暂时认为每个智能体的状态空间一样
            else:
                self.state_space.append(self.get_state_size()) # todo: 这里需要修改，暂时认为每个智能体的状态空间一样

        # if self.use_stacked_frames:
        #     self.stacked_local_obs = np.zeros((self.n_agents, self.stacked_frames, int(self.get_obs_size()[0]/self.stacked_frames)), dtype=np.float32)
        #     self.stacked_global_state = np.zeros((self.n_agents, self.stacked_frames, int(self.get_state_size()[0]/self.stacked_frames)), dtype=np.float32)


        ## 默认参数
        self.last_action = np.zeros((self.n_agents, self.action_space_agent), dtype=np.float32)

        # 渲染相关
        self.fig = None
        self.ax = None
        
    def get_env_info(self):
        info = {
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "obs_shape": self.observation_space[0][0],
            "state_shape": self.state_space[0][0] * self.n_agents, # todo: 这里需要注意，state是把obs展开的
            "action_space": self.action_space[0], # 活力通道个数，多少个离散空间，可能每个离散空间大小不同。
            "n_actions": self.action_space[0].high.max() - self.action_space[0].low.min() + 1,
            # "n_actions": self.action_space[0].num_discrete_space, # 每个火力通道的action （0～4）。
        }
        return info

    '''
    目前，obs和state都是一样的,只是state是obs的展开!
    '''
    def get_obs_missiles_size(self):
        # 返回导弹的观察空间大小
        nf_en = self.missile_feature_dim
        if self.add_center_xy:
            nf_en += 2
        return self.max_monster_batches, nf_en

    def get_state_missiles_size(self):
        # 返回导弹的状态空间大小,与obs类似
        nf_en = self.missile_feature_dim
        if self.add_center_xy:
            nf_en += 2
        return self.max_monster_batches, nf_en

    def get_obs_battalions_size(self):
        # 返回火力营的观察空间大小
        nf_al = self.battalion_feature_dim
        if self.add_center_xy:
            nf_al += 2
        nf_al = self.battalion_feature_dim
        if self.add_center_xy:
            nf_al += 2

        if self.obs_last_action:
            nf_al += self.action_space_agent

        return self.tower_num - 1, nf_al # 除了自己外的火力营数量, -1是因为自己不需要观察自己

    def get_state_battalions_size(self):
        # 返回火力营的观察空间大小
        nf_al = self.battalion_feature_dim
        if self.add_center_xy:
            nf_al += 2

        if self.obs_last_action:
            nf_al += self.action_space_agent

        return self.tower_num - 1, nf_al  # 除了自己外的火力营数量, -1是因为自己不需要观察自己


    def get_obs_target_size(self):
        # 返回重要目标的观察空间大小
        target_feats = self.target_feature_dim
        if self.add_center_xy:
            target_feats += 2
        return self.target_num, target_feats

    def get_state_target_size(self):
        # 返回重要目标的状态空间大小
        target_feats = self.target_feature_dim
        if self.add_center_xy:
            target_feats += 2
        return self.target_num, target_feats

    def get_obs_own_feats_size(self):
        # 返回自身的观察空间大小
        own_feats = self.self_feature_dim
        if self.add_center_xy:
            own_feats += 2

        return own_feats

    def get_state_own_feats_size(self):
        # 返回自身的观察空间大小
        own_feats = self.self_feature_dim
        if self.add_center_xy:
            own_feats += 2
        # if self.obs_last_action:
        #     # 如果观察上一步的动作, 则需要加上导弹的总批次，与action维度一样
        #     own_feats += self.action_space_agent

        return own_feats

    def get_obs_size(self):

        own_feats_size = self.get_obs_own_feats_size()
        n_enemies, n_enemy_feats = self.get_obs_missiles_size()
        n_allies, n_ally_feats = self.get_obs_battalions_size() # 这里无所谓，own和ally的观察空间一样，维数不变
        n_target, n_target_feats = self.get_obs_target_size()

        enemy_feats = n_enemies * n_enemy_feats
        ally_feats = n_allies * n_ally_feats
        target_feats = n_target * n_target_feats

        all_feats = enemy_feats + ally_feats + own_feats_size + target_feats

        agent_id_feats = 0
        timestep_feats = 0

        if self.obs_agent_id:
            agent_id_feats = self.n_agents
            all_feats += agent_id_feats

        # if self.obs_timestep_number:
        #     timestep_feats = 1
        #     all_feats += timestep_feats

        return [all_feats, [n_allies, n_ally_feats], [n_enemies, n_enemy_feats], [n_target, n_target_feats],
                [1, own_feats_size + agent_id_feats + timestep_feats]]


    def get_state_size(self):
        if self.obs_instead_of_state:
            return [self.get_obs_size()[0] , [self.n_agents, self.get_obs_size()[0]]]

        if self.use_state_agent: # 用全局状态代替agent状态， state为全局，所以不需要own
            n_enemies, n_enemy_feats = self.get_state_missiles_size()
            n_allies, n_ally_feats = self.get_state_battalions_size()
            n_target, n_target_feats = self.get_state_target_size()

            size = n_enemy_feats * self.max_monster_batches + n_ally_feats * self.tower_num + n_target_feats * self.target_num
            # size 是记录全局信息。
            if self.add_center_xy:
                size += 2

            timestep_feats = 0
            # if self.state_timestep_feats:
            #     timestep_feats = 1
            #     size += timestep_feats

            if self.state_last_action:
                size += self.action_space_agent

            return [size, [n_enemies, n_enemy_feats], [n_allies, n_ally_feats], [n_target, n_target_feats]]

        ## 使用agent状态, 与obs类似
        nf_al, nf_en = self.battalion_feature_dim, self.missile_feature_dim

        if self.add_center_xy:
            nf_en += 2
            nf_al += 2

        if self.state_last_action:
            nf_en += self.action_space_agent
            nf_al += self.action_space_agent

        # if self.add_visible_state:
        #     nf_al += 1
        #     nf_en += 1
        #
        # if self.add_xy_state:
        #     nf_al += 2
        #     nf_en += 2

        enemy_state = self.max_monster_batches * nf_en
        ally_state = self.tower_num * nf_al # 全局情况下，没有own，所以个数为battalion_num
        target_state = self.target_num * self.target_feature_dim

        size = enemy_state + ally_state + target_state

        obs_agent_size = 0
        timestep_state = 0

        if self.add_local_obs:  # 为什么还需要增加local_obs? 因为需要知道agent的id? todo:需要确认
            obs_agent_size = self.get_obs_size()[0]
            size += obs_agent_size

        # if self.state_timestep_feats:
        #     timestep_state = 1
        #     size += timestep_state

        return [size, [self.n_agents, nf_al],
                [self.max_monster_batches, nf_en], [self.target_num, self.target_feature_dim],
                [1, obs_agent_size + timestep_state]]

    def form_available_actions(self):
        '''
        根据环境下的self.action_mask构建可用动作， shape: [n_agents, action_space]
        每个元素是发射枚数，每个元素的取值范围是[0, max_fire_onebatch+1]
        :return: available_actions: [n_agents, action_space]
        '''
        available_actions = np.sum(self.action_mask, axis=-1)-1
        available_actions = available_actions.astype(int)
        return available_actions

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(self._seed)
        else:
            np.random.seed(seed)

    def render(self, mode="human"):
        '''
        渲染环境
        :return:
        '''
        self.ax.cla()
        # 地图整体渲染
        self.ax.plot([self.border[0][1], self.border[1][1], self.border[2][1], self.border[3][1], self.border[0][1]],
                 [self.border[0][0], self.border[1][0], self.border[2][0], self.border[3][0], self.border[0][0]],
                 'b-')

        # 重要目标渲染, 重要目标用+表示,散点图
        self.ax.scatter([target.location[1] for target in self.target_points],
                     [target.location[0] for target in self.target_points], marker='*',s=200)
        for target in self.target_points:
            self.ax.annotate(f'{target.stronghold_id}', (target.location[1], target.location[0]), fontsize=15, color='r')

        # 火力营渲染
        self.ax.scatter([fire_battalion.tower_location[1] for fire_battalion in self.defense_towers],
                 [fire_battalion.tower_location[0] for fire_battalion in self.defense_towers], marker='+',s=100)
        for fire_battalion in self.defense_towers:
            self.ax.annotate(f'{fire_battalion.tower_id}', (fire_battalion.tower_location[1], fire_battalion.tower_location[0]), fontsize=15, color='r')
            lat = fire_battalion.tower_location[0]
            lon = fire_battalion.tower_location[1]
            radius_km = fire_battalion.attack_range
            # 将千米半径转换为经纬度差异
            lat_radius, lon_radius = km_to_lat_lon_radius(lat, radius_km)

            # 使用圆绘制对应的范围，注意需要分别处理经度和纬度差异
            circle = plt.Circle((lon, lat), lon_radius, color='grey', fill=False, linewidth=0.5, linestyle='--')
            self.ax.add_patch(circle)

        # 导弹发射位渲染
        self.ax.scatter([nest.location[1] for nest in self.monster_nests],
                 [nest.location[0] for nest in self.monster_nests], marker='o', s=30)

        # 导弹渲染，画出导弹从发射位到当前位置的轨迹
        for monster in self.monsters:
            if monster is not None and monster.now_loc is not None:
                self.ax.plot([monster.start_loc[1], monster.now_loc[1]], [monster.start_loc[0], monster.now_loc[0]], 'b-.', linewidth=0.1)
                # plt.plot(monster.now_loc[1], monster.now_loc[0], 'bo')

            # 如果导弹进入射程，标记出导弹与火力营的直线
            for fire_b in self.defense_towers:
                try:
                    dist = Geo.get_horizontal_distance(monster.now_loc, fire_b.tower_location)
                except Exception:
                    continue

                if dist < fire_b.attack_range:
                    self.ax.plot([monster.now_loc[1], fire_b.tower_location[1]], [monster.now_loc[0], fire_b.tower_location[0]], 'r-', linewidth=0.2)
                    self.ax.annotate(f'{monster.monster_id}', (monster.now_loc[1], monster.now_loc[0]), fontsize=5, color='b')

                if monster.monster_id in fire_b.locked_targets:
                    self.ax.plot([monster.now_loc[1], fire_b.tower_location[1]], [monster.now_loc[0], fire_b.tower_location[0]], 'g-', linewidth=0.4)
                    self.ax.annotate(f'{monster.monster_id}', (monster.now_loc[1], monster.now_loc[0]), fontsize=15, color='y')

        plt.pause(0.1)



    def high_speed(self, ignore=True):
        '''
        加速模式，导弹从发射阵地出发到接近其中agent的射程 这段时间内的导弹轨迹
        params: ignore: 是否忽略导弹从发射到进入射程之间的迭代轨迹， 在训练过程中默认为True，加速训练；可视化时可设置为False,以显示中间过程
        '''
        buffer_dist = 0.1 # 0.1km 缓冲， 保证导弹没有进入射程，需要保证最快导弹快要进入视野内
        time_m2range = np.inf # 更新时间间隔
        _hs_iter = 0
        while True:
            _hs_iter += 1
            if _hs_iter > 5000:
                print(f"[WARN] high_speed stuck after {_hs_iter} iters, monsters={len(self.monsters)}/{self.max_monster_batches}")
                break
            mark = False # 标记是否有满足导弹进入缓冲距离（buffer_dist）

            # 更新导弹位置和状态
            for nest in self.monster_nests:
                if nest.ready_to_spawn():
                    monsters = nest.spawn_monsters(len(self.monsters), self.target_points)
                    if monsters is None or len(monsters) == 0:
                        continue
                    # for mm in monsters:
                    #     if mm is not None and mm.now_loc is None:
                    #         print(f'monster-{mm.monster_id} is None')

                    self.monsters.extend(monsters)

            if ignore and len(self.monsters) == self.max_monster_batches: # 忽略导弹从发射到进入射程之间的迭代轨迹, 已发射完毕
                for monster in self.monsters:
                    for fire_b in self.defense_towers:
                        try:
                            dist = Geo.get_horizontal_distance(monster.now_loc, fire_b.tower_location)
                        except Exception as e:
                            print(e)
                            continue
                        t = math.ceil((dist - (fire_b.attack_range + buffer_dist)) * 1000 / monster.speed)
                        # 单位转换为s,且向上取整，导弹会到达射程末端的缓冲区
                        if t < time_m2range:
                            time_m2range = t

                for monster in self.monsters: # 更新导弹位置
                    monster.update_position(time_m2range)

                mark = True
            else:
                for monster in self.monsters:
                    for fire_b in self.defense_towers:
                        try:
                            dist = Geo.get_horizontal_distance(monster.now_loc, fire_b.tower_location)
                        except Exception as e:
                            print(e)
                            continue
                        if dist < fire_b.attack_range + buffer_dist: # todo: 0.1km 缓冲， 保证导弹没有进入射程，需要保证最快导弹要进入视野内时，所有导弹都已发射
                            mark = True

            if mark == False:
                for nest in self.monster_nests:
                    # 更新导弹位置和状态， 发射阵地更新时间，在发射完毕之前，不断已dt更新发射阵地的时间，直到发射完毕。
                    nest.update_spawn_cooldown(self.dt)

                for monster in self.monsters:
                    monster.update_position(self.dt)
                    # if monster.now_loc is None:
                    #     print(f'monster-{monster.monster_id} is None')
            else:
                # 计算发射所有导弹当量，用于计算max_reward
                for monster in self.monsters:
                    if monster is not None:
                        self.sum_damage_monsters += monster.damage * monster.hp
                self.prev_damage_monsters = self.sum_damage_monsters
                for battalion in self.defense_towers:
                    self.sum_ammo_towers += battalion.max_ammo
                self.prev_ammo_towers = self.sum_ammo_towers
                break

    def get_missile(self, id):
        # 根据missile ID值获取对应对象
        return self.monsters[id]

    def get_target(self, id):
        # 根据target ID值获取对应对象
        return self.target_points[id]

    def get_battalion(self, id):
        # 根据battalion ID值获取对应对象
        return self.defense_towers[id]

    # def get_index(self,id):
    #     temp = id[0]
    #     if temp == 'm':
    #         parts = id.split("m-")
    #         if len(parts) > 1 and parts[1].isdigit():
    #             return int(parts[1])
    #         else:
    #             return None
    #     elif temp == 't':
    #         parts = id.split("t-")
    #         if len(parts) > 1 and parts[1].isdigit():
    #             return int(parts[1])
    #         else:
    #             return None
    #     elif temp == 'b':
    #         parts = id.split("b-")
    #         if len(parts) > 1 and parts[1].isdigit():
    #             return int(parts[1])
    #         else:
    #             return None

    def construct_stronghold(self, target_points_parems):
        '''
        构建重要目标
        :return:
        '''
        try:
            assert len(target_points_parems) == self.target_num
            for i in range(self.target_num):
                target = Stronghold(target_points_parems[i], i)
                self.target_points.append(target)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'重要目标数量不匹配: {e}')

    def construct_defense_tower(self, battalion_params):
        '''
        构建火力营, 分为不同类型的火力营
        :param battalion_params:
        :return:
        '''
        try:
            total_length = sum(map(len, [params.get('location', []) for params in battalion_params]))
            assert self.tower_num == total_length

            i = 0  # 火力营id
            for battalion_param in battalion_params:
                for loc in battalion_param['location']:
                    b = DefenseTower(battalion_param, i, loc)
                    self.defense_towers.append(b)
                    i += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'火力营数量不匹配: {e}')

    def construct_monster_nests(self):
        '''
        构建怪兽母巢
        :return:
        '''
        self.monster_nests = []
        try:
            for i in range(self.nest_num):
                nest_params = dict()
                # 生成高斯分布的随机位置
                location = self.random_missile_launch_location(30,40,100,110)
                nest_params['location'] = location
                nest_params['quantity'] = int(self.max_monster_batches/self.nest_num)
                nest_params['spawn_interval'] = self.spawn_interval
                nest_params['monster_types'] = self.monster_types
                nest_params['dt'] = self.dt
                nest_params['nest_id'] = i
                nest_params['stronghold_num'] = self.target_num
                m = MonsterNest(nest_params)
                self.monster_nests.append(m)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'怪兽母巢构建失败: {e}')

    def construct_monsters(self):
        '''
            构建来袭导弹
            :return:
                '''
        self.monsters = []
        for nest in self.monster_nests:
            # print(f'missile_launch-{nest.nest_id} nest.ready_to_spawn():{nest.ready_to_spawn()}')
            if nest.ready_to_spawn():
                monster_batch = nest.spawn_monsters(len(self.monsters), self.target_points)
                if monster_batch is not None:
                    self.monsters.extend(monster_batch)


    def random_missile_launch_location(self, min_x=None, max_x=None, min_y=None, max_y=None):
            '''
            随机生成导弹的位置，基于JSON5预设位置做高斯扰动
            :return:
            '''
            locations = np.array(self.monster_spawn_locations)
            mean_x = np.mean(locations[:, 0])
            mean_y = np.mean(locations[:, 1])
            std_dev_x = np.std(locations[:, 0])
            std_dev_y = np.std(locations[:, 1])
            min_x = min(locations[:, 0])
            max_x = max(locations[:, 0])
            min_y = min(locations[:, 1])
            max_y = max(locations[:, 1])

            while True:
                x = random.gauss(mean_x, std_dev_x)
                y = random.gauss(mean_y, std_dev_y)
                if min_x <= x <= max_x and min_y <= y <= max_y:
                    # 生成坐标
                    coord = (x, y)
                    # 精度为小数后一位
                    rounded_coord = (round(coord[0], 1), round(coord[1], 1))
                    return rounded_coord

    def reset(self):
        '''
        重置环境， 包括重置所有的目标、导弹、火力营，重置全局状态，为每个agent分配最近的导弹，形成各自的
        Id命名规则：目标id: 't-Id', 导弹id: 'm-Id', 火力营id: 'b-Id', 火力架id: 'a-Id'
        :return:
        '''
        # 重置全局变量
        self._episode_steps = 0
        self._total_steps = 0
        self.sum_damage_monsters = 0
        self.sum_ammo_towers = 0
        self.prev_damage_monsters = 0 # 上一时刻导弹总tnt, 用于归一化奖励
        self.prev_ammo_towers = 0 # 上一时刻火力营总tnt, 用于归一化奖励
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents, self.action_space_agent), dtype=np.float32)

        # 重置重要目标参数
        for target in self.target_points:
            target.reset_stronghold()

        # 重置导弹发射位
        self.construct_monster_nests()

        # 重置导弹, 发射位置随机
        self.construct_monsters()
        self.set_norm_scale_missiles()

        # 重置火力营, 火力营位置固定，数量固定，弹药量固定，火力架数量固定
        for battalion in self.defense_towers:
            battalion.reset_defense_tower()

        # 重置action_mask
        self.action_mask = []
        for fire_battalion in self.defense_towers:
            self.action_mask.append(fire_battalion.action_mask[np.newaxis, :, :])
        self.action_mask = np.vstack(self.action_mask)
        # 发射0枚时，mask为1
        self.action_mask[:, :, 0] = 1

        if self.show_ui:
            self.close_ui()
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.render()
        #todo Reward重置

        '''
        快进
        '''
        self.high_speed() # 高速模拟,导弹都没有进入射程
        available_actions = np.zeros((self.n_agents, self.action_space_agent), dtype=np.float32)

        if self.use_state_agent:
            global_state = self.get_state()
        else:
            global_state = None

        local_obs = self.get_obs()

        return local_obs, global_state, available_actions


    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
            return obs_concat

        '''
        将全局信息载入，主要包括，monster, target, fire_battalion，通常算法会用obs代替下面state，下面代码并未进行详细勘查
        '''

        # 获取各类目标的特征维度
        _, enemy_feats_dim = self.get_state_missiles_size()
        _, agent_own_feats_dim = self.get_state_own_feats_size() # 每个agent观测都可以视为self
        _, target_feats_dim = self.get_state_target_size()

        enemy_feats = np.zeros([self.max_monster_batches, enemy_feats_dim], dtype=np.float32)
        agent_feats = np.zeros([self.n_agents, agent_own_feats_dim], dtype=np.float32)
        target_feats = np.zeros([self.target_num, target_feats_dim], dtype=np.float32)

        x_coords = [point[0] for point in self.border]
        y_coords = [point[1] for point in self.border]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        max_x_distance = max(x_coords) - min(x_coords)
        max_y_distance = max(y_coords) - min(y_coords)

        # 对所有导弹进行观察
        for monster in self.monsters:
            if monster is not None and monster.now_loc is not None:  # 导弹还在,但可能已经打击目标， 需要有个标记位
                state_m = np.array([0.0] * enemy_feats_dim, dtype=np.float32) #

                index = 0  # # 用于记录导弹特征的索引
                # 攻击目标
                state_m[index: (index + self.target_num)] = self.one_hot_encode(monster.stronghold_id, self.target_num)
                index += self.target_num
                # 导弹类型
                state_m[index: (index + len(self.monster_types))] = self.one_hot_encode(monster.monster_type,
                                                                                       len(self.monster_types))
                index += len(self.monster_types)
                # 速度
                state_m[index] = Geo.speed_m2Mach(monster.speed)
                # 剩余导弹，归一化
                state_m[index + 1] = monster.hp / monster.max_hp
                # TNT当量
                state_m[index + 2] = monster.damage / 100  # TNT 缩小100倍，防止数值过大 todo: check TNT
                # 导弹角度,需转换为0~2 pi，
                state_m[index + 3] = monster.angle / 360 * 2 * np.pi  # 导弹角度, 0-360, 转换为弧度
                # 导弹上次打击时间
                state_m[index + 4] = monster.cooldown_remaining / 60  # 转换为分钟
                # 被打击次数
                state_m[index + 5] = monster.times_hit  # 被打击次数
                ## 相对于中心的位置, 放到最后位置
                if self.add_center_xy:
                    state_m[index + 6] = (monster.now_loc[0] - center_x) / max_x_distance  # x轴相对位置,相对于中心
                    state_m[index + 7] = (monster.now_loc[1] - center_y) / max_y_distance  # y轴相对位置,相对于中心

                # 赋值给missile_state
                enemy_feats[monster.monster_id] = state_m

        # 对火力营进行观察
        # 去掉自己外的火力营
        units = [other_unit for other_unit in self.defense_towers]
        for i, other_unit in enumerate(units):
            state_b = np.array([0.0] * agent_own_feats_dim, dtype=np.float32)
            index = 0
            # 上个时刻的action，大小为火力通道数
            state_b[index:(index + self.max_attack_channels)] = self.last_action[other_unit.tower_id]
            index = self.max_attack_channels
            # 火力营剩余导弹
            state_b[
                index] = other_unit.current_ammo / other_unit.max_ammo  # 这里需要注意，有的火力营可能会出现负值，需要debug in fire_battalion
            # 导弹拦截速度,转换为马赫数
            state_b[index + 1] = Geo.speed_m2Mach(other_unit.projectile_speed)
            # 导弹拦截范围
            state_b[index + 2] = other_unit.attack_range / other_unit.norm_scale['attack_range']
            # 导弹拦截概率
            state_b[index + 3] = other_unit.hit_rate
            # 固定反应时间（min）
            state_b[index + 4] = other_unit.fixed_reaction_time / 60
            index += 5
            # 拦截列表， 也就是available_actions
            state_b[index: (index + self.max_monster_batches)] = self.one_hot_encode(other_unit.locked_targets,
                                                                                   self.max_monster_batches)  # 拦截列表， 也就是available_actions
            index += self.max_monster_batches
            # 可见导弹
            state_b[index+self.max_monster_batches: index + self.max_monster_batches *2] = self.one_hot_encode(other_unit.insght_missile, self.max_monster_batches) # 视野列表

            index += self.max_monster_batches
            if self.add_center_xy:
                state_b[index] = (other_unit.tower_location[0] - center_x) / max_x_distance  # x轴相对位置,相对于中心
                state_b[index + 1] = (other_unit.tower_location[1] - center_y) / max_y_distance  # y轴相对位置,相对于中心

            agent_feats[i] = state_b

            # 对重点目标进行观察
            for i, target in enumerate(self.target_points):
                state_t = np.array([0.0] * target_feats_dim, dtype=np.float32)
                index = 0

                # 重要目标HP，
                state_t[index] = target.hp if target.max_hp == 0 else target.hp / target.max_hp
                # 重要性
                state_t[index + 1] = target.importance

                index =+ 2
                if self.add_center_xy:
                    state_t[index] = (target.location[0] - center_x) / max_x_distance
                    state_t[index + 1] = (target.location[1] - center_y) / max_y_distance

                target_feats[i] = state_t

            global_info = []
            # 添加全局信息
            if self.obs_timestep_number:
                global_info.append(1)

            global_info = np.array(global_info, dtype=np.float32)

            state = np.concatenate((agent_feats.flatten(),
                                    enemy_feats.flatten(),
                                    target_feats.flatten(),
                                    global_info.flatten(),
                                   )).astype(np.float32)
            return state

    # def get_state_agent(self, agent_id):
    #     '''
    #     Returns observation for agent_id. The observation is composed of:
    #
    #     - enemy features (monsters features from return_latest_state)
    #     - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
    #     - target features (target features from return_latest_state)
    #
    #     All of this information is flattened and concatenated into a list,
    #     in the aforementioned order. To know the sizes of each of the
    #     features inside the final list of features, take a look at the
    #     functions ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` ,
    #     ``get_obs_target_feats_size()``, and
    #     ``get_obs_own_feats_size()``.
    #
    #     The size of the observation vector may vary, depending on the
    #     environment configuration and type of units present in the map.
    #     For instance, non-Protoss units will not have shields, movement
    #     features may or may not include terrain height and pathing grid,
    #     unit_type is not included if there is only one type of unit in the
    #     map etc.).
    #
    #     NOTE: Agents should have access only to their local observations
    #     during decentralised execution.
    #
    #     获取agent的状态
    #     :param agent_id:
    #     :return:
    #     '''
    #
    #     if self.obs_instead_of_state:
    #         obs_concat = np.concatenate(self.get_obs_agent(agent_id), axis=0).astype(np.float32)
    #         return obs_concat
    #
    #     '''
    #     暂时以下代码需要确认，与get_state_size不对应。
    #     '''
    #
    #     unit = self.defense_towers[agent_id]
    #
    #     # 获取各类目标的特征维度
    #     enemy_feats_dim = self.get_obs_missiles_size()
    #     ally_feats_dim = self.get_obs_battalions_size()
    #     target_feats_dim = self.get_obs_target_size()
    #     own_feats_dim = self.get_obs_own_feats_size()
    #
    #     enemy_feats  = np.zeros(enemy_feats_dim, dtype=np.float32)
    #     ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
    #     target_feats = np.zeros(target_feats_dim, dtype=np.float32)
    #     own_feats = np.zeros(own_feats_dim, dtype=np.float32)
    #     agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
    #
    #     x_coords = [point[0] for point in self.border]
    #     y_coords = [point[1] for point in self.border]
    #     center_x = sum(x_coords) / len(x_coords)
    #     center_y = sum(y_coords) / len(y_coords)
    #     max_x_distance = max(x_coords) - min(x_coords)
    #     max_y_distance = max(y_coords) - min(y_coords)
    #
    #     # 火力营还有导弹
    #     if unit.current_ammo > 0:
    #         # 对导弹进行观察
    #         for monster in self.monsters:
    #             if monster is not None and monster.now_loc is not None: # 导弹还在,但可能已经打击目标， 需要有个标记位
    #                 state_m = np.array([0.0] * enemy_feats_dim[1], dtype=np.float32)
    #
    #                 dist = Geo.get_horizontal_distance(monster.now_loc, unit.tower_location) # 返回的是km
    #                 dist_x = Geo.get_horizontal_distance(monster.now_loc, [monster.now_loc[0], unit.tower_location[1]]) # x轴相对距离
    #                 dist_y = Geo.get_horizontal_distance(monster.now_loc, [unit.tower_location[0], monster.now_loc[1]]) # y轴相对距离
    #                 state_m[0] = dist/unit.attack_range
    #                 state_m[1] = dist_x/unit.attack_range
    #                 state_m[2] = dist_y/unit.attack_range
    #
    #                 if dist < unit.attack_range:
    #                     state_m[3] = 1 # visible
    #
    #                 index = 4 # # 用于记录导弹特征的索引
    #                 # 攻击目标
    #                 state_m[index: (index+self.target_num)] = self.one_hot_encode(monster.stronghold_id, self.target_num)
    #                 index += self.target_num
    #                 # 导弹类型
    #                 state_m[index: (index+len(self.monster_types))] = self.one_hot_encode(monster.monster_type, len(self.monster_types))
    #                 index += len(self.monster_types)
    #                 # 速度
    #                 state_m[index] = Geo.speed_m2Mach(monster.speed)
    #                 # 剩余导弹，归一化
    #                 state_m[index+1] = monster.hp / monster.max_hp
    #                 # TNT当量
    #                 state_m[index+2] = monster.damage/100 # TNT 缩小100倍，防止数值过大 todo: check TNT
    #                 # 导弹角度,需转换为0~2 pi，
    #                 state_m[index+3] = monster.angle/360 * 2 * np.pi # 导弹角度, 0-360, 转换为弧度
    #                 # 导弹上次打击时间
    #                 state_m[index+4] = monster.cooldown_remaining/60 # 转换为分钟
    #                 # 被打击次数
    #                 state_m[index+5] = monster.times_hit # 被打击次数
    #                 ## 相对于中心的位置, 放到最后位置
    #                 if self.add_center_xy:
    #                     state_m[index+6] = (monster.now_loc[0] - center_x) / max_x_distance  # x轴相对位置,相对于中心
    #                     state_m[index+7] = (monster.now_loc[1] - center_y) / max_y_distance  # y轴相对位置,相对于中心
    #
    #                 # 赋值给missile_state
    #                 enemy_feats[monster.monster_id] = state_m
    #
    #         # 对火力营进行观察
    #         # 去掉自己外的火力营
    #         other_units = [other_unit for other_unit in self.defense_towers if other_unit.tower_id != unit.tower_id]
    #         for i, other_unit in enumerate(other_units):
    #             state_b = np.array([0.0] * ally_feats_dim[1], dtype=np.float32)
    #
    #             dist = Geo.get_horizontal_distance(other_unit.tower_location, unit.tower_location) # 火力营与本火力营的距离
    #             dist_x = Geo.get_horizontal_distance(other_unit.tower_location, [other_unit.tower_location[0], unit.tower_location[1]]) # x轴相对距离
    #             dist_y = Geo.get_horizontal_distance(other_unit.tower_location, [unit.tower_location[0], other_unit.tower_location[1]]) # y轴相对距离
    #             state_b[0] = dist/unit.attack_range
    #             state_b[1] = dist_x/unit.attack_range
    #             state_b[2] = dist_y/unit.attack_range
    #
    #             if dist < unit.attack_range: # 是否在射程内，意为是否可协同打击
    #                 state_b[3] = 1 # visible
    #
    #             # 上个时刻的action，大小为火力通道数
    #             state_b[4:(4 + self.max_attack_channels)] = self.last_action[other_unit.tower_id]
    #             index = 4 + self.max_attack_channels
    #             # 火力营剩余导弹
    #             state_b[index] = other_unit.current_ammo / other_unit.max_ammo # 这里需要注意，有的火力营可能会出现负值，需要debug in fire_battalion
    #             # 导弹拦截速度,转换为马赫数
    #             state_b[index + 1] = Geo.speed_m2Mach(other_unit.projectile_speed)
    #             # 导弹拦截范围
    #             state_b[index + 2] = other_unit.attack_range/other_unit.norm_scale['attack_range']
    #             # 导弹拦截概率
    #             state_b[index + 3] = other_unit.hit_rate
    #             # 固定反应时间（min）
    #             state_b[index + 4] = other_unit.fixed_reaction_time/60
    #             index += 5
    #             # 拦截列表， 也就是available_actions
    #             state_b[index: (index+self.max_monster_batches)] = self.one_hot_encode(other_unit.locked_targets, self.max_monster_batches) # 拦截列表， 也就是available_actions
    #             # 可见导弹
    #             # state_b[9+self.max_monster_batches: 9 + self.max_monster_batches *2] = self.one_hot_encode(other_unit.insght_missile, self.max_monster_batches) # 视野列表
    #
    #             index += self.max_monster_batches
    #             if self.add_center_xy:
    #                 state_b[index] = (other_unit.tower_location[0] - center_x) / max_x_distance # x轴相对位置,相对于中心
    #                 state_b[index+1] = (other_unit.tower_location[1] - center_y) / max_y_distance # y轴相对位置,相对于中心
    #
    #             ally_feats[i] = state_b
    #
    #         # 对重点目标进行观察
    #         for i, target in enumerate(self.target_points):
    #             state_t = np.array([0.0] * target_feats_dim[1], dtype=np.float32)
    #
    #             dist = Geo.get_horizontal_distance(target.location, unit.tower_location) # 重要目标与火力营的距离
    #             dist_x = Geo.get_horizontal_distance(target.location, [target.location[0], unit.tower_location[1]]) # x轴相对距离
    #             dist_y = Geo.get_horizontal_distance(target.location, [unit.tower_location[0], target.location[1]]) # y轴相对距离
    #             state_t[0] = dist/unit.attack_range
    #             state_t[1] = dist_x/unit.attack_range
    #             state_t[2] = dist_y/unit.attack_range
    #
    #             if dist < unit.attack_range: # 是否在射程内，意为是否可直接拦截
    #                 state_t[3] = 1 # visible
    #
    #             # 重要目标HP，
    #             state_t[4] = target.hp if target.max_hp == 0 else target.hp/target.max_hp
    #             # 重要性
    #             state_t[5] = target.importance
    #
    #             index = 6
    #             if self.add_center_xy:
    #                 state_t[index] = (target.location[0] - center_x) / max_x_distance
    #                 state_t[index+1] = (target.location[1] - center_y) / max_y_distance
    #
    #             target_feats[i] = state_t
    #
    #         # 对自身进行观察
    #         own_feats[:3] = 0 # 对应火力营特征，前三个特征为距离，相对x，相对y，直接置0
    #         own_feats[3] = 1 # visible
    #         own_feats[4: 4 + self.max_attack_channels] = self.last_action[unit.tower_id]  # 上个时刻的action
    #         index = 4 + self.max_attack_channels
    #         own_feats[index] = unit.current_ammo / unit.max_ammo # 剩余导弹
    #         own_feats[index + 1] = Geo.speed_m2Mach(unit.projectile_speed) # 导弹拦截速度
    #         own_feats[index + 2] = unit.attack_range/unit.norm_scale['attack_range'] # 导弹拦截范围
    #         own_feats[index + 3] = unit.hit_rate # 导弹拦截概率
    #         own_feats[index + 4] = unit.fixed_reaction_time/60 # 固定反应时间（min）
    #         index += 5
    #         own_feats[index: (index+self.max_monster_batches)] = self.one_hot_encode(unit.locked_targets, self.max_monster_batches) # 拦截列表， 也就是available_actions
    #         own_feats[index + self.max_monster_batches: index + self.max_monster_batches * 2] = self.one_hot_encode(unit.visible_monsters, self.max_monster_batches) # 视野列表
    #         index = index + self.max_monster_batches * 2
    #
    #         if self.add_center_xy:
    #             own_feats[index] = (unit.tower_location[0] - center_x) / max_x_distance
    #             own_feats[index + 1] = (unit.tower_location[1] - center_y) / max_y_distance
    #
    #     state = np.concatenate((ally_feats.flatten(),
    #                             enemy_feats.flatten(),
    #                             target_feats.flatten(),
    #                             own_feats.flatten())).astype(np.float32)
    #
    #     # 添加全局信息
    #
    #     # 添加agent_id?
    #     if self.obs_agent_id:
    #         agent_id_feats[agent_id] = 1
    #         state = np.concatenate((state, agent_id_feats.flatten())).astype(np.float32)
    #
    #     # if self.debug:
    #     #     print(f"obs state shape: {state.shape}")
    #         # print(f"enemy_feats shape: {enemy_feats.shape}")
    #         # print(f"ally_feats shape: {ally_feats.shape}")
    #         # print(f"target_feats shape: {target_feats.shape}")
    #         # print(f"own_feats shape: {own_feats.shape}")
    #         # print(f"agent_id_feats shape: {agent_id_feats.shape}")
    #
    #     return state

    def mask_actions(self, actions):
        '''
        actions 输入为 shape [n_agents]，每个元素为0-4的离散值，表示使用哪个火力通道（0=不开火，1-4=第1-4个通道开火1发）
        转换为 [n_agents, attack_channels] 的格式供 perform_action 使用
        '''
        actions = np.array(actions).astype(int)
        actions = actions.reshape(-1)  # [n_agents]
        n_agents = actions.shape[0]

        # 将单个离散动作转换为 per-channel 格式
        per_channel_actions = np.zeros((n_agents, self.action_space_agent), dtype=int)
        for i in range(n_agents):
            a = actions[i]
            if a > 0 and a <= self.action_space_agent:
                per_channel_actions[i, a - 1] = 1

        # 根据 action_mask 限制不可用的动作
        action_mask_index = np.sum(self.action_mask, axis=-1) - 1
        per_channel_actions = np.minimum(per_channel_actions, action_mask_index).astype(int)

        return per_channel_actions

    def _save_episode_ratio(self, defense_ratio):
        import csv, os
        save_dir = self._ratio_save_dir or "."
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, "episode_interception_ratios.csv")
        file_exists = os.path.isfile(filepath)
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["episode", "interception_ratio", "total_steps"])
            writer.writerow([self._episode_count + 1, defense_ratio, self._total_steps])
        self._episode_count += 1

    def step(self, action):
        '''
        先执行动作，再基于dt更新所有状态，返回新的状态和观察
        :param action: shape [火力营数量，火力通道数量, 单次最大发射量]
        :return:
        '''
        # mask actions that are not available in the current state
        action = self.mask_actions(action) # 将不可执行的动作置为0

        info = [{} for _ in range(self.n_agents)] # 用于存储额外信息
        dones = np.zeros(self.n_agents, dtype=bool) # 用于判断是否结束

        self.last_action = action # 保存上一步的动作

        # 根据action执行动作
        for a in range(len(action)): # 遍历每个火力营
            self.defense_towers[a].perform_action(action[a],self.monsters) # 执行拦截动作，在火力营层面更新信息

        # 经过dt时间后，全局信息更新
        # 更新导弹发射位状态, 发射后更新时间
        if self.max_monster_batches < len(self.monsters): # 如果导弹还没有发射完毕，则继续发射，不然直接跳过发射过程。
            for nest in self.monster_nests:
                if nest.quantity <= 0: # 发射完毕
                    continue

                if nest.ready_to_spawn():
                    batch_monsters = nest.spawn_monsters(len(self.monsters), self.target_points)
                    if batch_monsters is not None:
                        self.monsters.extend(batch_monsters)
                nest.update_spawn_cooldown(self.dt)

        # 更新导弹位置和状态
        for monster in self.monsters:
            if monster is None or monster.now_loc is None: # 被击落 或已完成攻击任务
                continue
            monster.update_position(self.dt)
        self.update_missile_state() # 更新导弹状态, 例如是否击中目标, 击中目标则直接更新目标状态, 更新冷却时间

        # 经过dt时间后,更新火力营状态, 注意，执行打击之后，原来的导弹可能已经被击落，所以需要重新更新视野和拦截目标
        self.update_fire_battalion_state()

        self._total_steps += 1
        self._episode_steps += 1

        # 两种结束方式: 1.所有来袭导弹都被击落或者都到达目标 2.所有huoliying都没有导弹
        # 判断是否结束， 来袭目标用完, dones: 用于表示每个agent是否结束
        # 为每个agent评估奖励及损失
        reward, defense_ratio = self.evaluate_reward()  # todo 成功拦截导弹，浪费导弹负值，重要目标被攻击
        done, done_mark, dones = self.check_done(dones)

        # print(f'reward:{reward}')
        available_actions = self.form_available_actions() # 需要与action_mask对应,获得available_actions

        # todo: 结算奖励, 重要目标是否被打击完 or 拦截弹用完 or 来袭目标用完
        if done and not self.defeat_counted: # 游戏结束, self.win_counted, self.defeat_counted用于标记是否已经计算过胜利或者失败,只要有标记就表示结束，两者只有一个是True
            # 清算奖励，基于目标被打击的情况
            rewards_ending, defense_ratio = self.Settlement_Reward(done_mark)
            reward += rewards_ending
            print(f'{self._episode_steps} step: reward:{reward}')
            print(f'{self._episode_steps} step: interception ratio: {defense_ratio}')
            # Save per-episode interception ratio
            self._save_episode_ratio(defense_ratio)
            self.defeat_counted = True
        elif done and self.defeat_counted: # 游戏结束，并已设置标志位，不再计算奖励
            reward = 0


        for ai in range(self.n_agents):
            info[ai] = {
                "locked_targets": len([x for x in self.defense_towers[ai].locked_targets if x is not None]),
                "killed_monsters": self.defense_towers[ai].locked_targets,
                "visible_monsters": len(self.defense_towers[ai].visible_monsters),
                "rest_ammo": self.defense_towers[ai].current_ammo,
                "done": dones[ai], # 记录每个agent是否还有剩余弹药，没有则为True
                "reward": reward * self.reward_scale,
                "individual_reward": self.defense_towers[ai].get_reward()/self.sum_damage_monsters * self.reward_scale,
                "defense_ratio": defense_ratio, # 记录总的拦截信息
            }

        if self.use_state_agent:
            global_state = self.get_state() # 代码默认使用obs
            # global_state = self.update_global_state()
        else:
            global_state = None

        local_obs = self.get_obs()

        rewards = reward * self.reward_scale * np.ones(self.n_agents)  # per-agent rewards

        dones = np.array([done] * self.n_agents)

        return local_obs, global_state, rewards, dones, info, available_actions #r 单步reward

    def evaluate_reward(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """


        r_battalion_ammo = 0
        for battalion in self.defense_towers:
            r_battalion_ammo += battalion.current_ammo # 火力营的消耗弹量
            # 记录剩余弹药，用于log，展示

        delta_battalion_ammo = self.prev_ammo_towers - r_battalion_ammo
        self.prev_ammo_towers = r_battalion_ammo

        # for battalion in self.defense_towers:
        #     delta_battalion_tnt += battalion.get_reward() * 1.0  # 目前是单步
        #     # todo 确认reward是单步还是累计，需要对应处理fire_battalion
        #     battalion.reset_reward()

        shown_missiles = []
        for monster in self.monsters:
            if monster is not None:
                shown_missiles.append([monster.monster_id, monster.hp])

        r_missiles_tnt = 0
        for monster in self.monsters:
            if monster is not None: # 只记录被拦截的导弹，被拦截的会设置为0.
                r_missiles_tnt += monster.damage * monster.hp

        delta_missiles_tnt = self.prev_damage_monsters - r_missiles_tnt
        self.prev_damage_monsters = r_missiles_tnt

        return delta_missiles_tnt/self.sum_damage_monsters - delta_battalion_ammo/self.sum_ammo_towers, 1 - r_missiles_tnt/self.sum_damage_monsters


    def check_done(self, dones):
        '''
        判断是否结束， 重要目标是否被打击完 or 拦截弹用完 or 来袭目标用完
        :return:
        '''
        # 所有来袭导弹都被击落或者都到达目标
        rest_missiles = [monster for monster in self.monsters if monster is not None] # 剩余未被拦截导弹
        mark_missiles = all(x is None for x in self.monsters) # 所有来袭导弹都被击落
        mark_battalions = all(x.current_ammo <= 0 for x in self.defense_towers) # 所有火力营都没有弹药
        mark_rest_missiles = all(x.now_loc is None for x in rest_missiles) # 所有来袭导弹都到达
        for i in range(self.n_agents):
            if self.defense_towers[i].current_ammo <= 0:
                dones[i] = True

        return mark_missiles or mark_battalions or mark_rest_missiles, [mark_missiles, mark_battalions, mark_rest_missiles], dones

    def check_done_v1(self):
        '''
        判断是否结束， 使用原来版本的判断方法
        :return:
        '''
        rest_missiles = [monster for monster in self.monsters if monster is not None]  # 剩余未被拦截导弹
        return all(x is None for x in self.monsters) or all(x.now_loc is None for x in rest_missiles), []

    def update_missile_state(self):
        '''
        更新导弹状态, 例如是否击中目标, 击中目标则直接更新目标状态, 更新冷却时间
        :return:
        '''
        for monster in self.monsters:
            if monster is None:
                continue

            index = monster.monster_id
            if monster.hp == 0: # 被击落
                self.monsters[index] = None
                continue
            if monster.now_loc is None: # 到达目标, 到达目标，monster.now_loc为None,也从missiles中删除？
                target = self.get_target(monster.stronghold_id)
                target.take_damage(monster.damage)
                # self.monsters[index] = None
                continue
            monster.update_attacked_time(self.dt)  # 更新导弹时间
            monster.update_angle()  # 更新导弹角度



    def update_fire_battalion_state(self):
        '''
        更新火力营状态
        :return:
        '''
        # 更新视野数据,进入视野的导弹，添加到insight，离开视野的导弹，从insight中删除；同时视野内的导弹可能在执行阶段被拦截，所以需要更新
        # 同时更新 进入视野和离开视野的导弹，更新拦截目标

        for mloc in range(len(self.monsters)):
            monster = self.monsters[mloc]
            for battalion in self.defense_towers:
                # 如果missile为None, 肯定是被击落的，需要更新各个battalion的视野和拦截目标列表
                if monster is None:
                    if mloc in battalion.visible_monsters:
                        battalion.visible_monsters.remove(mloc)
                    if mloc in battalion.locked_targets:
                        battalion.release_locked_target(mloc)
                    continue
                # 如果missile不为None, 但now_loc为None, 说明导弹已经到达目标，需要从各个battalion的视野和拦截目标列表删除
                if monster.now_loc is None:
                    if mloc in battalion.visible_monsters:
                        battalion.visible_monsters.remove(mloc)
                    if mloc in battalion.locked_targets:
                        battalion.release_locked_target(mloc)
                    continue
                # 如果missile不为None, 且now_loc不为None, 说明导弹在飞行中，需要判断是否在视野内，以更新视野列表（添加或删除）
                dist = Geo.get_horizontal_distance(monster.now_loc, battalion.tower_location) # 计算导弹与火力营的水平距离
                if dist < battalion.attack_range: # 如果导弹在视野内
                    if mloc not in battalion.visible_monsters: # 如果导弹不在视野列表中，添加到视野内
                        battalion.add_visible_monster(mloc)
                else: # 如果导弹不在视野内，可能是离开视野（删除），也可能是从未进入视野（无需操作）
                    if mloc in battalion.visible_monsters: # 如果导弹在视野列表中，删除视野
                        battalion.visible_monsters.remove(mloc)
                    if mloc in battalion.locked_targets: # 如果导弹在拦截目标列表中，删除拦截目标
                        battalion.release_locked_target(mloc)

        # 选取拦截目标,更新env中的action_mask
        for i in range(len(self.defense_towers)):
            if self.defense_towers[i].current_ammo <= 0: # 如果火力营没有弹药，不需要更新拦截目标, action_mask只需要全0
                self.action_mask[i] = np.zeros_like(self.action_mask[i])
                self.action_mask[i][:, 0] = 1
            else:
                self.defense_towers[i].select_target(self.monsters)
                self.action_mask[i] = self.defense_towers[i].action_mask.copy()  # 直接赋值就可以

            self.defense_towers[i].reset_action_mask() # 重置action_mask
        # self.action_mask = np.vstack(self.action_mask)


    # 更新汇总全局的状态
    def update_global_state(self):
        """Returns the global state.
                NOTE: This functon should not be used during decentralised execution.
                """
        # todo 为了规避init_hp==0
        for target in self.target_points:
            if target.max_hp == 0:
                target.max_hp = 1

        # 特征的数量等于属性名的数量
        nf_battalion = DefenseTower.get_ally_num_attributes()
        nf_missile = Missile.get_enemy_num_attributes()
        nf_tar = Target.get_tar_num_attributes()

        battalion_state = np.zeros((self.tower_num, nf_battalion-1+self.tower_num)) # 去掉battalionId, 添加one-hot
        missile_state = np.zeros((self.max_monster_batches, nf_missile-1+self.max_monster_batches)) # 去掉monster_id, 添加one-hot
        target_state = np.zeros((self.target_num, nf_tar-1+self.target_num)) # 去掉targetId, 添加one-hot

        # 计算地图中间点和边界值，用来处理归一化
        x_coords = [point[0] for point in self.border]
        y_coords = [point[1] for point in self.border]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        max_x_distance = max(x_coords) - min(x_coords)
        max_y_distance = max(y_coords) - min(y_coords)

        # 遍历火力营
        for battalion in self.defense_towers:
            if battalion.current_ammo > 0:
                state_b = battalion.return_latest_state()

                # 归一化坐标
                state_b[1] = (state_b[1] - center_x) / max_x_distance
                state_b[2] = (state_b[2] - center_y) / max_y_distance
                # 计算并存储相对剩余弹药
                state_b = np.append(state_b, (battalion.current_ammo / battalion.max_ammo))

                # one-hot id值
                id_onehot = self.one_hot_encode(state_b[0], self.tower_num)
                bat_satte = np.append(id_onehot, state_b[1:])

                # 赋值给battalion_state
                battalion_state[battalion.tower_id] = bat_satte

        # todo 认为导弹不应该在global state中
        # 遍历导弹
        # for monster in self.monsters:
        #     if monster.hp > 0:
        #         state_m = monster.return_latest_state()
        #         # 归一化坐标
        #         state_m[1] = (state_m[1] - center_x) / max_x_distance
        #         state_m[2] = (state_m[2] - center_y) / max_y_distance
        #         # 计算相对剩余导弹
        #         state_m = np.append(state_m, (monster.hp / monster.max_hp))

        #         # one-hot tar_id值
        #         id_onsehot = self.one_hot_encode(state_m[0], self.max_monster_batches)
        #         state_missile = np.append(id_onsehot, state_m[1:])

        #         # 赋值给missile_state
        #         missile_state[monster.monster_id] = state_missile

        # 遍历目标
        for target in self.target_points:
            state_t = target.return_latest_state()

            # 归一化坐标
            state_t[1] = (state_t[1] - center_x) / max_x_distance
            state_t[2] = (state_t[2] - center_y) / max_y_distance
            # 计算相对剩余HP
            state_t = np.append(state_t, (target.hp / target.max_hp))
            # one-hot id值
            id_onehot = self.one_hot_encode(state_t[0], self.target_num)
            state_t = np.append(id_onehot, state_t[1:])
            # 赋值给target_state
            target_state[target.stronghold_id] = state_t

        state_dict = {"battalion": battalion_state, "monster": missile_state, "target": target_state}

        state_all = np.append(
            state_dict["battalion"].flatten(), state_dict["target"].flatten()
        )

        # 处理last_action和timestep
        # if "last_action" in state_dict:
        #     state = np.append(state, state_dict["last_action"].flatten())
        # if "timestep" in state_dict:
        #     state = np.append(state, state_dict["timestep"])

        return state_all


    def one_hot_encode(self, id_mask, total_nums):
        """
        将分类数据转换为one-hot编码。 id_mask是一个列表，包含要设置为1的索引和None。
        id_mask中的数值仅出现一次，如果id_mask中的数值为None，则返回全0向量。

        :param id_mask: 要设置为1的索引列表或者单个索引
        :param total_nums: one-hot编码的总数
        :return: one-hot编码
        """
        # 判断id_mask是列表还是数字
        if not isinstance(id_mask, list):
            id_mask = [id_mask]

        one_hot = []
        # 如果id_mask列表中全为None，则返回全0向量。
        if id_mask.count(None) == len(id_mask):
            return np.zeros(total_nums)

        # 如果id_mask列表为空，则返回全0向量。
        if len(id_mask) == 0:
            return np.zeros(total_nums)

        for i in range(total_nums):
            if i in id_mask:
                one_hot.append(1)
            else:
                one_hot.append(0)
        return np.array(one_hot)

    def assign_observations(self):
        # todo 为了规避init_hp==0
        for target in self.target_points:  ##
            if target.max_hp == 0:
                target.max_hp = 1

        self.observations = [self.get_obs_agent(i) for i in range(self.tower_num)]
        return self.observations


    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

           - agent movement features (where it can move to, height information and pathing grid)
           - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
           - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
           - agent unit features (health, shield, unit_type)

           All of this information is flattened and concatenated into a list,
           in the aforementioned order. To know the sizes of each of the
           features inside the final list of features, take a look at the
           functions ``get_obs_move_feats_size()``,
           ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
           ``get_obs_own_feats_size()``.

           The size of the observation vector may vary, depending on the
           environment configuration and type of units present in the map.
           For instance, non-Protoss units will not have shields, movement
           features may or may not include terrain height and pathing grid,
           unit_type is not included if there is only one type of unit in the
           map etc.).

           NOTE: Agents should have access only to their local observations
           during decentralised execution.
        """
        unit = self.defense_towers[agent_id]

        # 获取各类目标的特征维度
        enemy_feats_dim = self.get_state_missiles_size()
        ally_feats_dim = self.get_state_battalions_size()
        target_feats_dim = self.get_state_target_size()
        own_feats_dim = self.get_state_own_feats_size()

        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        target_feats = np.zeros(target_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)

        x_coords = [point[0] for point in self.border]
        y_coords = [point[1] for point in self.border]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        max_x_distance = max(x_coords) - min(x_coords)
        max_y_distance = max(y_coords) - min(y_coords)

        # 火力营还有导弹
        if unit.current_ammo > 0:
            # 对导弹进行观察
            for monster in self.monsters:
                if monster is not None and monster.now_loc is not None:  # 导弹还在,但可能已经打击目标， 需要有个标记位
                    state_m = np.array([0.0] * enemy_feats_dim[1], dtype=np.float32)

                    dist = Geo.get_horizontal_distance(monster.now_loc, unit.tower_location)  # 返回的是km
                    dist_x = Geo.get_horizontal_distance(monster.now_loc,
                                                         [monster.now_loc[0], unit.tower_location[1]])  # x轴相对距离
                    dist_y = Geo.get_horizontal_distance(monster.now_loc,
                                                         [unit.tower_location[0], monster.now_loc[1]])  # y轴相对距离
                    state_m[0] = dist / unit.attack_range
                    state_m[1] = dist_x / unit.attack_range
                    state_m[2] = dist_y / unit.attack_range

                    if dist < unit.attack_range:
                        state_m[3] = 1  # visible

                    index = 4  # # 用于记录导弹特征的索引
                    # 攻击目标
                    state_m[index: (index + self.target_num)] = self.one_hot_encode(monster.stronghold_id, self.target_num)
                    index += self.target_num
                    # 导弹类型
                    state_m[index: (index + len(self.monster_types))] = self.one_hot_encode(monster.monster_type,
                                                                                           len(self.monster_types))
                    index += len(self.monster_types)
                    # 速度
                    state_m[index] = Geo.speed_m2Mach(monster.speed)
                    # 剩余导弹，归一化
                    state_m[index + 1] = monster.hp / monster.max_hp
                    # TNT当量
                    state_m[index + 2] = monster.damage / 100  # TNT 缩小100倍，防止数值过大 todo: check TNT
                    # 导弹角度,需转换为0~2 pi，
                    state_m[index + 3] = monster.angle / 360 * 2 * np.pi  # 导弹角度, 0-360, 转换为弧度
                    # 导弹上次打击时间
                    state_m[index + 4] = monster.cooldown_remaining / 60  # 转换为分钟
                    # 被打击次数
                    state_m[index + 5] = monster.times_hit  # 被打击次数
                    ## 相对于中心的位置, 放到最后位置
                    if self.add_center_xy:
                        state_m[index + 6] = (monster.now_loc[0] - center_x) / max_x_distance  # x轴相对位置,相对于中心
                        state_m[index + 7] = (monster.now_loc[1] - center_y) / max_y_distance  # y轴相对位置,相对于中心

                    # 赋值给missile_state
                    enemy_feats[monster.monster_id] = state_m

            # 对火力营进行观察
            # 去掉自己外的火力营
            other_units = [other_unit for other_unit in self.defense_towers if other_unit.tower_id != unit.tower_id]
            for i, other_unit in enumerate(other_units):
                state_b = np.array([0.0] * ally_feats_dim[1], dtype=np.float32)

                dist = Geo.get_horizontal_distance(other_unit.tower_location,
                                                   unit.tower_location)  # 火力营与本火力营的距离
                dist_x = Geo.get_horizontal_distance(other_unit.tower_location, [other_unit.tower_location[0],
                                                                                     unit.tower_location[
                                                                                         1]])  # x轴相对距离
                dist_y = Geo.get_horizontal_distance(other_unit.tower_location, [unit.tower_location[0],
                                                                                     other_unit.tower_location[
                                                                                         1]])  # y轴相对距离
                state_b[0] = dist / unit.attack_range
                state_b[1] = dist_x / unit.attack_range
                state_b[2] = dist_y / unit.attack_range

                if dist < unit.attack_range:  # 是否在射程内，意为是否可协同打击
                    state_b[3] = 1  # visible

                # 上个时刻的action，大小为火力通道数
                state_b[4:(4 + self.max_attack_channels)] = self.last_action[other_unit.tower_id]
                index = 4 + self.max_attack_channels
                # 火力营剩余导弹
                state_b[
                    index] = other_unit.current_ammo / other_unit.max_ammo  # 这里需要注意，有的火力营可能会出现负值，需要debug in fire_battalion
                # 导弹拦截速度,转换为马赫数
                state_b[index + 1] = Geo.speed_m2Mach(other_unit.projectile_speed)
                # 导弹拦截范围
                state_b[index + 2] = other_unit.attack_range / other_unit.norm_scale['attack_range']
                # 导弹拦截概率
                state_b[index + 3] = other_unit.hit_rate
                # 固定反应时间（min）
                state_b[index + 4] = other_unit.fixed_reaction_time / 60
                index += 5
                # 拦截列表， 也就是available_actions
                state_b[index: (index + self.max_monster_batches)] = self.one_hot_encode(other_unit.locked_targets,
                                                                                       self.max_monster_batches)  # 拦截列表， 也就是available_actions
                # 可见导弹
                # state_b[9+self.max_monster_batches: 9 + self.max_monster_batches *2] = self.one_hot_encode(other_unit.insght_missile, self.max_monster_batches) # 视野列表

                index += self.max_monster_batches
                if self.add_center_xy:
                    state_b[index] = (other_unit.tower_location[0] - center_x) / max_x_distance  # x轴相对位置,相对于中心
                    state_b[index + 1] = (other_unit.tower_location[1] - center_y) / max_y_distance  # y轴相对位置,相对于中心

                ally_feats[i] = state_b

            # 对重点目标进行观察
            for i, target in enumerate(self.target_points):
                state_t = np.array([0.0] * target_feats_dim[1], dtype=np.float32)

                dist = Geo.get_horizontal_distance(target.location, unit.tower_location)  # 重要目标与火力营的距离
                dist_x = Geo.get_horizontal_distance(target.location,
                                                     [target.location[0], unit.tower_location[1]])  # x轴相对距离
                dist_y = Geo.get_horizontal_distance(target.location,
                                                     [unit.tower_location[0], target.location[1]])  # y轴相对距离
                state_t[0] = dist / unit.attack_range
                state_t[1] = dist_x / unit.attack_range
                state_t[2] = dist_y / unit.attack_range

                if dist < unit.attack_range:  # 是否在射程内，意为是否可直接拦截
                    state_t[3] = 1  # visible

                # 重要目标HP，
                state_t[4] = target.hp if target.max_hp == 0 else target.hp / target.max_hp
                # 重要性
                state_t[5] = target.importance

                index = 6
                if self.add_center_xy:
                    state_t[index] = (target.location[0] - center_x) / max_x_distance
                    state_t[index + 1] = (target.location[1] - center_y) / max_y_distance

                target_feats[i] = state_t

            # 对自身进行观察
            own_feats[:3] = 0  # 对应火力营特征，前三个特征为距离，相对x，相对y，直接置0
            own_feats[3] = 1  # visible
            own_feats[4: 4 + self.max_attack_channels] = self.last_action[unit.tower_id]  # 上个时刻的action
            index = 4 + self.max_attack_channels
            own_feats[index] = unit.current_ammo / unit.max_ammo  # 剩余导弹
            own_feats[index + 1] = Geo.speed_m2Mach(unit.projectile_speed)  # 导弹拦截速度
            own_feats[index + 2] = unit.attack_range / unit.norm_scale['attack_range']  # 导弹拦截范围
            own_feats[index + 3] = unit.hit_rate  # 导弹拦截概率
            own_feats[index + 4] = unit.fixed_reaction_time / 60  # 固定反应时间（min）
            index += 5
            own_feats[index: (index + self.max_monster_batches)] = self.one_hot_encode(unit.locked_targets,
                                                                                     self.max_monster_batches)  # 拦截列表， 也就是available_actions
            own_feats[index + self.max_monster_batches: index + self.max_monster_batches * 2] = self.one_hot_encode(
                unit.visible_monsters, self.max_monster_batches)  # 视野列表
            index = index + self.max_monster_batches * 2

            if self.add_center_xy:
                own_feats[index] = (unit.tower_location[0] - center_x) / max_x_distance
                own_feats[index + 1] = (unit.tower_location[1] - center_y) / max_y_distance

        state = np.concatenate((ally_feats.flatten(),
                                enemy_feats.flatten(),
                                target_feats.flatten(),
                                own_feats.flatten())).astype(np.float32)

        # 添加全局信息

        # 添加agent_id?
        if self.obs_agent_id:
            agent_id_feats[agent_id] = 1
            state = np.concatenate((state, agent_id_feats.flatten())).astype(np.float32)

        return state


        # # 空数组
        # enemy_obs = np.zeros((self.max_monster_batches, self.missile_feature_dim),dtype=np.float32)
        # ally_obs = np.zeros((self.ally_num, self.battalion_feature_dim),dtype=np.float32)
        # self_obs = np.zeros((1, self.self_feature_dim),dtype=np.float32)
        # target_obs = np.zeros((self.target_num, self.target_feature_dim),dtype=np.float32)
        #
        #
        # # battalion
        # agent_bat = self.defense_towers[agent_id]
        # sight_range = agent_bat.attack_range
        #
        # # 遍历导弹
        # for i in range(self.max_monster_batches):
        #     current_missile = self.monsters[i]
        #
        #     if current_missile is not None:
        #         this_obs = current_missile.return_latest_obs(sight_range, agent_bat.tower_location)
        #         this_obs[-2] = this_obs[-2] / self.missile_max_values['damage'] # 归一化TNT
        #         this_obs[-3] = this_obs[-3] / self.missile_max_values['speed'] # 归一化quantity
        #
        #         distance = Geo.get_horizontal_distance(current_missile.now_loc, agent_bat.tower_location)
        #         distance_norm = distance / sight_range # 归一化距离
        #         this_obs = np.append(this_obs, distance_norm)
        #
        #         if distance < sight_range:
        #             this_obs = np.insert(this_obs,0 ,1)
        #         else:
        #             this_obs = np.insert(this_obs,0 ,0)
        #     else:
        #         this_obs = np.zeros(self.missile_feature_dim)
        #
        #
        #     enemy_obs[i] = this_obs #(8,)
        #
        # # 遍历火力营
        # bat_count = 0
        # for bat in self.defense_towers:
        #     if bat.tower_id != agent_id:
        #         # 不是自己的火力营才考虑
        #
        #         distance = Geo.get_horizontal_distance(bat.tower_location, agent_bat.tower_location)
        #         if distance < sight_range:
        #             # 在视野范围内
        #             this_obs = bat.return_latest_obs(sight_range, agent_bat.tower_location)
        #             distance_norm = distance / sight_range # 归一化距离
        #             this_obs = np.append(this_obs, distance_norm)
        #             ally_obs[bat_count] = this_obs # （12，）
        #         bat_count += 1
        #
        # # 遍历target
        # for i in range(self.target_num):
        #     distance = Geo.get_horizontal_distance(self.target_points[i].location, agent_bat.tower_location)
        #     if distance < sight_range:
        #         # 在视野范围内才考虑
        #         this_obs = self.target_points[i].return_latest_obs(sight_range, agent_bat.tower_location)
        #         distance_norm = distance / sight_range
        #         this_obs = np.append(this_obs, distance_norm) # （6，）
        #
        #         # one-hot编码target id
        #         one_hot = np.zeros(self.target_num)
        #         one_hot[i] = 1
        #         this_obs = np.append(this_obs, one_hot)
        #
        #         target_obs[i] = this_obs # （6+targe_num，）
        #
        # # self obs
        # own_bat_obs = agent_bat.return_latest_obs(1., agent_bat.tower_location)
        # own_bat_obs = own_bat_obs[3:] # 去掉位置信息
        # self_obs[0] = own_bat_obs # （7，）
        #
        # return np.concatenate((enemy_obs.flatten(),
        #                        ally_obs.flatten(),
        #                        target_obs.flatten(),
        #                        self_obs.flatten(),
        #                        ))

    def close(self):
        '''
        关闭环境
        :return:
        '''
        # 关闭UI
        if self.show_ui:
            self.close_ui()

        # 清除导弹发射位
        self.monster_nests = None
        # 清楚导弹
        self.monsters = None
        # 清除火力营
        self.defense_towers = None
        # 清除目标
        self.target_points = None
        # 清除全局状态
        self.state = None
        # 清除观察
        self.observations = None
        # 清除动作
        self.action_mask = None
        return

    def close_ui(self):
        '''
        关闭UI
        :return:
        '''
        plt.close()

    def Settlement_Reward(self, done_mark):
        '''
        当游戏结束时，结算奖励,
        :param done_mark: [mark_missiles, mark_battalions, mark_rest_missiles]：
                    【所有来袭导弹都被击落，所有huoliying都没有导弹，所有来袭目标都已攻击完毕】
        '''
        r_battalion = []
        if done_mark[0]:
            print('所有来袭导弹都被击落，奖励为1')
            for battalion in self.defense_towers:
                r_battalion.append(np.max(battalion.current_ammo, 0)) # 剩余弹药数量作为奖励（枚，0~30）
                # 重要目标HP因为是0或者负值，所以不作为奖励（也算是最终奖励）。

            assert np.sum(r_battalion) >= 0, 'r_battalion should be positive, when all monsters have been intercepted'
            # 返回击落的导弹数量占比作为奖励，意味着击落的导弹越多，奖励越高; 同时返回拦截导弹占比，因为是所有来袭导弹都被击落，因此
            return np.sum(r_battalion)/self.sum_ammo_towers + 1, 1

        if (done_mark[1] and self.monsters_Is_exist()) or done_mark[2]: # 所有huoliying都没有导弹, 其余来袭导弹会直接攻击重要目标，进行计时和汇总
            if done_mark[1] and self.monsters_Is_exist():
                print('所有huoliying都没有导弹,') # 火力营没有弹药，需要看是否还有来袭目标，如果有，需要计算导弹弹量，
            if done_mark[2]:
                print('所有来袭目标都已攻击完毕, huoliying可能有剩余没发射的导弹')
            # 后续攻击加速完成
            for monster in self.monsters:
                if monster is not None: # 来袭目标没有被击落的部分
                    r_battalion.append(monster.damage * monster.hp)
            return 1 - np.sum(r_battalion)/self.sum_damage_monsters, 1 - np.sum(r_battalion)/self.sum_damage_monsters  # 返回击落的导弹数量占比作为奖励，意味着击落的导弹越多，奖励越高

        else:
            print('游戏结束，但没有合适的收益清算')
            return None


    def set_norm_scale_battalion(self):
        max_values = {
            "attack_range" : [],
            "projectile_speed" : [],
            "hit_rate" : [],
            "fixed_reaction_time" : [],
            "max_attack_per_channel" : [],
            "attack_channels" : []
        }
        # 获取最大值
        for bat in self.defense_towers:
            max_values["attack_range"].append(bat.attack_range)
            max_values["projectile_speed"].append(bat.projectile_speed)
            max_values["hit_rate"].append(bat.hit_rate)
            max_values["fixed_reaction_time"].append(bat.fixed_reaction_time)
            max_values["max_attack_per_channel"].append(bat.max_attack_per_channel)
            max_values["attack_channels"].append(bat.attack_channels)
        for key in max_values.keys():
            max_values[key] = max(max_values[key])
        for bat in self.defense_towers:
            bat.set_norm_scale(max_values)
        return

    def set_norm_scale_missiles(self):
        max_values = {
            "speed" : 400,   # defaults from monster_types
            "damage" : 100,
        }
        # Get max from monster_types config
        if hasattr(self, 'monster_types') and len(self.monster_types) > 0:
            speeds = [mt.get('speed', 400) for mt in self.monster_types]
            damages = [mt.get('damage', 100) for mt in self.monster_types]
            max_values['speed'] = max(speeds) if speeds else 400
            max_values['damage'] = max(damages) if damages else 100

        self.monster_max_values = max_values
        return

    def missiles_Is_exist(self):
        '''
        判断导弹是否存在或到达目标
        :return:
        '''
        for monster in self.monsters:
            if monster is not None and monster.now_loc is not None:
                return True
        return False
