import math

from .tower import Tower
from .monster import Monster
from .monster_spawn import MonsterSpawn
from .target import Target
import numpy as np
import random
from .multi_discrete import MultiDiscrete
from .utils import geo as Geo
from gym import spaces
import json5
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
plt.ioff()


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
    def __init__(self, args):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(f"{current_dir}/env_config.json5", "r", encoding="utf-8") as f:
            parameters = json5.load(f)

        # 默认的参数
        self.debug = True
        self.obs_agent_id = True
        self.obs_timestep_number = False
        self.stacked_frames = args.stacked_frames
        self.use_stacked_frames = args.use_stacked_frames
        self.obs_last_action = False # 是否使用上一步的动作作为观察,  todo:后续如果需要，添加
        self.add_center_xy = args.add_center_xy # 是否添加中心坐标
        self.state_last_action = False # 是否使用上一步的动作作为状态 todo:后续如果需要，添加

        self.obs_instead_of_state = args.use_obs_instead_of_state # 是否使用观察代替状态
        self.use_state_agent = args.use_state_agent # 是否使用状态代替agent
        self.add_local_obs = args.add_local_obs # 是否添加局部观察
        self.add_visible_state = args.add_visible_state # 是否添加可见状态
        self.use_stacked_frames = args.use_stacked_frames # 是否使用堆叠帧, 默认为False
        self.show_ui = args.show_ui # 是否显示UI界面
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.reward_negative_scale = 0.5
        self.use_last_action = args.add_last_action # 是否使用上一步的动作作为观察, 由于不同时刻action是错位的，可能带来扰动。
        # 如果reward_only_positive为False，设置负奖励的比例, 默认为0.5
        # 如果reward_only_positive为True，则忽略reward_negative_scale参数
        self.reward_only_positive = True  # 是否只有正奖励, 默认为True
        self.reward_scale = 10  # 奖励缩放因子, 默认为10
        self.quantity = parameters['monster_type'][0]['quantity']

        '''
        是否引入模型进行目标分配, str with ['distance', 'TNT', 'speed', 'comprehensive']
        '''
        self.assign_model_policy = args.assign_model_policy

        self._seed = 2024
        # 初始化环境边界（单位：经纬度）
        # 西南角 (SW)：30.0° N, 100.0° E 西北角 (NW)：40.0° N, 100.0° E
        # 东北角 (NE)：40.0° N, 110.0° E 东南角 (SE)：30.0° N, 110.0° E
        # 一个纬度差为10度、经度差为10度的区域。
        self.border = parameters['border']

        # 设置采样时间（单位：秒）1/2/3
        self.dt = parameters['dt']

        # sum_tnt_missiles: 用于归一化奖励,敌方怪兽的最大数量
        self.sum_monster_hp = 0
        # 上一时刻怪兽总HP
        self.prev_monster_hp = 0
        # sum_ammo_battalions: 用于归一化奖励,敌方防御塔的最大数量
        self.sum_tower_ammo = 0
        # 上一时刻防御塔总弹药
        self.prev_tower_ammo = 0

        # 标记是否赢得比赛
        self.win_counted = False
        # 标记是否输掉比赛
        self.defeat_counted = False

        '''
        重要目标设置
        '''
        self.target_num = parameters['target_num']  # 设置重要目标数量
        self.target_points = []
        self.construct_target(parameters['target_points_parems'])

        '''
        防御塔设置 (原防御塔)
        '''
        self.tower_num = parameters['tower_num']  # tower_num
        self.n_agents = self.tower_num
        self.towers = []
        self.construct_towers(parameters['tower_params'])
        self.set_norm_scale_towers()
        # 计算max_tower_channels，是所有防御塔中最大的通道数量
        max_tower_channels = 0
        for tower in self.towers:
            if tower.num_channels > max_tower_channels:
                max_tower_channels = tower.num_channels
        self.max_tower_channels = max_tower_channels

        # todo: add monster spawn module, 当防御塔的最大攻击批次为2时，需要判断是否开启简化版模式.
        # todo:   条件：如果发射模式为False，则不启用怪兽生成模块；否则，需要加入运筹模块。
        self.monsters_spawn_module = False if self.towers[0].max_attack_per_channel == 2 and args.lunch_model_bal == False else True

        '''
        怪兽巢穴 & 怪兽波次设置 (原来袭导弹)
        '''
        # 怪兽巢穴
        self.monster_spawn_num = parameters['monster_spawn_num']  # monster_spawn_num
        self.monster_spawn_interval = parameters['monster_spawn_interval']  # monster_spawn_interval
        self.monster_max_values = None
        self.monster_spawn_list = []
        self.monster_spawn_locations = parameters.get('monster_spawn_locations', [
            [-34.7, 74], [-34.7, 75], [-32.9, 76.4], [-33, 76.6],
            [-33.1, 76.8], [-34.9, 77.3], [-35.66, 76.9]
        ])

        self.monster_wave_num = parameters['monster_wave_num']  # monster_wave_num
        self.monster_type = parameters['monster_type']  # monster_type
        self.monsters = []

        ''' 基于每个防御塔构建观察observation '''
        self.observations = []

        ''' 遍历怪兽、防御塔、重要目标构建全局状态 '''
        self.state = []

        ''' 初始化action_mask '''
        self.action_mask = []
        for tower in self.towers:
            self.action_mask.append(tower.action_mask[np.newaxis, :, :])
        self.action_mask = np.vstack(self.action_mask)
        self.action_mask[:, :, 0] = 1
        self.action_space_agent = self.max_tower_channels

        ''' 获取环境中各个状态维度信息 '''
        self.see_monster_window = 4

        # ally obs
        self.ally_num = self.tower_num - 1
        self.tower_feature_dim = 9 + self.max_tower_channels + self.monster_wave_num

        self.monster_feature_dim = 4 + self.target_num + 6 + len(self.monster_type)

        # self obs
        self.self_feature_dim = 9 + self.max_tower_channels + 2 * self.monster_wave_num

        # target obs
        self.target_feature_dim = 6

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        for i in range(self.n_agents):
            self.action_space.append(MultiDiscrete([[0, self.towers[0].max_attack_per_channel - 1] for _ in range(self.max_tower_channels)]))
            self.observation_space.append(self.get_obs_size())
            self.share_observation_space.append(self.get_state_size())

        if self.use_stacked_frames:
            self.stacked_local_obs = np.zeros((self.n_agents, self.stacked_frames, int(self.get_obs_size()[0]/self.stacked_frames)), dtype=np.float32)
            self.stacked_global_state = np.zeros((self.n_agents, self.stacked_frames, int(self.get_state_size()[0]/self.stacked_frames)), dtype=np.float32)


        ## 默认参数
        self.last_action = np.zeros((self.n_agents, self.action_space_agent), dtype=np.float32)

        # 渲染相关
        self.fig = None
        self.ax = None

    def get_obs_monsters_size(self):
        # 返回导弹的观察空间大小
        nf_en = self.monster_feature_dim
        if self.add_center_xy:
            nf_en += 2
        return self.monster_wave_num, nf_en

    def get_state_monsters_size(self):
        # 返回导弹的状态空间大小,与obs类似
        nf_en = self.monster_feature_dim
        if self.add_center_xy:
            nf_en += 2
        return self.monster_wave_num, nf_en

    def get_obs_towers_size(self):
        # 返回防御塔的观察空间大小
        nf_al = self.tower_feature_dim
        if self.add_center_xy:
            nf_al += 2

        if self.obs_last_action:
            nf_al += self.monster_wave_num

        return self.tower_num - 1, nf_al # 除了自己外的防御塔数量, -1是因为自己不需要观察自己

    def get_state_towers_size(self):
        # 返回防御塔的观察空间大小
        nf_al = self.tower_feature_dim
        if self.add_center_xy:
            nf_al += 2

        if self.obs_last_action:
            nf_al += self.action_space_agent

        return self.tower_num - 1, nf_al  # 除了自己外的防御塔数量, -1是因为自己不需要观察自己


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
        n_enemies, n_enemy_feats = self.get_obs_monsters_size()
        n_allies, n_ally_feats = self.get_obs_towers_size()
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

        if self.obs_timestep_number:
            timestep_feats = 1
            all_feats += timestep_feats

        return [all_feats * self.stacked_frames if self.use_stacked_frames
                else all_feats, [n_allies, n_ally_feats], [n_enemies, n_enemy_feats], [n_target, n_target_feats],
                [1, own_feats_size + agent_id_feats + timestep_feats]]

    def get_state_size(self):
        if self.obs_instead_of_state:
            return [self.get_obs_size()[0] * self.n_agents, [self.n_agents, self.get_obs_size()[0]]]

        if self.use_state_agent: # 使用全局状态代替agent状态
            own_feats_size = self.get_state_own_feats_size()
            n_enemies, n_enemy_feats = self.get_state_monsters_size()
            n_allies, n_ally_feats = self.get_state_towers_size()
            n_target, n_target_feats = self.get_state_target_size()

            enemy_feats = n_enemies * n_enemy_feats
            ally_feats = n_allies * n_ally_feats
            target_feats = n_target * n_target_feats

            all_feats = enemy_feats + ally_feats + own_feats_size + target_feats

            agent_id_feats = 0
            timestep_feats = 0

            if self.obs_agent_id:
                agent_id_feats = self.n_agents
                all_feats += agent_id_feats

            if self.obs_timestep_number:
                timestep_feats = 1
                all_feats += timestep_feats

            return [all_feats * self.stacked_frames if self.use_stacked_frames
                    else all_feats, [n_allies, n_ally_feats], [n_enemies, n_enemy_feats], [n_target, n_target_feats], [1, own_feats_size + agent_id_feats+timestep_feats]]

        ## 使用agent状态, 与obs类似
        nf_al, nf_en = self.tower_feature_dim, self.monster_feature_dim

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

        enemy_state = self.monster_wave_num * nf_en
        ally_state = self.tower_num * nf_al
        target_state = self.target_num * self.target_feature_dim

        size = enemy_state + ally_state + target_state

        obs_agent_size = 0
        timestep_state = 0

        if self.add_local_obs:
            obs_agent_size = self.get_obs_size()[0]
            size += obs_agent_size

        return [size * self.stacked_frames if self.use_stacked_frames else size, [self.n_agents, nf_al],
                [self.monster_wave_num, nf_en], [self.target_num, self.target_feature_dim], [1, obs_agent_size + timestep_state]]

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
        # Presentation map: clean tactical view with custom, low-noise symbols.
        xs = [self.border[0][1], self.border[1][1], self.border[2][1], self.border[3][1], self.border[0][1]]
        ys = [self.border[0][0], self.border[1][0], self.border[2][0], self.border[3][0], self.border[0][0]]
        self.ax.set_facecolor('#F8FAFC')
        self.ax.fill(xs, ys, color='#F1F5F9', alpha=1.0, zorder=0)
        self.ax.plot(xs, ys, color='#0F172A', linewidth=1.3, alpha=0.72, zorder=4)
        self.ax.set_xlim(min(xs) - 0.45, max(xs) + 0.45)
        self.ax.set_ylim(min(ys) - 0.45, max(ys) + 0.45)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, color='#CBD5E1', linestyle='-', linewidth=0.35, alpha=0.34)
        self.ax.tick_params(colors='#64748B', labelsize=8)
        self.ax.set_xlabel('Longitude', color='#475569', fontsize=9)
        self.ax.set_ylabel('Latitude', color='#475569', fontsize=9)
        for spine in self.ax.spines.values():
            spine.set_color('#334155')
            spine.set_linewidth(0.75)

        tower_range_colors = {180: '#2563EB', 200: '#0891B2', 350: '#059669'}
        tower_range_edge  = {180: '#1E40AF', 200: '#0E7490', 350: '#047857'}
        tower_range_label = {180: 'Short-range radar', 200: 'Mid-range radar', 350: 'Long-range radar'}

        # Coverage first: quiet rings in the background.
        first_of_type = {}
        for tower in self.towers:
            r = tower.range
            c = tower_range_colors.get(r, '#64748B')
            ec = tower_range_edge.get(r, '#334155')
            lat, lon = tower.location[0], tower.location[1]
            _, lon_radius = km_to_lat_lon_radius(lat, tower.range)
            self.ax.add_patch(plt.Circle((lon, lat), lon_radius, color=c, fill=True,
                                         alpha=0.018, linewidth=0, zorder=1))
            self.ax.add_patch(plt.Circle((lon, lat), lon_radius, color=ec, fill=False,
                                         linewidth=0.62, linestyle='-', alpha=0.28, zorder=2))

        # Hostile nests: sharper military-style launch-complex symbols.
        spawn_lons = [ms.location[1] for ms in self.monster_spawn_list]
        spawn_lats = [ms.location[0] for ms in self.monster_spawn_list]
        self.ax.scatter(spawn_lons, spawn_lats,
                     marker='D', s=78, facecolors='#FEF2F2', edgecolors='#7F1D1D',
                     linewidths=1.05, alpha=0.96, zorder=8, label='Hostile nest')
        self.ax.scatter(spawn_lons, spawn_lats,
                     marker='x', s=42, c='#991B1B', linewidths=0.95,
                     alpha=0.92, zorder=10)
        self.ax.scatter(spawn_lons, spawn_lats,
                     marker='o', s=12, c='#111827', edgecolors='#FEE2E2',
                     linewidths=0.25, alpha=0.92, zorder=11)
        for lon, lat in zip(spawn_lons, spawn_lats):
            self.ax.plot([lon - 0.085, lon - 0.035], [lat, lat], color='#7F1D1D', linewidth=0.7, alpha=0.62, zorder=9)
            self.ax.plot([lon + 0.035, lon + 0.085], [lat, lat], color='#7F1D1D', linewidth=0.7, alpha=0.62, zorder=9)
            self.ax.plot([lon, lon], [lat - 0.085, lat - 0.035], color='#7F1D1D', linewidth=0.7, alpha=0.62, zorder=9)
            self.ax.plot([lon, lon], [lat + 0.035, lat + 0.085], color='#7F1D1D', linewidth=0.7, alpha=0.62, zorder=9)
            self.ax.scatter([lon], [lat], marker='D', s=145, facecolors='none', edgecolors='#DC2626',
                         linewidths=0.38, alpha=0.22, zorder=6)
        if spawn_lons and spawn_lats:
            cx, cy = np.mean(spawn_lons), np.mean(spawn_lats)
            self.ax.scatter([cx], [cy], marker='o', s=1250, facecolors='none',
                         edgecolors='#DC2626', linewidths=0.85, alpha=0.15, zorder=3)
            self.ax.scatter([cx], [cy], marker='o', s=520, facecolors='none',
                         edgecolors='#7F1D1D', linewidths=0.62, alpha=0.13, zorder=3)

        # Incoming monsters/missiles: thin tracks with compact red heads.
        for monster in self.monsters:
            if monster is not None and monster.now_loc is not None:
                self.ax.plot([monster.start_loc[1], monster.now_loc[1]],
                           [monster.start_loc[0], monster.now_loc[0]],
                           color='#B91C1C', linewidth=0.55, alpha=0.30, zorder=6)
                self.ax.scatter(monster.now_loc[1], monster.now_loc[0],
                             marker='>', s=18, c='#DC2626', edgecolors='white',
                             linewidths=0.32, alpha=0.9, zorder=12)

        # Protected assets: high-value objective symbols with a restrained crosshair.
        target_lons = [target.location[1] for target in self.target_points]
        target_lats = [target.location[0] for target in self.target_points]
        self.ax.scatter(target_lons, target_lats,
                     marker='o', s=230, facecolors='#EFF6FF', edgecolors='#1D4ED8',
                     linewidths=1.05, zorder=13, label='Protected high-value asset')
        self.ax.scatter(target_lons, target_lats,
                     marker='o', s=72, facecolors='#FEF3C7', edgecolors='#D97706',
                     linewidths=0.9, zorder=14)
        for target in self.target_points:
            lon, lat = target.location[1], target.location[0]
            self.ax.plot([lon - 0.13, lon - 0.055], [lat, lat], color='#1D4ED8', linewidth=0.85, alpha=0.82, zorder=15)
            self.ax.plot([lon + 0.055, lon + 0.13], [lat, lat], color='#1D4ED8', linewidth=0.85, alpha=0.82, zorder=15)
            self.ax.plot([lon, lon], [lat - 0.13, lat - 0.055], color='#1D4ED8', linewidth=0.85, alpha=0.82, zorder=15)
            self.ax.plot([lon, lon], [lat + 0.055, lat + 0.13], color='#1D4ED8', linewidth=0.85, alpha=0.82, zorder=15)
            self.ax.annotate('HVT-%d' % target.targetId,
                          (lon, lat),
                          fontsize=6.6, color='#1E3A8A', fontweight='bold',
                          textcoords='offset points', xytext=(5, 4), zorder=16,
                          bbox=dict(boxstyle='round,pad=0.12', fc='#EFF6FF', ec='#93C5FD', lw=0.35, alpha=0.84))

        # Defense towers: radar station glyphs with tiny scan arcs.
        for tower in self.towers:
            r = tower.range
            c = tower_range_colors.get(r, '#64748B')
            ec = tower_range_edge.get(r, '#334155')
            lbl = tower_range_label.get(r, 'Radar tower') if r not in first_of_type else None
            first_of_type[r] = True
            lat, lon = tower.location[0], tower.location[1]
            self.ax.scatter(lon, lat,
                         marker='^', s=150, c=c, edgecolors='white',
                         linewidths=1.05, zorder=16, label=lbl)
            self.ax.plot([lon, lon], [lat + 0.03, lat + 0.16], color=ec, linewidth=1.0, alpha=0.95, zorder=17)
            self.ax.add_patch(Arc((lon, lat + 0.16), 0.30, 0.16, angle=0, theta1=22, theta2=158,
                                  color=ec, linewidth=0.9, alpha=0.82, zorder=17))
            self.ax.add_patch(Arc((lon, lat + 0.16), 0.48, 0.25, angle=0, theta1=26, theta2=154,
                                  color=ec, linewidth=0.65, alpha=0.45, zorder=17))
            self.ax.annotate('%d' % tower.tower_id,
                          (lon, lat - 0.02), fontsize=6.5, color='white', fontweight='bold',
                          ha='center', va='center', zorder=18)

        # Engagement links after tower icons so active intercepts read clearly.
        for monster in self.monsters:
            if monster is None or monster.now_loc is None:
                continue
            for tower in self.towers:
                try:
                    dist = Geo.get_horizontal_distance(monster.now_loc, tower.location)
                except:
                    continue
                if dist < tower.range:
                    c = tower_range_colors.get(tower.range, '#64748B')
                    self.ax.plot([monster.now_loc[1], tower.location[1]],
                               [monster.now_loc[0], tower.location[0]],
                               color=c, linewidth=0.34, alpha=0.11, linestyle=':', zorder=7)
                if monster.monster_id in tower.attack_targets:
                    self.ax.plot([monster.now_loc[1], tower.location[1]],
                               [monster.now_loc[0], tower.location[0]],
                               color='#65A30D', linewidth=1.15, alpha=0.88, zorder=19)

        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.055), fontsize=8,
                      framealpha=0.94, facecolor='white', edgecolor='#CBD5E1',
                      markerscale=0.85, ncol=5, handletextpad=0.45, columnspacing=0.9)

        plt.pause(0.1)
    def high_speed(self, ignore=True):
        '''
        加速模式，导弹从发射阵地出发到接近其中agent的射程 这段时间内的导弹轨迹
        params: ignore: 是否忽略导弹从发射到进入射程之间的迭代轨迹， 在训练过程中默认为True，加速训练；可视化时可设置为False,以显示中间过程
        '''
        buffer_dist = 0.1 # 0.1km 缓冲， 保证导弹没有进入射程，需要保证最快导弹快要进入视野内
        time_m2range = np.inf # 更新时间间隔
        while True:
            mark = False # 标记是否有满足导弹进入缓冲距离（buffer_dist）

            # 更新导弹位置和状态
            for monster_spawn in self.monster_spawn_list:
                if monster_spawn.ready_to_spawn():
                    monsters = monster_spawn.spawn_monsters(len(self.monsters), self.target_points)
                    if monster is None or len(monsters) == 0:
                        continue
                    # for mm in monsters:
                    #     if mm is not None and mm.now_loc is None:
                    #         print(f'monster-{mm.monster_id} is None')

                    self.monsters.extend(monsters)

            if ignore and len(self.monsters) == self.monster_wave_num: # 忽略导弹从发射到进入射程之间的迭代轨迹, 已发射完毕
                for monster in self.monsters:
                    for tower in self.towers:
                        try:
                            dist = Geo.get_horizontal_distance(monster.now_loc, tower.location)
                        except Exception as e:
                            print(e)
                            continue
                        t = math.ceil((dist - (tower.range + buffer_dist)) * 1000 / monster.speed)
                        # 单位转换为s,且向上取整，导弹会到达射程末端的缓冲区
                        if t < time_m2range:
                            time_m2range = t

                for monster in self.monsters: # 更新导弹位置
                    monster.update_position(time_m2range)

                mark = True
            else:
                for monster in self.monsters:
                    for tower in self.towers:
                        try:
                            dist = Geo.get_horizontal_distance(monster.now_loc, tower.location)
                        except Exception as e:
                            print(e)
                            continue
                        if dist < tower.range + buffer_dist: # todo: 0.1km 缓冲， 保证导弹没有进入射程，需要保证最快导弹要进入视野内时，所有导弹都已发射
                            mark = True

            if mark == False:
                for monster_spawn in self.monster_spawn_list:
                    # 更新导弹位置和状态， 发射阵地更新时间，在发射完毕之前，不断已dt更新发射阵地的时间，直到发射完毕。
                    monster_spawn.update_spawn_time(self.dt)

                for monster in self.monsters:
                    monster.update_position(self.dt)
                    # if monster.now_loc is None:
                    #     print(f'monster-{monster.monster_id} is None')
            else:
                # 计算发射所有导弹当量，用于计算max_reward
                for monster in self.monsters:
                    if monster is not None:
                        self.sum_monster_hp += monster.HP * monster.quantity
                self.prev_monster_hp = self.sum_monster_hp
                for tower in self.towers:
                    self.sum_tower_ammo += tower.ammo_capacity
                self.prev_tower_ammo = self.sum_tower_ammo
                break

    def get_monster(self, id):
        # 根据missile ID值获取对应对象
        return self.monsters[id]

    def get_target(self, id):
        # 根据target ID值获取对应对象
        return self.target_points[id]

    def get_tower(self, id):
        # 根据battalion ID值获取对应对象
        return self.towers[id]

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

    def construct_target(self, target_points_parems):
        '''
        构建重要目标
        :return:
        '''
        try:
            assert len(target_points_parems) == self.target_num
            for i in range(self.target_num):
                target = Target(target_points_parems[i], i)
                self.target_points.append(target)
        except:
            print('重要目标数量不匹配')

    def construct_towers(self, tower_params):
        '''
        构建防御塔
        :param tower_params:
        :return:
        '''
        try:
            total_length = sum(map(len, [params.get('location', []) for params in tower_params]))
            assert self.tower_num == total_length

            i = 0  # 防御塔id
            for tower_param in tower_params:
                for loc in tower_param['location']:
                    b = Tower(tower_param, i, loc)
                    self.towers.append(b)
                    i += 1
        except:
            print('防御塔数量不匹配')

    def construct_monster_spawns(self):
        '''
        构建怪兽巢穴
        :return:
        '''
        self.monster_spawn_list = []
        try:
            for i in range(self.monster_spawn_num):
                spawn_params = dict()
                # 生成高斯分布的随机位置
                location = self.random_monster_spawn_location(30,40,100,110)
                spawn_params['location'] = location
                spawn_params['wave_count'] = int(self.monster_wave_num/self.monster_spawn_num)
                spawn_params['spawn_interval'] = self.monster_spawn_interval
                spawn_params['monster_type'] = self.monster_type
                spawn_params['dt'] = self.dt
                spawn_params['monster_spawn_id'] = i
                spawn_params['target_num'] = self.target_num
                m = MonsterSpawn(spawn_params)
                self.monster_spawn_list.append(m)
        except:
            print('怪物巢穴数量不匹配')

    def construct_monsters(self):
        '''
        构建怪兽
        :return:
        '''
        self.monsters = []
        for monster_spawn in self.monster_spawn_list:
            if monster_spawn.ready_to_spawn():
                spawned = monster_spawn.spawn_monsters(len(self.monsters), self.target_points)
                if spawned is not None:
                    self.monsters.extend(spawned)


    def random_monster_spawn_location(self,min_x, max_x, min_y, max_y):
        '''
        随机生成导弹的位置，基于预设的发射位位置用高斯分布扰动
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
        重置环境， 包括重置所有的目标、导弹、防御塔，重置全局状态，为每个agent分配最近的导弹，形成各自的
        Id命名规则：目标id: 't-Id', 导弹id: 'm-Id', 防御塔id: 'b-Id', 火力架id: 'a-Id'
        :return:
        '''
        # 重置全局变量
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.sum_monster_hp = 0
        self.sum_tower_ammo = 0
        self.prev_monster_hp = 0 # 上一时刻导弹总tnt, 用于归一化奖励
        self.prev_tower_ammo = 0 # 上一时刻防御塔总tnt, 用于归一化奖励
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents, self.action_space_agent), dtype=np.float32)

        # 重置重要目标参数
        for target in self.target_points:
            target.reset_Target()

        # 重置导弹发射位
        self.construct_monster_spawns()

        # 重置导弹, 发射位置随机
        self.construct_monsters()
        self.set_norm_scale_monsters()

        # 重置防御塔
        for tower in self.towers:
            tower.reset_tower()

        # 重置action_mask
        self.action_mask = []
        for tower in self.towers:
            self.action_mask.append(tower.action_mask[np.newaxis, :, :])
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
            global_state = [self.get_state_agent(agent_id) for agent_id in range(self.n_agents)]
            # global_state = self.update_global_state()
        else:
            global_state = None
            # global_state = [self.get_state(agent_id) for agent_id in range(self.n_agents)]

        # local_obs = self.get_obs()
        # local_obs = self.assign_observations()

        return global_state, global_state, available_actions




    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs


    def get_state_agent(self, agent_id):
        '''
        Returns observation for agent_id. The observation is composed of:

        - enemy features (monsters features from return_latest_state)
        - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
        - target features (target features from return_latest_state)

        All of this information is flattened and concatenated into a list,
        in the aforementioned order. To know the sizes of each of the
        features inside the final list of features, take a look at the
        functions ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` ,
        ``get_obs_target_feats_size()``, and
        ``get_obs_own_feats_size()``.

        The size of the observation vector may vary, depending on the
        environment configuration and type of units present in the map.
        For instance, non-Protoss units will not have shields, movement
        features may or may not include terrain height and pathing grid,
        unit_type is not included if there is only one type of unit in the
        map etc.).

        NOTE: Agents should have access only to their local observations
        during decentralised execution.

        获取agent的状态
        :param agent_id:
        :return:
        '''

        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
            return obs_concat

        unit = self.towers[agent_id]

        # 获取各类目标的特征维度
        enemy_feats_dim = self.get_state_monsters_size()
        ally_feats_dim = self.get_state_towers_size()
        target_feats_dim = self.get_state_target_size()
        own_feats_dim = self.get_state_own_feats_size()

        enemy_feats  = np.zeros(enemy_feats_dim, dtype=np.float32)
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

        # 防御塔还有导弹
        if unit.remaining_ammo > 0:
            # 对导弹进行观察
            for monster in self.monsters:
                if monster is not None and monster.now_loc is not None: # 导弹还在,但可能已经打击目标， 需要有个标记位
                    state_m = np.array([0.0] * enemy_feats_dim[1], dtype=np.float32)

                    dist = Geo.get_horizontal_distance(monster.now_loc, unit.location) # 返回的是km
                    dist_x = Geo.get_horizontal_distance(monster.now_loc, [monster.now_loc[0], unit.location[1]]) # x轴相对距离
                    dist_y = Geo.get_horizontal_distance(monster.now_loc, [unit.location[0], monster.now_loc[1]]) # y轴相对距离
                    state_m[0] = dist/unit.range
                    state_m[1] = dist_x/unit.range
                    state_m[2] = dist_y/unit.range

                    if dist < unit.range:
                        state_m[3] = 1 # visible

                    index = 4 # # 用于记录导弹特征的索引
                    # 攻击目标
                    state_m[index: (index+self.target_num)] = self.one_hot_encode(monster.targetId, self.target_num)
                    index += self.target_num
                    # 导弹类型
                    state_m[index: (index+len(self.monster_type))] = self.one_hot_encode(monster.mtype, len(self.monster_type))
                    index += len(self.monster_type)
                    # 速度
                    state_m[index] = Geo.speed_m2Mach(monster.speed)
                    # 剩余导弹，归一化
                    state_m[index+1] = monster.quantity / monster.init_quantity
                    # TNT当量
                    state_m[index+2] = monster.HP/100 # TNT 缩小100倍，防止数值过大 todo: check TNT
                    # 导弹角度,需转换为0~2 pi，
                    state_m[index+3] = monster.angle/360 * 2 * np.pi # 导弹角度, 0-360, 转换为弧度
                    # 导弹上次打击时间
                    state_m[index+4] = monster.last_attacked_time/60 # 转换为分钟
                    # 被打击次数
                    state_m[index+5] = monster.attack_times # 被打击次数
                    ## 相对于中心的位置, 放到最后位置
                    if self.add_center_xy:
                        state_m[index+6] = (monster.now_loc[0] - center_x) / max_x_distance  # x轴相对位置,相对于中心
                        state_m[index+7] = (monster.now_loc[1] - center_y) / max_y_distance  # y轴相对位置,相对于中心

                    # 赋值给missile_state
                    enemy_feats[monster.monster_id] = state_m

            # 对防御塔进行观察
            # 去掉自己外的防御塔
            other_units = [other_unit for other_unit in self.towers if other_unit.tower_id != unit.tower_id]
            for i, other_unit in enumerate(other_units):
                state_b = np.array([0.0] * ally_feats_dim[1], dtype=np.float32)

                dist = Geo.get_horizontal_distance(other_unit.location, unit.location) # 防御塔与本防御塔的距离
                dist_x = Geo.get_horizontal_distance(other_unit.location, [other_unit.location[0], unit.location[1]]) # x轴相对距离
                dist_y = Geo.get_horizontal_distance(other_unit.location, [unit.location[0], other_unit.location[1]]) # y轴相对距离
                state_b[0] = dist/unit.range
                state_b[1] = dist_x/unit.range
                state_b[2] = dist_y/unit.range

                if dist < unit.range: # 是否在射程内，意为是否可协同打击
                    state_b[3] = 1 # visible

                # 上个时刻的action，大小为火力通道数
                state_b[4:(4 + self.max_tower_channels)] = self.last_action[other_unit.tower_id]
                index = 4 + self.max_tower_channels
                # 防御塔剩余导弹
                state_b[index] = other_unit.remaining_ammo / other_unit.ammo_capacity # 这里需要注意，有的防御塔可能会出现负值，需要debug in fire_battalion
                # 导弹拦截速度,转换为马赫数
                state_b[index + 1] = Geo.speed_m2Mach(other_unit.projectile_speed)
                # 导弹拦截范围
                state_b[index + 2] = other_unit.range/other_unit.norm_scale['range']
                # 导弹拦截概率
                state_b[index + 3] = other_unit.hit_probability
                # 固定反应时间（min）
                state_b[index + 4] = other_unit.reaction_time/60
                index += 5
                # 拦截列表， 也就是available_actions
                state_b[index: (index+self.monster_wave_num)] = self.one_hot_encode(other_unit.attack_targets, self.monster_wave_num) # 拦截列表， 也就是available_actions
                # 可见导弹
                # state_b[9+self.monster_wave_num: 9 + self.monster_wave_num *2] = self.one_hot_encode(other_unit.insght_missile, self.monster_wave_num) # 视野列表

                index += self.monster_wave_num
                if self.add_center_xy:
                    state_b[index] = (other_unit.location[0] - center_x) / max_x_distance # x轴相对位置,相对于中心
                    state_b[index+1] = (other_unit.location[1] - center_y) / max_y_distance # y轴相对位置,相对于中心

                ally_feats[i] = state_b

            # 对重点目标进行观察
            for i, target in enumerate(self.target_points):
                state_t = np.array([0.0] * target_feats_dim[1], dtype=np.float32)

                dist = Geo.get_horizontal_distance(target.location, unit.location) # 重要目标与防御塔的距离
                dist_x = Geo.get_horizontal_distance(target.location, [target.location[0], unit.location[1]]) # x轴相对距离
                dist_y = Geo.get_horizontal_distance(target.location, [unit.location[0], target.location[1]]) # y轴相对距离
                state_t[0] = dist/unit.range
                state_t[1] = dist_x/unit.range
                state_t[2] = dist_y/unit.range

                if dist < unit.range: # 是否在射程内，意为是否可直接拦截
                    state_t[3] = 1 # visible

                # 重要目标HP，
                state_t[4] = target.HP if target.init_HP == 0 else target.HP/target.init_HP
                # 重要性
                state_t[5] = target.importance

                index = 6
                if self.add_center_xy:
                    state_t[index] = (target.location[0] - center_x) / max_x_distance
                    state_t[index+1] = (target.location[1] - center_y) / max_y_distance

                target_feats[i] = state_t

            # 对自身进行观察
            own_feats[:3] = 0 # 对应防御塔特征，前三个特征为距离，相对x，相对y，直接置0
            own_feats[3] = 1 # visible
            own_feats[4: 4 + self.max_tower_channels] = self.last_action[unit.tower_id]  # 上个时刻的action
            index = 4 + self.max_tower_channels
            own_feats[index] = unit.remaining_ammo / unit.ammo_capacity # 剩余导弹
            own_feats[index + 1] = Geo.speed_m2Mach(unit.projectile_speed) # 导弹拦截速度
            own_feats[index + 2] = unit.range/unit.norm_scale['range'] # 导弹拦截范围
            own_feats[index + 3] = unit.hit_probability # 导弹拦截概率
            own_feats[index + 4] = unit.reaction_time/60 # 固定反应时间（min）
            index += 5
            own_feats[index: (index+self.monster_wave_num)] = self.one_hot_encode(unit.attack_targets, self.monster_wave_num) # 拦截列表， 也就是available_actions
            own_feats[index + self.monster_wave_num: index + self.monster_wave_num * 2] = self.one_hot_encode(unit.visible_monsters, self.monster_wave_num) # 视野列表
            index = index + self.monster_wave_num * 2

            if self.add_center_xy:
                own_feats[index] = (unit.location[0] - center_x) / max_x_distance
                own_feats[index + 1] = (unit.location[1] - center_y) / max_y_distance

        state = np.concatenate((ally_feats.flatten(),
                                enemy_feats.flatten(),
                                target_feats.flatten(),
                                own_feats.flatten())).astype(np.float32)

        # 添加全局信息

        # 添加agent_id?
        if self.obs_agent_id:
            agent_id_feats[agent_id] = 1
            state = np.concatenate((state, agent_id_feats.flatten())).astype(np.float32)

        # if self.debug:
        #     print(f"obs state shape: {state.shape}")
            # print(f"enemy_feats shape: {enemy_feats.shape}")
            # print(f"ally_feats shape: {ally_feats.shape}")
            # print(f"target_feats shape: {target_feats.shape}")
            # print(f"own_feats shape: {own_feats.shape}")
            # print(f"agent_id_feats shape: {agent_id_feats.shape}")

        return state

    def mask_actions(self, actions):
        '''
        actions 是ndarray, shape [1, agent_num, action_space_agent], 元素为离散值（如0-4）
        self.action_mask 是ndarray, shape [agent_num, action_space_agent, max_fire_onebatch], 元素为0或1, 表示是否可执行
        本函数将actions中不可执行的动作置为0, 使其不可执行
        '''
        try:
            actions = actions.reshape(self.n_agents, self.action_space_agent)
        except:
            print(f'actions shape error: {actions.shape}')

        assert actions.shape == (self.n_agents, self.action_space_agent), f'actions shape error: {actions.shape}'

        action_mask_index = np.sum(self.action_mask, axis=-1)-1 # shape: [n_agents, action_space_agent]
        # action_mask_index = np.where(self.action_mask == 1)[2].reshape(self.n_agents, self.max_tower_channels) # 返回可执行的动作索引
        # 将action_mask_index转换为整数形式
        action_mask_index = np.minimum(actions, action_mask_index).astype(int) # 将actions中不可执行的动作置为0

        return action_mask_index

    def step(self, action):
        '''
        先执行动作，再基于dt更新所有状态，返回新的状态和观察
        :param action: shape [防御塔数量，火力通道数量, 单次最大发射量] or [防御塔数量，火力通道数量, 0 or 1 (是否发射)]
        :return:
        '''
        # mask actions that are not available in the current state
        action = self.mask_actions(action) # 将不可执行的动作置为0
        info = [{} for _ in range(self.n_agents)] # 用于存储额外信息
        dones = np.zeros(self.n_agents, dtype=bool) # 用于判断是否结束
        self.last_action = action # 保存上一步的动作

        # todo: 启用 self.monsters_spawn_module 模式，主要判断是否有协同打击目标，对目标进行分配并直接执行！
        if self.monsters_spawn_module:
            self.reassign_interception_target_and_perform_action(action)
        else:
            # 根据action执行动作
            for a in range(len(action)):  # 遍历每个防御塔
                self.towers[a].perform_action(action[a], self.monsters)  # 执行拦截动作，在防御塔层面更新信息

        # 经过dt时间后，全局信息更新
        # 更新导弹发射位状态, 发射后更新时间
        if self.monster_wave_num < len(self.monsters): # 如果导弹还没有发射完毕，则继续发射，不然直接跳过发射过程。
            for monster_spawn in self.monster_spawn_list:
                if monster_spawn.wave_count <= 0: # 发射完毕
                    continue

                if monster_spawn.ready_to_spawn():
                    batch_missiles = monster_spawn.spawn_monsters(len(self.monsters), self.target_points)
                    if batch_missiles is not None:
                        self.monsters.extend(batch_missiles)
                monster_spawn.update_spawn_time(self.dt)

        # 更新导弹位置和状态
        for monster in self.monsters:
            if monster is None or monster.now_loc is None: # 被击落 或已完成攻击任务
                continue
            monster.update_position(self.dt)
        self.update_monster_state() # 更新导弹状态, 例如是否击中目标, 击中目标则直接更新目标状态, 更新冷却时间

        # 经过dt时间后,更新防御塔状态, 注意，执行打击之后，原来的导弹可能已经被击落，所以需要重新更新视野和拦截目标
        self.update_tower_state()

        self._total_steps += 1
        self._episode_steps += 1

        # 两种结束方式: 1.所有怪兽都被消灭或者都到达目标 2.所有防御塔都没有弹药
        # 判断是否结束， 来袭目标用完, dones: 用于表示每个agent是否结束
        # 为每个agent评估奖励及损失
        reward, inter_ratio = self.evaluate_reward()  # todo 成功攻击怪兽，浪费导弹负值，重要目标被攻击
        done, done_mark, dones = self.check_done(dones)

        # print(f'reward:{reward}')
        available_actions = self.form_available_actions() # 需要与action_mask对应,获得available_actions

        # todo: 结算奖励, 重要目标是否被打击完 or 弹药用完 or 来袭目标用完
        if done and not self.defeat_counted: # 游戏结束, self.win_counted, self.defeat_counted用于标记是否已经计算过胜利或者失败,只要有标记就表示结束，两者只有一个是True
            # 清算奖励，基于目标被打击的情况
            rewards_ending, inter_ratio = self.Settlement_Reward(done_mark)
            reward += rewards_ending
            print(f'{self._episode_steps} step: reward:{reward}')
            print(f'{self._episode_steps} step: interception ratio: {inter_ratio}')
            self.defeat_counted = True
        elif done and self.defeat_counted: # 游戏结束，并已设置标志位，不再计算奖励
            reward = 0

        for ai in range(self.n_agents):
            info[ai] = {
                "interception_target": len([x for x in self.towers[ai].attack_targets if x is not None]),
                "intercepted_missile": self.towers[ai].attack_targets,
                "insight_missile": len(self.towers[ai].visible_monsters),
                "rest_ammo": self.towers[ai].remaining_ammo,
                "done": dones[ai], # 记录每个agent是否还有剩余弹药，没有则为True
                "reward": reward * self.reward_scale,
                "individual_reward": self.towers[ai].get_reward()/self.sum_monster_hp * self.reward_scale,
                "inter_ratio": inter_ratio, # 记录总的拦截信息
            }

        if self.use_state_agent:
            global_state = [self.get_state_agent(agent_id) for agent_id in range(self.n_agents)]
            # global_state = self.update_global_state()
        else:
            global_state = None

        # if self.reward_scale:
        #     reward /= np.float32(self.max_tnt)/self.reward_scale_rate

        rewards = [[reward * self.reward_scale]]*self.n_agents # 每个agent的reward
        dones = np.array([done] * self.n_agents)

        return global_state, global_state, rewards, dones, info, available_actions #r 单步reward


    def reassign_interception_target_and_perform_action(self, action):
        """
        Reassign interception target based on the action of each battalion **when self.monsters_spawn_module = True**,
        主要考察是否有协同打击目标，如果有协同打击的目标，则根据这批目标是否被拦截过进行弹药分配：
            如果未被拦截过则发射最多三枚，如果被拦截过一次，则最多发射两枚，如果被拦截过多次，则只发射一枚，根据拦截概率边际增益最大为原则；
        若没有协同打击的目标，则根据action分配弹药。同样的规则同上。
        以上规则是为了保证弹药的最大利用率，提高拦截效率。并且对action进行了mask，使得不可执行的动作不会被执行。
        将action（[防御塔，火力通道，是否发射]）转换为[防御塔，火力通道，发射几枚]，后续直接执行，其中“发射几枚”是基于“二拦一”设定的，即来袭导弹数量 x 2。
        """
        if np.sum(action) == 0:  # 如果没有拦截动作，则直接返回
            return action

        action_num = np.zeros((self.n_agents, self.action_space_agent), dtype=np.float32)
        interception_target_record = np.zeros([self.n_agents, self.action_space_agent], dtype=np.float32) - 1
        # 记录每个防御塔的拦截目标, 用于判断是否协同打击，记录ID，-1表示没有拦截目标

        # 首先，先判断是否有协同打击的目标
        for agent_id in range(self.n_agents):
            ## todo: 记录拦截目标，且action>0（即可执行打击任务），才被记录。
            misId_loc = [i for i in range(len(self.towers[agent_id].attack_targets)) if self.towers[agent_id].attack_targets[i] is not None and action[agent_id, i] > 0]
            if len(misId_loc) == 0: # 打击列表中无目标
                continue
            else:
                for interceptId in misId_loc: # 保存拦截目标Id
                    interception_target_record[agent_id, interceptId] = self.towers[agent_id].attack_targets[interceptId]

        unique_elements, counts = np.unique(interception_target_record, return_counts=True)
        unique_elements = unique_elements[1:] # 去掉-1
        counts = counts[1:] # 去掉-1

        # 或者离重要目标距离作为排序依据
        unique_elements = sorted(unique_elements, key=lambda x: Geo.get_horizontal_distance(self.monsters[int(x)].now_loc, self.target_points[0].location))
        # todo 综合排序. 或者离重要目标距离排序

        for unique_element in unique_elements:  ## todo: 在是否打击的过程中，直接分配完弹药后，直接打击，返回打击效果。
            agent_id, col_id = np.where(interception_target_record == unique_element) # 对于某个目标，找到拦截目标的防御塔
            monster = self.monsters[int(unique_element)]
            prob_list = dict()
            for a_id in agent_id:
                if self.towers[a_id].remaining_ammo <= 0: # 如果没有弹药，则不参与打击
                    continue
                prob_list[self.towers[a_id].hit_probability] = min(self.towers[a_id].remaining_ammo, 4) # todo: 每一轮rest_ammo是否需要更新？基于rest_ammo?

            if len(prob_list) == 0: # 如果没有弹药，则不参与打击
                continue
            ## todo：目前是最接近来袭目标数计算的，后续可以考虑其他计算方式，毕竟要协同，可以考虑概率大的agent至少发射一枚。
            probDict = self.interaction_strategy_by_pro(monster, prob_list) # 得到最接近的拦截概率和发射组合
            dynamic_time_list = [] # 记录agent的最长拦截时间

            for loc in range(len(agent_id)): # todo: 协同打击，但是需要判断这种协同是否符合probDict的打击要求。
                a_id, h_id = agent_id[loc], col_id[loc]
                if self.towers[a_id].hit_probability not in probDict.keys(): # 如果拦截概率为0，则不执行
                    continue

                action_num[a_id, h_id] = probDict[self.towers[a_id].hit_probability] # 更新action_num
                num_missiles = probDict[self.towers[a_id].hit_probability]  # 发射导弹数量

                tower = self.towers[a_id]
                # 检查弹药是否足够，不足则按剩余量发射
                if tower.remaining_ammo < num_missiles:
                    num_missiles = max(0, int(tower.remaining_ammo))
                if num_missiles <= 0:
                    continue
                # 更新tower的剩余弹药量
                temp_action = np.zeros_like(action[0])
                temp_action[h_id] = num_missiles
                tower.update_remaining_ammo(temp_action)

                monster.attack_times += num_missiles  # 记录怪兽被攻击次数
                monster.attacked_prob_list.extend([tower.hit_probability] * int(num_missiles))
                # 开始执行打击任务。
                for _ in range(int(num_missiles)):  # 产生随机数，判断是否命中
                    if np.random.rand() < tower.hit_probability and monster.quantity > 0:
                        tower.reward_adding(monster.loss)  # 奖励
                        tower.killed_monsters.append([monster.monster_id])  # 记录攻击怪兽
                        monster.remove_one_monster()  # quantity值减1

                # 从视野删除命中目标信息，释放火力通道
                if monster.quantity == 0:
                    tower.visible_monsters.remove(monster.monster_id)
                    tower.remove_attack_target(monster.monster_id)
                else:
                    # 记录观察需要的时间
                    dynamic_time = (Geo.get_horizontal_distance(tower.location, monster.now_loc)
                                    / Geo.speed_ms2kmh(tower.projectile_speed) * 3600)
                    dynamic_time_list.append(dynamic_time + tower.reaction_time)

            if monster.quantity > 0: # 如果没有被打掉，需要设置拦截时间，用于判断之后的拦截条件。
                monster.last_attacked_time = max(dynamic_time_list) # 更新最长拦截时间



    def interaction_strategy_by_pro(self, monster, probDict):
        """
        基于目前场景，来袭目标批次中含弹 1 or 2, 对于为被拦截的默认为2，计算拦截概率pro, 2<pro<3.
                                            对于被拦截过的，默认为1， 计算拦截概率pro，要根据之前加和的概率计算，否则总是发射2枚
        如果发射导弹数冲突了，按照距离目标的距离进行排序，距离近的先发射！！

        :param monster: 要拦截的目标
        :param probDict: 可同时打击该目标的导弹型号概率字典 {pro: num_of_intercept}
        """
        # 为了计算方面，此处采用采样方式计算概率，不采用数学计算（可能存在组合较多的问题）
        # 将probDict => [pro1, pro1, pro1, pro2, pro2, pro2, ...]，用于采样
        problist = [[pro] * num for pro, num in probDict.items()]
        problist = [item for sublist in problist for item in sublist]
        attacked_prob = sum(monster.attacked_prob_list) # 已经被拦截的概率和
        rest_prob = self.quantity - float(attacked_prob) # 剩余拦截概率, 剩余拦截率覆盖此概率就可以
        if rest_prob <= np.min(problist): # 如果已经发射的弹药概率很接近超过2，则选1个小概率的去拦截就可以了
            return {np.min(problist): 1} # #如果出现多个防御塔的拦截概率一样，则选择剩余弹量多的发射 返回拦截概率和发射个数
        elif sum(problist) <= rest_prob: # 剩余弹量不多，不足以弥补剩余期望，全部发射
            return probDict
        else:
            probDict = dict()
            # 采样 从int(rest_prob) ~ min(4, len(problist)). 注：可能存在弹量不超过4的情况
            samples_list = []
            prob_s = []
            for sample_num in range(int(rest_prob), len(problist)+1): # int(rest_prob)+1 ~ len(problist) [包含]
                samples = [random.sample(problist, sample_num) for _ in range(100)]
                prob_samples = np.abs(np.array(samples).sum(axis=1) - rest_prob)
                samples_list.append(samples[np.argmin(prob_samples)])
                prob_s.append(prob_samples[np.argmin(prob_samples)])
            # 选择最小的概率
            prob_comb = samples_list[np.argmin(prob_s)] # 选择最接近rest_prob的概率组合
            for p, n in zip(*np.unique(prob_comb, return_counts=True)):
                probDict[p] = int(n)

            return probDict


    def evaluate_reward(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """


        r_tower_ammo = 0
        for tower in self.towers:
            r_tower_ammo += tower.remaining_ammo # 防御塔的消耗弹量
            # 记录剩余弹药，用于log，展示

        delta_tower_ammo = self.prev_tower_ammo - r_tower_ammo
        self.prev_tower_ammo = r_tower_ammo

        # for tower in self.towers:
        #     delta_battalion_tnt += battalion.get_reward() * 1.0  # 目前是单步
        #     # todo 确认reward是单步还是累计，需要对应处理fire_battalion
        #     battalion.reset_reward()

        shown_monsters = []
        for monster in self.monsters:
            if monster is not None:
                shown_monsters.append([monster.monster_id, monster.quantity])

        r_monster_hp = 0
        for monster in self.monsters:
            if monster is not None: # 只记录被拦截的导弹，被拦截的会设置为0.
                r_monster_hp += monster.HP * monster.quantity

        delta_monster_hp = self.prev_monster_hp - r_monster_hp
        self.prev_monster_hp = r_monster_hp

        return delta_monster_hp/self.sum_monster_hp - delta_tower_ammo/self.sum_tower_ammo, 1 - r_monster_hp/self.sum_monster_hp


    def check_done(self, dones):
        '''
        判断是否结束， 重要目标是否被打击完 or 弹药用完 or 来袭目标用完
        :return:
        '''
        # 所有怪兽都被消灭或者都到达目标
        rest_missiles = [monster for monster in self.monsters if monster is not None] # 剩余未被攻击怪兽
        mark_missiles = all(x is None for x in self.monsters) # 所有怪兽都被消灭
        mark_battalions = all(x.remaining_ammo <= 0 for x in self.towers) # 所有防御塔都没有弹药
        mark_rest_missiles = all(x.now_loc is None for x in rest_missiles) # 所有来袭导弹都到达
        for i in range(self.n_agents):
            if self.towers[i].remaining_ammo <= 0:
                dones[i] = True

        return mark_missiles or mark_battalions or mark_rest_missiles, [mark_missiles, mark_battalions, mark_rest_missiles], dones

    def check_done_v1(self):
        '''
        判断是否结束， 使用原来版本的判断方法
        :return:
        '''
        rest_missiles = [monster for monster in self.monsters if monster is not None]  # 剩余未被攻击怪兽
        return all(x is None for x in self.monsters) or all(x.now_loc is None for x in rest_missiles), []

    def update_monster_state(self):
        '''
        更新导弹状态, 例如是否击中目标, 击中目标则直接更新目标状态, 更新冷却时间
        :return:
        '''
        for monster in self.monsters:
            if monster is None:
                continue

            index = monster.monster_id
            if monster.quantity == 0: # 被击落
                self.monsters[index] = None
                continue
            if monster.now_loc is None: # 到达目标, 到达目标，monster.now_loc为None,也从missiles中删除？
                target = self.get_target(monster.targetId)
                target.attacked(monster.HP)
                # self.monsters[index] = None
                continue
            monster.update_attacked_time(self.dt)  # 更新导弹时间
            monster.update_angle()  # 更新导弹角度



    def update_tower_state(self):
        '''
        更新防御塔状态
        :return:
        '''
        # 更新视野数据,进入视野的导弹，添加到insight，离开视野的导弹，从insight中删除；同时视野内的导弹可能在执行阶段被拦截，所以需要更新
        # 同时更新 进入视野和离开视野的导弹，更新拦截目标

        for mloc in range(len(self.monsters)):
            monster = self.monsters[mloc]
            for tower in self.towers:
                # 如果missile为None, 肯定是被击落的，需要更新各个battalion的视野和拦截目标列表
                if monster is None:
                    if mloc in tower.visible_monsters:
                        tower.visible_monsters.remove(mloc)
                    if mloc in tower.attack_targets:
                        tower.remove_attack_target(mloc)
                    continue
                # 如果missile不为None, 但now_loc为None, 说明导弹已经到达目标，需要从各个battalion的视野和拦截目标列表删除
                if monster.now_loc is None:
                    if mloc in tower.visible_monsters:
                        tower.visible_monsters.remove(mloc)
                    if mloc in tower.attack_targets:
                        tower.remove_attack_target(mloc)
                    continue
                # 如果missile不为None, 且now_loc不为None, 说明导弹在飞行中，需要判断是否在视野内，以更新视野列表（添加或删除）
                dist = Geo.get_horizontal_distance(monster.now_loc, tower.location) # 计算导弹与防御塔的水平距离
                if dist < tower.range: # 如果导弹在视野内
                    if mloc not in tower.visible_monsters: # 如果导弹不在视野列表中，添加到视野内
                        tower.add_visible_monster(mloc)
                else: # 如果导弹不在视野内，可能是离开视野（删除），也可能是从未进入视野（无需操作）
                    if mloc in tower.visible_monsters: # 如果导弹在视野列表中，删除视野
                        tower.visible_monsters.remove(mloc)
                    if mloc in tower.attack_targets: # 如果导弹在拦截目标列表中，删除拦截目标
                        tower.remove_attack_target(mloc)

        # 选取拦截目标,更新env中的action_mask
        for i in range(len(self.towers)):
            if self.towers[i].remaining_ammo <= 0: # 如果防御塔没有弹药，不需要更新拦截目标, action_mask只需要全0
                self.action_mask[i] = np.zeros_like(self.action_mask[i])
                self.action_mask[i][:, 0] = 1
            else:
                self.towers[i].select_targets(self.monsters, self.assign_model_policy)
                self.action_mask[i] = self.towers[i].action_mask.copy()  # 直接赋值就可以

            self.towers[i].reset_action_mask() # 重置action_mask
        # self.action_mask = np.vstack(self.action_mask)


    # 更新汇总全局的状态
    def update_global_state(self):
        """Returns the global state.
                NOTE: This functon should not be used during decentralised execution.
                """
        # todo 为了规避init_hp==0
        for target in self.target_points:
            if target.init_HP == 0:
                target.init_HP = 1

        # 特征的数量等于属性名的数量
        nf_tower = Tower.get_ally_num_attributes()
        nf_monster = Monster.get_enemy_num_attributes()
        nf_tar = Target.get_tar_num_attributes()

        tower_state = np.zeros((self.tower_num, nf_tower-1+self.tower_num))
        monster_state = np.zeros((self.monster_wave_num, nf_monster-1+self.monster_wave_num))
        target_state = np.zeros((self.target_num, nf_tar-1+self.target_num)) # 去掉targetId, 添加one-hot

        # 计算地图中间点和边界值，用来处理归一化
        x_coords = [point[0] for point in self.border]
        y_coords = [point[1] for point in self.border]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        max_x_distance = max(x_coords) - min(x_coords)
        max_y_distance = max(y_coords) - min(y_coords)

        # 遍历防御塔
        for tower in self.towers:
            if tower.remaining_ammo > 0:
                state_b = tower.return_latest_state()

                # 归一化坐标
                state_b[1] = (state_b[1] - center_x) / max_x_distance
                state_b[2] = (state_b[2] - center_y) / max_y_distance
                # 计算并存储相对剩余弹药
                state_b = np.append(state_b, (tower.remaining_ammo / tower.ammo_capacity))

                # one-hot id值
                id_onehot = self.one_hot_encode(state_b[0], self.tower_num)
                bat_state = np.append(id_onehot, state_b[1:])

                # 赋值给tower_state
                tower_state[tower.tower_id] = bat_state

        # todo 认为导弹不应该在global state中
        # 遍历导弹
        # for monster in self.monsters:
        #     if monster.quantity > 0:
        #         state_m = monster.return_latest_state()
        #         # 归一化坐标
        #         state_m[1] = (state_m[1] - center_x) / max_x_distance
        #         state_m[2] = (state_m[2] - center_y) / max_y_distance
        #         # 计算相对剩余导弹
        #         state_m = np.append(state_m, (monster.quantity / monster.init_quantity))

        #         # one-hot tar_id值
        #         id_onsehot = self.one_hot_encode(state_m[0], self.monster_wave_num)
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
            state_t = np.append(state_t, (target.HP / target.init_HP))
            # one-hot id值
            id_onehot = self.one_hot_encode(state_t[0], self.target_num)
            state_t = np.append(id_onehot, state_t[1:])
            # 赋值给target_state
            target_state[target.targetId] = state_t

        state_dict = {"tower": tower_state, "monster": monster_state, "target": target_state}

        state_all = np.append(
            state_dict["tower"].flatten(), state_dict["target"].flatten()
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
            if target.init_HP == 0:
                target.init_HP = 1

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

        # 空数组
        enemy_obs = np.zeros((self.monster_wave_num, self.monster_feature_dim),dtype=np.float32)
        ally_obs = np.zeros((self.ally_num, self.tower_feature_dim),dtype=np.float32)
        self_obs = np.zeros((1, self.self_feature_dim),dtype=np.float32)
        target_obs = np.zeros((self.target_num, self.target_feature_dim),dtype=np.float32)


        # battalion
        agent_tower = self.towers[agent_id]
        sight_range = agent_tower.range

        # 遍历导弹
        for i in range(self.monster_wave_num):
            current_missile = self.monsters[i]

            if current_missile is not None:
                this_obs = current_missile.return_latest_obs(sight_range, agent_tower.location)
                this_obs[-2] = this_obs[-2] / self.monster_max_values['HP'] # 归一化HP
                this_obs[-3] = this_obs[-3] / self.monster_max_values['speed'] # 归一化quantity

                distance = Geo.get_horizontal_distance(current_missile.now_loc, agent_tower.location)
                distance_norm = distance / sight_range # 归一化距离
                this_obs = np.append(this_obs, distance_norm)

                if distance < sight_range:
                    this_obs = np.insert(this_obs,0 ,1)
                else:
                    this_obs = np.insert(this_obs,0 ,0)
            else:
                this_obs = np.zeros(self.monster_feature_dim)


            enemy_obs[i] = this_obs #(8,)

        # 遍历防御塔
        bat_count = 0
        for tower in self.towers:
            if tower.tower_id != agent_id:
                # 不是自己的防御塔才考虑

                distance = Geo.get_horizontal_distance(tower.location, agent_tower.location)
                if distance < sight_range:
                    # 在视野范围内
                    this_obs = tower.return_latest_obs(sight_range, agent_tower.location)
                    distance_norm = distance / sight_range # 归一化距离
                    this_obs = np.append(this_obs, distance_norm)
                    ally_obs[bat_count] = this_obs # （12，）
                bat_count += 1

        # 遍历target
        for i in range(self.target_num):
            distance = Geo.get_horizontal_distance(self.target_points[i].location, agent_tower.location)
            if distance < sight_range:
                # 在视野范围内才考虑
                this_obs = self.target_points[i].return_latest_obs(sight_range, agent_tower.location)
                distance_norm = distance / sight_range
                this_obs = np.append(this_obs, distance_norm) # （6，）

                # one-hot编码target id
                one_hot = np.zeros(self.target_num)
                one_hot[i] = 1
                this_obs = np.append(this_obs, one_hot)

                target_obs[i] = this_obs # （6+targe_num，）

        # self obs
        own_bat_obs = agent_tower.return_latest_obs(1., agent_tower.location)
        own_bat_obs = own_bat_obs[3:] # 去掉位置信息
        self_obs[0] = own_bat_obs # （7，）

        return np.concatenate((enemy_obs.flatten(),
                               ally_obs.flatten(),
                               target_obs.flatten(),
                               self_obs.flatten(),
                               ))

    def close(self):
        '''
        关闭环境
        :return:
        '''
        # 关闭UI
        if self.show_ui:
            self.close_ui()

        # 清除导弹发射位
        self.monster_spawn_list = None
        # 清楚导弹
        self.monsters = None
        # 清除防御塔
        self.towers = None
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
                    【所有怪兽都被消灭，所有防御塔都没有弹药，所有怪兽都已到达目标】
        '''
        r_battalion = []
        if done_mark[0]:
            print('所有怪兽都被消灭，奖励为1')
            for tower in self.towers:
                r_battalion.append(np.max(tower.remaining_ammo, 0)) # 剩余弹药量作为奖励（枚，0~30）
                # 重要目标HP因为是0或者负值，所以不作为奖励（也算是最终奖励）。

            assert np.sum(r_battalion) >= 0, 'r_battalion should be positive, when all monsters have been intercepted'
            # 返回击落的导弹数量占比作为奖励，意味着击落的导弹越多，奖励越高; 同时返回攻击怪兽占比，因为是所有怪兽都被消灭，因此
            return np.sum(r_battalion)/self.sum_tower_ammo + 1, 1

        if (done_mark[1] and self.monsters_Is_exist()) or done_mark[2]: # 所有防御塔都没有弹药, 其余来袭导弹会直接攻击重要目标，进行计时和汇总
            if done_mark[1] and self.monsters_Is_exist():
                print('所有防御塔都没有弹药,') # 防御塔没有弹药，需要看是否还有来袭目标，如果有，需要计算导弹弹量，
            if done_mark[2]:
                print('所有怪兽都已到达目标, 防御塔可能有剩余弹药')
            # 后续攻击加速完成
            for monster in self.monsters:
                if monster is not None: # 来袭目标没有被击落的部分
                    r_battalion.append(monster.HP * monster.quantity)
            return 1 - np.sum(r_battalion)/self.sum_monster_hp, 1 - np.sum(r_battalion)/self.sum_monster_hp  # 返回击落的导弹数量占比作为奖励，意味着击落的导弹越多，奖励越高

        else:
            print('游戏结束，但没有合适的收益清算')
            return None


    def set_norm_scale_towers(self):
        max_values = {
            "range" : [],
            "projectile_speed" : [],
            "hit_probability" : [],
            "reaction_time" : [],
            "max_attack_per_channel" : [],
            "num_channels" : []
        }
        # 获取最大值
        for tower in self.towers:
            max_values["range"].append(tower.range)
            max_values["projectile_speed"].append(tower.projectile_speed)
            max_values["hit_probability"].append(tower.hit_probability)
            max_values["reaction_time"].append(tower.reaction_time)
            max_values["max_attack_per_channel"].append(tower.max_attack_per_channel)
            max_values["num_channels"].append(tower.num_channels)
        for key in max_values.keys():
            max_values[key] = max(max_values[key])
        for tower in self.towers:
            tower.norm_scale = max_values
        return

    def set_norm_scale_monsters(self):
        max_values = {
            "speed" : [],
            "HP" : [],
        }
        # 获取最大值
        for mis in self.monsters:
            max_values['speed'].append(mis.speed)
            max_values['HP'].append(mis.HP)

        for key in max_values.keys():
            max_values[key] = max(max_values[key])

        self.monster_max_values = max_values

        return

    def monsters_Is_exist(self):
        '''
        判断导弹是否存在或到达目标
        :return:
        '''
        for monster in self.monsters:
            if monster is not None and monster.now_loc is not None:
                return True
        return False

