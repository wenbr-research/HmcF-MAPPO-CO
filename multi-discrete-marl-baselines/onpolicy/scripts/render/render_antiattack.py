#!/usr/bin/env python
"""Render Tower Defense episode as GIF using Pillow."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from onpolicy.config import get_config
from onpolicy.envs.anti_Attack.environment import Environment, km_to_lat_lon_radius
from onpolicy.envs.anti_Attack.utils import geo as Geo

def render_frame(env):
    ax=env.ax; ax.cla()
    b=env.border
    ax.plot([b[0][1],b[1][1],b[2][1],b[3][1],b[0][1]],[b[0][0],b[1][0],b[2][0],b[3][0],b[0][0]],'navy',lw=2)
    ax.scatter([t.location[1] for t in env.target_points],[t.location[0] for t in env.target_points],marker='*',s=350,c='gold',ec='darkred',lw=1.5,zorder=10)
    for t in env.target_points:
        ax.annotate(f'T{t.stronghold_id} HP:{t.hp:.0f}',(t.location[1],t.location[0]),fontsize=8,color='darkred',ha='center',fontweight='bold')
    ax.scatter([fb.tower_location[1] for fb in env.defense_towers],[fb.tower_location[0] for fb in env.defense_towers],marker='s',s=150,c='limegreen',ec='forestgreen',lw=1.5,zorder=10)
    for fb in env.defense_towers:
        ax.annotate(f'D{fb.tower_id} Am:{fb.current_ammo}',(fb.tower_location[1],fb.tower_location[0]),fontsize=7,color='darkgreen',ha='center')
        _,lr=km_to_lat_lon_radius(fb.tower_location[0],fb.attack_range)
        ax.add_patch(plt.Circle((fb.tower_location[1],fb.tower_location[0]),lr,color='green',fill=True,alpha=0.04,ec='green',lw=0.8,ls='--'))
    ax.scatter([n.location[1] for n in env.monster_nests],[n.location[0] for n in env.monster_nests],marker='^',s=100,c='crimson',ec='maroon',lw=1,zorder=10)
    for n in env.monster_nests:
        ax.annotate(f'N{n.nest_id} Q:{n.quantity}',(n.location[1],n.location[0]),fontsize=7,color='maroon',ha='center')
    for m in env.monsters:
        if m is None or m.now_loc is None: continue
        ax.plot([m.start_loc[1],m.now_loc[1]],[m.start_loc[0],m.now_loc[0]],'grey',lw=0.6,alpha=0.4,ls=':')
        ax.plot(m.now_loc[1],m.now_loc[0],'o',ms=8,color='steelblue',alpha=0.85,zorder=8)
        for fb in env.defense_towers:
            try: dist=Geo.get_horizontal_distance(m.now_loc,fb.tower_location)
            except: continue
            if dist<fb.attack_range and m.monster_id in fb.locked_targets:
                ax.plot([m.now_loc[1],fb.tower_location[1]],[m.now_loc[0],fb.tower_location[0]],'magenta',lw=1.8,alpha=0.7)
                ax.annotate(f'M{m.monster_id}',(m.now_loc[1],m.now_loc[0]),fontsize=6,color='magenta',weight='bold',ha='right')
    alive=sum(1 for m in env.monsters if m is not None and m.now_loc is not None)
    ax.set_title(f'Step {env._episode_steps} | Alive: {alive}',fontsize=12)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.grid(True,alpha=0.2); ax.set_aspect('equal')
    env.fig.canvas.draw()
    return np.asarray(env.fig.canvas.buffer_rgba())[:,:,:3]

def main():
    p=get_config()
    args=p.parse_args(['--env_name','towerDefense','--seed','2024','--episode_length','300'])
    env=Environment(args); env.seed(args.seed)
    env.fig,env.ax=plt.subplots(figsize=(12,9),dpi=80); env._episode_steps=0
    obs,_,avail=env.reset()
    frames=[]; step=0; done=False
    while not done and step<300:
        frames.append(Image.fromarray(render_frame(env)))
        acts = np.zeros((env.n_agents,), dtype=int)
        n = np.random.randint(1, min(4, env.n_agents) + 1)
        idx = np.random.choice(env.n_agents, size=n, replace=False)
        acts[idx] = np.random.randint(1, 5, n)
        obs,_,_,d_arr,_,_=env.step(acts); step+=1
        if d_arr.any(): frames.append(Image.fromarray(render_frame(env))); done=True
    env.close(); plt.close(env.fig)
    output='render_output.gif'
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=180,loop=0,optimize=False)
    print(f'Saved {len(frames)} frames -> {output} ({os.path.getsize(output)/1024:.0f}KB)')

if __name__=='__main__': main()
