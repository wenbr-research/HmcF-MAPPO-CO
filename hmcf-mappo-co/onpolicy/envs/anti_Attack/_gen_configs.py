import json5, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
with open('env_config.json5', 'r', encoding='utf-8') as f:
    cfg = json5.load(f)

specs = {
    6:  {'n180': 2, 'n200': 2, 'n350': 2, 'wave': 167},
    10: {'n180': 4, 'n200': 3, 'n350': 3, 'wave': 167},
    15: {'n180': 6, 'n200': 5, 'n350': 4, 'wave': 250},
    20: {'n180': 8, 'n200': 6, 'n350': 6, 'wave': 333},
}

for n, s in specs.items():
    new_cfg = {k: v for k, v in cfg.items()}
    new_cfg['tower_num'] = n
    new_cfg['monster_wave_num'] = s['wave']
    new_params = []
    for tp in cfg['tower_params']:
        n_loc = s['n180'] if tp['range'] == 180 else (s['n200'] if tp['range'] == 200 else s['n350'])
        new_tp = {k: v for k, v in tp.items()}
        new_tp['location'] = tp['location'][:n_loc]
        new_params.append(new_tp)
    new_cfg['tower_params'] = new_params
    fname = f'env_config_{n}t.json5'
    with open(fname, 'w', encoding='utf-8') as f:
        json5.dump(new_cfg, f, indent=2, ensure_ascii=False)
    print(f'{fname}: {n} towers, wave={s["wave"]}')
