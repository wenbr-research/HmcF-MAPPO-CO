"""Export one complete, anonymized anti-attack episode as an animated GIF.

The default all-fire action is deliberately deterministic: this tool visualizes
the environment dynamics, not a trained MAPPO policy replay.
"""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from PIL import Image

from onpolicy.envs.anti_Attack.environment import Environment


def parse_args():
    parser = argparse.ArgumentParser(description="Export an interception episode GIF.")
    parser.add_argument("--output", default="visualizations/interception_episode.gif")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max_steps", type=int, default=240)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--frame_duration_ms", type=int, default=90)
    parser.add_argument("--width", type=int, default=1600,
                        help="GIF frame width in pixels.")
    parser.add_argument("--height", type=int, default=1000,
                        help="GIF frame height in pixels.")
    parser.add_argument("--dpi", type=int, default=100,
                        help="Matplotlib render resolution.")
    return parser.parse_args()


def environment_args():
    return SimpleNamespace(
        stacked_frames=1,
        use_stacked_frames=False,
        add_center_xy=True,
        use_obs_instead_of_state=False,
        use_state_agent=True,
        add_local_obs=False,
        add_visible_state=False,
        show_ui=True,
        add_last_action=False,
        assign_model_policy="distance",
        lunch_model_bal=False,
    )


def capture_frame(env):
    env.render()
    env.ax.set_axis_off()
    env.fig.tight_layout(pad=0)
    env.fig.canvas.draw()
    rgba = np.asarray(env.fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[..., :3]).convert("P", palette=Image.Palette.ADAPTIVE)


def main():
    args = parse_args()
    env = Environment(environment_args())
    env.seed(args.seed)

    env.reset()
    env.fig.set_size_inches(args.width / args.dpi, args.height / args.dpi)
    env.fig.set_dpi(args.dpi)

    frames = [capture_frame(env)]
    action = np.ones((env.n_agents, env.action_space_agent), dtype=np.int64)
    for step in range(1, args.max_steps + 1):
        _, _, _, dones, _, _ = env.step(action)
        if step % args.frame_stride == 0 or np.all(dones):
            frames.append(capture_frame(env))
        if np.all(dones):
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=args.frame_duration_ms,
        loop=0,
        optimize=False,
    )
    env.close()
    print(
        f"Saved {len(frames)} {args.width}x{args.height} frames to {output} "
        f"after {step} environment steps."
    )


if __name__ == "__main__":
    main()
