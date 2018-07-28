import numpy as np


def random_point_in_circle(angle_range=(0, 2*np.pi), radius=(0, 25)):
    angle = np.random.uniform(*angle_range)
    radius = radius if np.isscalar(radius) else np.random.uniform(*radius)
    x, y = np.cos(angle) * radius, np.sin(angle) * radius
    point = np.array([x, y])
    return point

def get_random_goal_logs(paths, goal_radius, fixed_goal_position=False):
    if fixed_goal_position:
        position_slice = slice(-3, -1)
    else:
        position_slice = slice(-5, -3)

    logs = []
    if len(paths) > 0:
        progs = [
            np.linalg.norm(path["observations"][-1][position_slice]
                           - path["observations"][0][position_slice])
            for path in paths
        ]

        time_in_goals = [
            np.sum(np.linalg.norm(
                (
                    path['observations'][:, position_slice]
                    - path['env_infos']['goal_position']
                )
                , axis=1
            ) < goal_radius)
            for path in paths
        ]

        logs += [
            ('AverageProgress', np.mean(progs)),
            ('MaxProgress', np.max(progs)),
            ('MinProgress', np.min(progs)),
            ('StdProgress', np.std(progs)),

            ('AverageTimeInGoal', np.mean(time_in_goals)),
            ('MaxTimeInGoal', np.max(time_in_goals)),
            ('MinTimeInGoal', np.min(time_in_goals)),
            ('StdTimeInGoal', np.std(time_in_goals)),
        ]

        goal_positions, final_positions = zip(*[
            (p['env_infos']['goal_position'][-1],
             p['observations'][-1][position_slice])
            for p in paths
        ])

        begin_goal_distances = [
            np.linalg.norm(goal_position) for goal_position in goal_positions]
        final_goal_distances = [
            np.linalg.norm(goal_position - final_position)
            for goal_position, final_position in zip(goal_positions, final_positions)
        ]
        progress_towards_goals = [
            begin_goal_distance - final_goal_distance
            for (begin_goal_distance, final_goal_distance)
            in zip(begin_goal_distances, final_goal_distances)
        ]


        for series, name in zip((begin_goal_distances,
                                 final_goal_distances,
                                 progress_towards_goals),
                                ('BeginGoalDistance',
                                 'FinalGoalDistance',
                                 'ProgressTowardsGoal')):
            for fn_name in ('mean', 'std', 'min', 'max'):
                fn = getattr(np, fn_name)
                logs.append((fn_name.capitalize() + name, fn(series)))

    return logs

def get_multi_direction_logs(paths):
    progs = [
        np.linalg.norm(path["observations"][-1][-3:-1]
                       - path["observations"][0][-3:-1])
        for path in paths
    ]
    logs = (
        ('AverageProgress', np.mean(progs)),
        ('MaxProgress', np.max(progs)),
        ('MinProgress', np.min(progs)),
        ('StdProgress', np.std(progs)),
    )

    return logs
