import numpy as np


def random_point_on_circle(angle_range=(0, 2*np.pi), radius=5):
    angle = np.random.uniform(*angle_range)
    x, y = np.cos(angle) * radius, np.sin(angle) * radius
    point = np.array([x, y])
    return point

def get_random_goal_logs(paths):
    logs = []
    if len(paths) > 0:
        progs = [
            np.linalg.norm(path["observations"][-1][-5:-3]
                           - path["observations"][0][-5:-3])
            for path in paths
        ]
        logs += [
            ('AverageProgress', np.mean(progs)),
            ('MaxProgress', np.max(progs)),
            ('MinProgress', np.min(progs)),
            ('StdProgress', np.std(progs)),
        ]

        goal_positions, final_positions = zip(*[
            (p['observations'][-1][-2:], p['observations'][-1][-5:-3])
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
