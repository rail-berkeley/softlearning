from softlearning.utils.misc import PROJECT_PATH


def get_git_rev(path=PROJECT_PATH, search_parent_directories=True):
    try:
        import git
    except ImportError:
        print(
            "Warning: gitpython not installed."
            " Unable to log git rev."
            " Run `pip install gitpython` if you want git revs to be logged.")
        return None

    try:
        repo = git.Repo(
            path, search_parent_directories=search_parent_directories)
        git_rev = repo.active_branch.commit.name_rev
    except TypeError:
        git_rev = repo.head.object.name_rev

    return git_rev
