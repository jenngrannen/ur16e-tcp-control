# ur16e-tcp-control

This is a repo for controlling the bimanual UR16e arms simultaneously using sockets. It is largely adapted from the [FlingBot](https://flingbot.cs.columbia.edu/) code originally written for controlling two UR5 arms.

To run this code, install all required dependencies listed in `requirements.txt` and run `python test_pair_move.py`.

There is a gym wrapper for RL training in `gym_wrapper.py` that can be tested with `python gym_wrapper.py`.

Additionally, there is collision checking support based in PyBullet in `collision_check/check_collision.py`. This can be run to either:
* start a PyBullet visualization of a sequence of actions by running `python check_collision.py` from the `collision_check` directory and defining the desired list of actions in the main method.
* run a collision checker in the background with the `URPairSim` class by setting `vis=False` and calling the `check_collision` method.
