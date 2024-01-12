import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
from queue import PriorityQueue
from astar import aStar
from anastarDI import anaStar

def main(env, algorithm, start, goal, metric):
    connect(use_gui=True)
    robots, obstacles = load_env(env)
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]
    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    
    path = []
    start_time = time.time()
    if algorithm == "A":
        path = aStar(start, goal, collision_fn, metric)
    elif algorithm == "ANA":
        path = anaStar(start, goal, collision_fn, metric)
    print("Planner run time: ", time.time() - start_time)

    if path == None:
        print("No Solution Found")
    
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    time.sleep(0.1)
    # wait_if_gui()
    disconnect()
    
if __name__ == "__main__":
    print("Expected time to run is about 28 mins. Three environments with two different heuristics will be executed for both A* and ANA*")
    
    start_time = time.time()
    env = {"env1.json":[(-2.1,2.1,0),(1.65, -1.85, np.pi/2)], 
           "env2.json":[(-2.1,-2.1,0),(1.65, -1.85, np.pi/2)],
           "env3.json":[(-3.4,-1.4,0),(2.6, -1.3, -np.pi/2)]}
    
    algorithm = ["ANA", "A"]
    metrics = ["euclidean", "octile"]
    
    for key, value in env.items():
        env_path = key
        start, goal = value[0], value[1]
        
        for heuristic in metrics:
            for planner in algorithm:
                print(f"Running {planner}* with {heuristic} for {env_path}")
                main(env_path, planner, start, goal, heuristic)
    
    total_time = (time.time()-start_time)/60
    print(f"Total time taken to run demp.py = {total_time} minutes", )
    
