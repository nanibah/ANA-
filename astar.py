import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time

from queue import PriorityQueue
global index

class Node:
    def __init__(self, x_in, y_in, theta_in, parent, g_s):
        self.x = x_in
        self.y = y_in
        self.theta = theta_in
        # self.id = id_in
        self.parent = parent
        self.g_s = g_s

    def printme(self):
        print("\tNode id", self.id,":", "x =", self.x, "y =",self.y, "theta =", self.theta, "parent:", self.parent)


def action_cost(n, m):
    t1 = (n[0] - m[0])**2 + (n[1] - m[1])**2
    t2 = min(np.abs(n[2] - m[2]), 2*np.pi - np.abs(n[2] - m[2])) * 0.25 / (np.pi/4.)
    cost = np.sqrt(t1 + t2**2)
    return cost


def heuristic(n, g, h_flag):
    if h_flag == "euclidean":
        trans = (n[0] - g[0])**2 + (n[1] - g[1])**2
        
    elif h_flag == "manhattan":
        trans = abs(n[0] - g[0]) + abs(n[1] - g[1])
        
    elif h_flag == "chebyshev":
        trans = max(abs(n[0] - g[0]), abs(n[1] - g[1]))
        
    # elif h_flag == "diagonal":
    #     t1 = abs(n[0] - g[0])
    #     t2 = abs(n[1] - g[1])
    #     t3 = np.sqrt(2) * min(t1, t2)
    #     trans= max(t1, t2, t3)
        
    elif h_flag == "octile":
        t1 = abs(n[0] - g[0])
        t2 = abs(n[1] - g[1])
        t3 = (np.sqrt(2)-1) * min(t1, t2)
        trans = max(t1, t2) + t3
    
    rot = min(np.abs(n[2] - g[2]), 2*np.pi - np.abs(n[2] - g[2])) * 0.25 / (np.pi / 4.)
    rot = rot ** 2
    h_n = np.sqrt(trans + rot)
    return h_n


def neighbours(cur_node, collision_fn, connectivity):
    x, y, theta = cur_node.x, cur_node.y, cur_node.theta
    valid_neighbours = []
    step_x, step_y = 0.25, 0.25
    del_theta = np.pi/4
    # global index
    
    if connectivity == 4:
        all_neighbours = [(x-step_x, y, theta), (x, y-step_y, theta), (x, y, theta-del_theta),
                          (x+step_x, y, theta), (x, y+step_y, theta), (x, y, theta+del_theta)]
        
    elif connectivity == 8:
        all_neighbours = [regularized_theta((x-step_x, y-step_y, theta-del_theta)), regularized_theta((x, y-step_y, theta-del_theta)), regularized_theta((x+step_x, y-step_y, theta-del_theta)), 
                          regularized_theta((x-step_x, y, theta-del_theta)), regularized_theta((x, y, theta-del_theta)), regularized_theta((x+step_x, y, theta-del_theta)),
                          regularized_theta((x-step_x, y+step_y, theta-del_theta)), regularized_theta((x, y+step_y, theta-del_theta)), regularized_theta((x+step_x, y+step_y, theta-del_theta)),
                          regularized_theta((x-step_x, y-step_y, theta)), regularized_theta((x, y-step_y, theta)), regularized_theta((x+step_x, y-step_y, theta)),
                          regularized_theta((x-step_x, y, theta)), regularized_theta((x+step_x, y, theta)), regularized_theta((x-step_x, y+step_y, theta)),
                          regularized_theta((x, y+step_y, theta)), regularized_theta((x+step_x, y+step_y, theta)), regularized_theta((x-step_x, y-step_y, theta+del_theta)),
                          regularized_theta((x, y-step_y, theta+del_theta)), regularized_theta((x+step_x, y-step_y, theta+del_theta)), regularized_theta((x-step_x, y, theta+del_theta)),
                          regularized_theta((x, y, theta+del_theta)), regularized_theta((x+step_x, y, theta+del_theta)), regularized_theta((x-step_x, y+step_y, theta+del_theta)),             
                          regularized_theta((x, y+step_y, theta+del_theta)), regularized_theta((x+step_x, y+step_y, theta+del_theta))]
    else:
        print("Wrong value passed for connectedness")
    
    for idx, neighbour in enumerate(all_neighbours):
        #if not collision_fn(neighbour): # and neighbour not in closed_set:
            # index += 1
            # print("index = ", index)
        node_g_s = cur_node.g_s + action_cost([x, y, theta], [neighbour[0], neighbour[1], neighbour[2]])
        valid_neighbours.append(Node(neighbour[0], neighbour[1], neighbour[2], cur_node, node_g_s)) #index
            ## draw_sphere_marker((neighbour[0], neighbour[1], 0.5), 0.1, (0, 0, 1, 0.5))
        # else:
            # draw_sphere_marker((neighbour[0], neighbour[1], 0.5), 0.1, (1, 0, 0, 0.5))
    
    return valid_neighbours

def regularized_theta(node):
    x, y, theta = node[0], node[1], node[2]
    while theta < 0.0:
        theta += 2*np.pi
    while theta >= 2*np.pi:
        theta -= 2*np.pi
    return (x, y, theta)


def slow_closed_set_check(succ, clos_set):
    for clos in clos_set:
        #if np.sqrt((succ[0] - clos[0])**2 + (succ[1] - clos[1]) ** 2) < 0.1:
        #    return True
        if action_cost(succ, clos) < 0.1:
            #print("hmmm")
            return True
    return False


def aStar(start, goal, collision_fn, metric):
    open_set = PriorityQueue()
    open_set.put((0, 0, Node(start[0], start[1], start[2], None, 0))) #priority no, unique_ID, node params
    closed_set = set()
    global index
    index = 0   
    
    goal_x, goal_y, goal_theta = goal[0], goal[1], goal[2]
    # goal_min = np.array([goal_x, goal_y, goal_theta]) - np.array([0.1, 0.1, 0.1])
    # goal_max = np.array([goal_x, goal_y, goal_theta]) + np.array([0.1, 0.1, 0.1])
    
    while not open_set.empty():
        priority, unique_id, node = open_set.get()
        cur_node = (node.x, node.y, node.theta)
        if slow_closed_set_check(cur_node, closed_set):
            continue
        closed_set.add(cur_node)
        
        # if (goal_min < cur_node).all() and (cur_node < goal_max).all():
        #    return reconstruct_path(node) #, path_roots)
        if action_cost(cur_node, goal) < 0.1:
            return reconstruct_path(node)
        
        possible_neighbours = neighbours(node, collision_fn, connectivity = 8)
        
        for new_node in possible_neighbours:
            successor = (new_node.x, new_node.y, new_node.theta)
            #if successor in closed_set:
             #   continue
            
            #if successor not in closed_set: # or g_n < g_of_ns[successor]:
            if not slow_closed_set_check(successor, closed_set):
                if not collision_fn(successor):
                    #slow_closed_set_check(successor, closed_set)
                    # g_n = action_cost(cur_node, successor) + node.g_s
                    g_n = new_node.g_s
                    f_n = g_n + heuristic(successor, goal, metric)
                    open_set.put((f_n, index, new_node))
                    index += 1
                # draw_sphere_marker((successor[0], successor[1], 0.5), 0.1, (0, 0, 1, 0.5))
                # path_roots[successor] = cur_node
            # else:
            #     print("Didn't enter if condition")
       
                
def reconstruct_path(goal):
    path = [(goal.x, goal.y, goal.theta)]
    cur_node = goal
    
    # i = 0
    path_cost = 0
    while cur_node.parent != None:
        # i += 1
        # print(i, "in the loop")
        parent_node = cur_node.parent
        path.append((parent_node.x, parent_node.y, parent_node.theta))
        # draw_sphere_marker((cur_node.x, cur_node.y, 0.5), 0.1, (parent_node.x, parent_node.y, 0.5))
        draw_sphere_marker((cur_node.x, cur_node.y, 0.5), 0.1, (0, 0, 0, 0.5))
        # path_cost += np.sqrt((cur_node.x - parent_node.x)**2 + (cur_node.y - parent_node.y)**2)
        path_cost += action_cost((cur_node.x, cur_node.y, cur_node.theta), (parent_node.x, parent_node.y, parent_node.theta))
        cur_node = parent_node
    print(f"Total path cost = {path_cost}")
    path.reverse()
    return path