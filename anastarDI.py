import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
global G, E, id_index, closed_set, g_score


class Node:
    def __init__(self, x_in, y_in, theta_in, parent, g_s, h_s, e_s):
        self.x = x_in
        self.y = y_in
        self.theta = theta_in
        self.parent = parent
        self.g_s = g_s
        self.h_s = h_s
        self.e_s = e_s

    def printme(self):
        print("x =", self.x, "y =",self.y, "theta =", self.theta, "parent:", self.parent)
        
    def __lt__(self, other):
        return self.e_s < other.e_s

def g(s, n):
    t1 = (s[0] - n[0])**2 + (s[1] - n[1])**2
    t2 = min(np.abs(s[2] - n[2]), 2*np.pi - np.abs(s[2] - n[2])) * 0.25 / (np.pi/4.)
    g_s = np.sqrt(t1 + t2**2)
    return g_s

def h(n, g, h_flag):  
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

def e_s(node):
    global G
    e_cost = (G - node.g_s) / (node.h_s + 1e-18)
    return e_cost
              
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
        # draw_sphere_marker((cur_node[2].x, cur_node[2].y, 0.5), 0.1, (parent_node.x, parent_node.y, 0.5))
        draw_sphere_marker((cur_node.x, cur_node.y, 0.5), 0.1, (0, 0, 0, 0.5))
        # path_cost += np.sqrt((cur_node.x - parent_node.x)**2 + (cur_node.y - parent_node.y)**2)
        path_cost += g((cur_node.x, cur_node.y, cur_node.theta), (parent_node.x, parent_node.y, parent_node.theta))
        cur_node = parent_node
    print(f"Total path cost = {path_cost}")
    path.reverse()
    return path

def improveSolution(open_set, start, goal, collision_fn, heuristic):
    global G, E, g_score, closed_set
    goal_min = np.array(goal) - np.array([0.1, 0.1, 0.1])
    goal_max = np.array(goal) + np.array([0.1, 0.1, 0.1])
    
    while open_set:
        max_tuple = max(open_set.values(), key=lambda x:x[0])
        s_e_s, s = max_tuple
        s_node = (s.x, s.y, s.theta)
        del open_set[s_node]
        if slow_closed_set_check(s_node, closed_set):
            continue
        closed_set.add(s_node)
                
        if s_e_s < E:
            E = s_e_s
        
        # if (goal_min < s_node).all() and (s_node < goal_max).all():
        if g(s_node, goal) < 0.1:
            G = s.g_s
            return open_set, s
        
        possible_neighbours = neighbours(s, collision_fn, start, goal, heuristic, connectivity = 8)
        
        for sucessor in possible_neighbours: 
            s_dash = (sucessor.x, sucessor.y, sucessor.theta)
            c_s_sd = g(s_node, s_dash) 
            
            if slow_closed_set_check(s_dash, closed_set):
                continue
            
            # if s.parent != None: ore
            #     s.g_s += s.parent.g_s
           
            if not collision_fn(s_dash):
                if (s.g_s + c_s_sd) < sucessor.g_s:
                    sucessor.g_s = s.g_s + c_s_sd
                    sucessor.parent = s
                    g_score[s_dash] = sucessor.g_s
                    # draw_sphere_marker((sucessor.x, sucessor.y, 0.5), 0.1, (0, 0, 0, 0.1))
                    
                    if (sucessor.g_s + sucessor.h_s) < G:
                        new_e_s = e_s(sucessor)
                        # if s_dash in open_set.keys():
                        if slow_closed_set_check(s_dash, open_set.keys()):
                            # updated_successor = Node(sucessor.x, sucessor.y, sucessor.theta, sucessor.parent, sucessor.g_s, sucessor.h_s, new_e_s)
                            # del open_set[s_dash]
                            old_e, old_node = open_set[s_dash]
                            old_node.parent = sucessor.parent
                            old_node.e_s = new_e_s
                            old_node.g_s = sucessor.g_s
                            open_set[s_dash] = (new_e_s, old_node)
                        else:
                            sucessor.e_s = new_e_s
                            open_set[s_dash] = (new_e_s, sucessor)
        
        # print(f"exited for loop, {len(open_set)}")
    return None, None

def anaStar(start, goal, collision_fn, heuristic):
    global G, E, id_index, closed_set, g_score
    G, E = 100000, 100000
    Es = []
    open_set = {}
    closed_set = set()
    g_score = dict()
    
    e_start = (G - 0) / h(start, goal, heuristic)
    start_node = Node(start[0], start[1], start[2], None, 0, h(start, goal, heuristic), e_start)
    open_set[start] = (e_start, start_node) 
    
    while open_set:
        open_set, node = improveSolution(open_set, start, goal, collision_fn, heuristic)
        Es.append(E)
        print(f"Current E-suboptimal solution = {E}")
        
        if open_set:
            reconstruct_node = node
            # path = reconstruct_path(reconstruct_node)
            open_set = updateKeys(open_set)
            open_set = prune(open_set)
            closed_set = set()
    
    # print(Es)
    path = reconstruct_path(reconstruct_node)    
    return path

def updateKeys(open_set):
    updatedSet = {}
    for key, node_info in open_set.items(): 
        e_score = e_s(node_info[1])
        node_info[1].e_s = e_score
        updatedSet[key] = (e_score, node_info[1])
    return updatedSet

def prune(open_set):
    prunedSet = {}
    for key, node_info in open_set.items():  
       node = node_info[1]
       if node.g_s + node.h_s < G:
            prunedSet[key] = node_info    
    return prunedSet

def neighbours(cur_node, collision_fn, start, goal, heuristic, connectivity):
    global id_index, g_score
    x, y, theta = cur_node.x, cur_node.y, cur_node.theta
    valid_neighbours = []
    step_x, step_y = 0.25, 0.25
    del_theta = np.pi/4
    
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
        # if not collision_fn(neighbour): # and neighbour not in closed_set:
            # if not inQueue(node, open_set)
            # print(f"index = {id_index+1}")
        g_sc = 100000
        if neighbour in g_score:
            g_sc = g_score[neighbour]
        else:
            g_score[neighbour] = g_sc
        node = Node(neighbour[0], neighbour[1], neighbour[2], None, g_sc, h(neighbour, goal, heuristic), None)
        node.e_s = e_s(node)
        valid_neighbours.append(node) #index
        # id_index += 1 
            # draw_sphere_marker((neighbour[0], neighbour[1], 0.5), 0.1, (0, 0, 1, 0.5))
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
        if g(succ, clos) < 0.1:
            #print("hmmm")
            return True
    return False