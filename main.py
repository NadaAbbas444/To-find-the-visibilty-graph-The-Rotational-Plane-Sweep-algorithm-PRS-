#!/usr/bin/python3

import sys
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
import csv

plt.style.use('dark_background')

def BFS_SP(graph, start, goal):
    explored = []
     
    # Queue for traversing the
    # graph in the BFS
    queue = [[start]]
     
    # If the desired node is
    # reached
    if start == goal:
        print("Same Node")
        return
     
    # Loop to traverse the graph
    # with the help of the queue
    while queue:
        path = queue.pop(0)
        node = path[-1]
        print(node)
        # Condition to check if the
        # current node is not visited
        if node not in explored:
            neighbours = graph[node]
             
            # Loop to iterate over the
            # neighbours of the node
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                 
                # Condition to check if the
                # neighbour node is the goal
                if neighbour == goal:
                    print("Shortest path = ", *new_path)
                    return new_path
            explored.append(node)
 
    # Condition when the nodes
    # are not connected
    print("So sorry, but a connecting"\
                "path doesn't exist :(")
    return
# --------------------------------------------------------------------------------------------------------------------------------------- here----------------

def CrossProduct(traversed_points):

  X1 = (traversed_points[1][0] - traversed_points[0][0])  # X in direction of vector A[1]A[0]
  Y1 = (traversed_points[1][1] - traversed_points[0][1]) # Y in direction of vector A[1]A[0]
  X2 = (traversed_points[2][0] - traversed_points[0][0])   # X in direction of vector A[2]A[0]
  Y2 = (traversed_points[2][1] - traversed_points[0][1])#Y in direction of vector A[2]A[0]
  CrossProduct = X1 * Y2 - Y1 * X2
  return(CrossProduct)

def Convex_check(shape_vertixes):

  N = len(shape_vertixes) # Stores count of edges in polygon
  prev = 0 #  direction of cross product of previos traversed edges
  curr = 0 #  direction of cross product of current traversed edges
  for i in range(N):
    temp = [shape_vertixes[i], shape_vertixes[(i + 1) % N], shape_vertixes[(i + 2) % N]]
    curr = CrossProduct(temp)
    if (curr != 0):
      if (curr * prev < 0): # If direction of cross product of all adjacent edges are not same (change the direction)
        return temp[0],temp[2]
      else:
        # Update curr
        prev = curr
  return True
# --------------------------------------------------------------------------------------------------------------------------------------- here----------------

def fetch_vertices(filename):
  '''
  Input: Filename
  Output: 
  start_p --> starting point extracted as (x,y)
  vertices --> list of all he vertices
  object_vertices --> list of vertices of each object coupled together
  '''
  objects=[] 
  vertices = []
  # Extract Start first
  with open(filename, newline='\n') as csvfile:
    csv_reader = csv.reader(csvfile,delimiter='\n')
    line_count = 0
    for row in csv_reader:
      if line_count != 0:
        thisrow = row[0].split(',')
        obj = int(thisrow[0])
        v_x = float(thisrow[1])
        v_y = float(thisrow[2])
        objects.append(obj)
        vertices.append((v_x,v_y))
      line_count += 1
  print('object IDs: ', objects)
  print('vertices from file: ', vertices)

  #objects=[0,1,1,1,1,2,2,2,2,3] ################################################################################# INPUT (1) ####################
  #vertices = [(8,10),(11,5),(12,5),(12,12), (11, 12), (13, 0), (15, 0),(15, 20), (13, 20),(18,10)] ############################### INPUT (2) ###################
  
  
  # Extract other vertices
  start_p = vertices[0]
  end_p = vertices[-1]
  
  
  object_vertices_code = {}
  for i in range(len(vertices)):
    object_vertices_code[vertices[i]] = objects[i]

  object_vertices = []
  
  n_objects = max(objects)+1

  for i in range(n_objects):
    object_vertices.append([])

  for j in range(n_objects):
    for i in range(len(vertices)):
        if objects[i] == j:
          object_vertices[j].append(vertices[i])

      


  print('------    Fetching  Vertices  ----------')
  print('start_p: ',start_p)
  print('vertices: ',vertices)
  print('object_vertices code: ',object_vertices_code)
  print('object_vertices: ',object_vertices)

  return [start_p, end_p, vertices, object_vertices_code, object_vertices]

def create_object_edges(object_vertices):
  '''
  Takes the object vertices list showing asssociation between objects and vertices and creates edges of objects
  Input: object_vertices
  Outputs: Coupled listed showing start and end of an edge (non-directional)
  edge_start --> list
  edge_end --> list
  '''
  print('------    Creating  Object Edges  ----------')
  print
  edge_start = []
  edge_end=[]
  for j in range(0,len(object_vertices)):
    for i in range(0,len(object_vertices[j])):
        start = (object_vertices[j][i])
        End = (object_vertices[j][i-1])
        if start != End:
          edge_start.append(start) 
          edge_end.append(End)

  print(edge_start) 
  print(edge_end) 
  return edge_start, edge_end

def plot_objects(edge_start,edge_end, start_p, end_p,valid_path_start=[],valid_path_end=[]):
  plt.rcParams["figure.figsize"] = [7.50, 3.50]
  plt.rcParams["figure.autolayout"] = True

  for n in range(0,len(edge_start)):
    X_value = [edge_start[n][0],edge_end[n][0]]
    #print ( X_value)
    Y_value = [edge_start[n][1],edge_end[n][1]]
    #print ( Y_value)
    plt.plot(X_value,Y_value,'*',linewidth=2, color='violet', linestyle='-',markersize=1)
    # plt.plot(X_value,Y_value,'*',linewidth=0, color='yellow', linestyle='-',markersize=12)
    # for i, j in zip(X_value, Y_value):
      # plt.text(i, j+0.5, '({}, {})'.format(i, j))
  for n_2 in range(0,len(valid_path_start)):
    X_value = [valid_path_start[n_2][0],valid_path_end[n_2][0]]
    Y_value = [valid_path_start[n_2][1],valid_path_end[n_2][1]]
    plt.plot(X_value,Y_value,linewidth=1.5, color='cyan', linestyle=':',markersize=12)

  X_value = [start_p[0],start_p[0]]
  Y_value = [start_p[1],start_p[1]]
  plt.plot(X_value,Y_value,'*',linewidth=2, color='yellow', linestyle='-',markersize=12)

  X_value = [end_p[0],end_p[0]]
  Y_value = [end_p[1],end_p[1]]
  plt.plot(X_value,Y_value,'*',linewidth=2, color='yellow', linestyle='-',markersize=12)

  for n in range(0,len(edge_start)):
    X_value = [edge_start[n][0],edge_end[n][0]]
    #print ( X_value)
    Y_value = [edge_start[n][1],edge_end[n][1]]
    #print ( Y_value)
    # plt.plot(X_value,Y_value,'*',linewidth=2, color='violet', linestyle='-',markersize=1)
    plt.plot(X_value,Y_value,'*',linewidth=0, color='yellow', linestyle='-',markersize=12)
  
  
  plt.savefig('destination_path.png', format='png')
  plt.show() 
  

def intersect(p1, p2, p3, p4):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return False
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return False
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return False
    x = round(x1 + ua * (x2-x1),2)
    y = round(y1 + ua * (y2-y1),2)
    return True


def Brute_Force_VG(filename):
  [start_p, end_p, vertices,object_vertices_code, object_vertices] = fetch_vertices(filename)
  [edge_start, edge_end] = create_object_edges(object_vertices)
  # plot_objects(edge_start,edge_end,start_p, end_p)
  valid_path_start = []
  valid_path_end = []
  intersected=False
  #curr_p = vertices[2]

  # This ensures the see abck functionality. 
  # If we have already seen from a to be, we need not see back from b to a at all
  # curr_idx = 0

  for curr_p in vertices:
    print('current point ',curr_p)
    # curr_idx = curr_idx + 1
    for curr_vertex in vertices:#[curr_idx-1:]:
      
      if object_vertices_code[curr_p] != object_vertices_code[curr_vertex]:
        #print("curr pt and curr vertex are not on same object")
      # If my current point is not current vertex 
      # AND 
      # current point and current vertex  
      # Make a segment ad check the intersection with all edges
        intersected==False
        for edge_i in range(len(edge_start)):
          # iterate over edges to see if the segment intersects any

          if curr_vertex != edge_start[edge_i] and curr_vertex != edge_end[edge_i] and\
              curr_p != edge_start[edge_i] and curr_p != edge_end[edge_i]:
            # if the current vertex and current point is not a part of edge, only then check intersection with that edge
            #print('checking intersection between ', curr_p, curr_vertex, ' and edge ', edge_start[edge_i],edge_end[edge_i])
            intersected = intersect(curr_p, curr_vertex, \
                                          edge_start[edge_i], edge_end[edge_i])
            if intersected:
                # print('intersection: ', intersected)
                break

        if intersected==False:
            #print('adding path ','from', curr_p, ' to ', curr_vertex)
            valid_path_start.append(curr_p)
            valid_path_end.append(curr_vertex)
        else:
            pass
            #print('IN VALID PATH, SORRY... ','from', curr_p, ' to ', curr_vertex)

  for edge_i in range(len(edge_start)):
    valid_path_start.append(edge_start[edge_i])
    valid_path_end.append(edge_end[edge_i])
  
  for r in range(0,len(object_vertices)):
    shape_vertixes = object_vertices[r] 
    ConvexCheck = Convex_check(shape_vertixes)
    if ConvexCheck == True:
      pass
    else:
      st=ConvexCheck[0][0],ConvexCheck[0][1]
      en=ConvexCheck[1][0],ConvexCheck[1][1]
      valid_path_start.append(st)
      valid_path_end.append(en)
  
  
  
  plot_objects(edge_start,edge_end, start_p, end_p,valid_path_start,valid_path_end)
  
  # Create graph format
  graph = {}
  for v in vertices:
    graph[v] = []

  # print(graph)

  for i in range(len(valid_path_start)):
    
    curr_neighbours = graph[valid_path_start[i]]
    if valid_path_end[i] not in curr_neighbours and valid_path_start[i] != valid_path_end[i]:
      graph[valid_path_start[i]].append(valid_path_end[i])

    curr_neighbours = graph[valid_path_end[i]]
    if valid_path_start[i] not in curr_neighbours and valid_path_start[i] != valid_path_end[i]:
      graph[valid_path_end[i]].append(valid_path_start[i])

  print('graph: ', graph)

  return valid_path_start,valid_path_end, graph, start_p, end_p, edge_start, edge_end


def find_path(graph,start_p,end_p,edge_start,edge_end):
  traverse_path_x = []
  traverse_path_y = []
  print(graph)
  waypoints = BFS_SP(graph, start_p, end_p)
  print('waypts', waypoints)
  for waypt_idx in range(len(waypoints)-1):
    traverse_path_x.append(waypoints[waypt_idx])
    traverse_path_y.append(waypoints[waypt_idx+1])
  print('short path',traverse_path_x)
  print('short path',traverse_path_y)
  plot_objects(edge_start,edge_end, start_p, end_p,traverse_path_x,traverse_path_y)
  print(start_p,end_p)
  return waypoints

def CCW(points, centre, object_vertices_code): # points = [()()()()], centre = ()
  if centre:
    centre_x, centre_y = centre
  else:
    centre_x, centre_y = sum([x for x,_ in points])/len(points), sum([y for _,y in points])/len(points)
  angles = [math.atan2(y - centre_y, x - centre_x) for x,y in points]
  counterclockwise_indices = sorted(range(len(points)), key=lambda i: angles[i])
  counterclockwise_points = [points[i] for i in counterclockwise_indices]
  sorted_points_list=counterclockwise_points
  sorted_points_list_NEW = []

  for vert in range(len(sorted_points_list)):
    # print(centre)
    # print(centre)
    #if object_vertices_code[centre] != object_vertices_code[sorted_points_list[vert]]:
    sorted_points_list_NEW.append(sorted_points_list[vert])

  return sorted_points_list_NEW

def union_calc(edge_start,edge_end,vertices):
  print('------   Finding  Unions between vertex and edges  ----------')
  union_list = {}
  for m in range(1,len(vertices)-1):
    #print('for point',m+1,'which is',edge_start[m],'the union of',edge_start.index(edge_start[m])+1,'&',edge_end.index(edge_start[m])+1)
    #print(vertices[m])
    #print(edge_start[vertices[m]])
    union_pair=(edge_start.index(vertices[m]),edge_end.index(vertices[m]))
    union_list[vertices[m]] = union_pair
  union_list[vertices[0]] = []
  union_list[vertices[-1]] = []
  return union_list


def RPS(filename):
  [start_p, end_p, vertices,object_vertices_code, object_vertices] = fetch_vertices(filename)
  [edge_start, edge_end] = create_object_edges(object_vertices)
#   plot_objects(edge_start,edge_end,start_p, end_p)
  valid_path_start = []
  valid_path_end = []
  intersected=False
  #curr_p = vertices[2]
  vertices = vertices

  associated_union = union_calc(edge_start,edge_end,vertices)

  for curr_p in vertices: ## SELECT A PT
    S=set({})
    print('current point ',curr_p)
    first_pt = True
    # Compute the new sorted vertices
    sorted_vertices = CCW(vertices, curr_p, object_vertices_code)


    for curr_vertex in sorted_vertices: ## MAKE SEGMENTS WITH OTHER PTS
      # Iterate over all vertices  
        intersected==False

        if first_pt: # if its te first vertex, check intersection
            print("CHECKING FIRST INTERSECTION")
            for edge_i in range(len(edge_start)):
            # iterate over edges to see if the segment intersects any
                if object_vertices_code[curr_vertex] != object_vertices_code[curr_p]:
                    if curr_vertex != edge_start[edge_i] and curr_vertex != edge_end[edge_i] and\
                        curr_p != edge_start[edge_i] and curr_p != edge_end[edge_i]:
                        # if the current vertex and current point is not a part of edge, only then check intersection with that edge
                        #print('checking intersection between ', curr_p, curr_vertex, ' and edge ', edge_start[edge_i],edge_end[edge_i])
                        intersected = intersect(curr_p, curr_vertex, \
                                                    edge_start[edge_i], edge_end[edge_i])
                        if intersected:
                            # print('intersection: ', intersected)
                            S.add(edge_i)
                            intersected = False

            first_pt = False
            print('S after first intersection', S)
            intersected = False
        #### FUNCTIONALITY OF S
        # Create the S list with current union
        print('associated union ',associated_union)
        print('------------------------- ')
        print('curr_vertex ',curr_vertex)
        curr_union = associated_union[curr_vertex]
        print('cur union', curr_union)
        if curr_union!=[]:
            if curr_union[0] not in S:
                S.add(curr_union[0])
            else:
                S.remove(curr_union[0])

            if curr_union[1] not in S:
                S.add(curr_union[1])
            else:
                S.remove(curr_union[1])
        if curr_union != []:
            print("UNION", edge_start[curr_union[0]], edge_end[curr_union[0]])
            print("UNION", edge_start[curr_union[1]], edge_end[curr_union[1]])
        print('S: ', S)

        for i in S:
            print('S ABsolute: ', edge_start[i], edge_end[i])

        #Check intersection between curr_segment (curr_pt, vertices[curr_vertex]) and S list
        intersected = False

        for curr_segment in S:
        #   print('curr_segment id',curr_segment)
        #   print('curr_segment val',edge_start[curr_segment-1],edge_end[curr_segment-1])
          print("CURRENT VERTEX ", curr_vertex)
          #ensure that we dont create intersection with unions of that vertex
          if curr_vertex != edge_start[curr_segment] and curr_vertex != edge_end[curr_segment] and\
                    curr_p != edge_start[curr_segment] and curr_p != edge_end[curr_segment]:
            intersected = intersect(curr_p, curr_vertex, \
                                        edge_start[curr_segment], edge_end[curr_segment])
                # print('intersection: ', intersected)
            if intersected:
                print('intersection of: ', curr_p, curr_vertex, 'with',\
                    edge_start[curr_segment], edge_end[curr_segment])
                break
        print('intersection: ', intersected)
        if intersected==False:
          print(' is valid.')
          #print('adding path')
          if object_vertices_code[curr_vertex] != object_vertices_code[curr_p] and\
            curr_vertex != start_p:
            valid_path_start.append(curr_p)
            valid_path_end.append(curr_vertex)
        else:
          print(' is IN VALID SORRY.')
        print("PATH START ", valid_path_start)
        print("PATH END ", valid_path_end)



  # Add edges as paths
  for edge_i in range(len(edge_start)):
    valid_path_start.append(edge_start[edge_i])
    valid_path_end.append(edge_end[edge_i])
  #plot_objects(edge_start,edge_end, start_p, end_p,valid_path_start,valid_path_end)

# --------------------------------------------------------------------------------------------------------------------------------------- here----------------
  for r in range(0,len(object_vertices)):
    shape_vertixes = object_vertices[r] 
    ConvexCheck = Convex_check(shape_vertixes)
    if ConvexCheck == True:
      pass
    else:
      st=ConvexCheck[0][0],ConvexCheck[0][1]
      en=ConvexCheck[1][0],ConvexCheck[1][1]
      valid_path_start.append(st)
      valid_path_end.append(en)
  plot_objects(edge_start,edge_end, start_p, end_p,valid_path_start,valid_path_end)
  # --------------------------------------------------------------------------------------------------------------------------------------- here----------------

  
  # Create graph format
  graph = {}
  for v in vertices:
    graph[v] = []

  print('graph',graph)

  for i in range(len(valid_path_start)):
    
    curr_neighbours = graph[valid_path_start[i]]
    if valid_path_end[i] not in curr_neighbours and valid_path_start[i] != valid_path_end[i]:
      graph[valid_path_start[i]].append(valid_path_end[i])

    curr_neighbours = graph[valid_path_end[i]]
    if valid_path_start[i] not in curr_neighbours and valid_path_start[i] != valid_path_end[i]:
      graph[valid_path_end[i]].append(valid_path_start[i])

  print('graph: ', graph)
  
  return valid_path_start,valid_path_end, graph, start_p, end_p, edge_start, edge_end


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\n ${} file_name.csv".format(sys.argv[0]))
    else:
        filename = sys.argv[1]
        [valid_path_start,valid_path_end,graph,start_p, end_p, edge_start, edge_end] = RPS(filename)
        # [valid_path_start,valid_path_end,graph,start_p, end_p, edge_start, edge_end] = Brute_Force_VG(filename)
        # waypoints = find_path(graph,start_p,end_p,edge_start,edge_end)

