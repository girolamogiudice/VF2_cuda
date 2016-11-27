# VF2_cuda
The VF2 algorithm is one of the most powerful algorithm for subgraph isomorphism.
Here, i would like to share the code i used for an improved version of the algorithm with CUDA.
The algorithm behaves very well with huge graph.
The code was in 2011, hence some modification in the CUDA part should be applied.
The idea is use the computational power of CUDA to filter the starting graph selecting only those edges and nodes that can perform a match. 
In principle a forrest of graphs is obtained. The forrest is then the input of the VF2 algorithm.
Unfortuately, I do not have time to work anymore on the code, hence, feel free to take the code and improve the algorithm.
Vf_cuda requeires in input a graph and a directory containing the queries. The format of the graph is the following:

GRAPH FILE FORMAT
-----------------
   #G1                     ->graph ID
   3                       ->number of vertices
   1                       ->label of each vertex
   2
   3
   3                       ->number of edges
   0 1 	                   ->end points of each edge
   0 2
   1 2

