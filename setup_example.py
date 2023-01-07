from dolfin import *
from mshr import *
import numpy as np

class Example(object):
    pass

class Example1(Example):

    def __init__(self):

        self.eta = Constant(0.6)
        self.gamma = 5.e-2             # barrier parameter
        self.intensity = Constant(1.0)

        self.T = 9
        self.N = 300
        
        room = Rectangle(Point(-0.1, -1.), Point(8.5, 3.))
        wall1 = Rectangle(Point(0.1, -0.3), Point(2.0, 0.0))
        wall2 = Rectangle(Point(2.3, -0.3), Point(5.5, 0.0))
        wall3 = Rectangle(Point(0.1, -0.3), Point(0.4, 2.3))
        wall4 = Rectangle(Point(0.1, 2.0), Point(3.5, 2.3))
        wall5 = Rectangle(Point(3.8, 2.0), Point(5.5, 2.3))
        
        geometry = room - wall1 - wall2 - wall3 - wall4 - wall5 # - wall6
        
        self.mesh = generate_mesh(geometry, 32)

        def wall_region(x):
            return (x[0] < 6.2 and x[0] > -0.1+DOLFIN_EPS \
                    and x[1] < 2.0-DOLFIN_EPS and x[1] > -1.+DOLFIN_EPS)

        def inside_room(x):
            return (x[0] < 5.9 and x[0] > 0.0 \
                    and x[1] < 2.4 and x[1] > -0.4)
        
        class Exits(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and not inside_room(x)

        class Walls(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and inside_room(x)
            
        self.exits = Exits()
        self.walls = Walls()
        self.inside_room = inside_room
        self.wall_region = wall_region
        
        self.rho_0 = Expression(('0.7*exp(-(pow(x[0]-1.3,2) + pow(x[1]-1.3,2))/0.12) \
        + 0.7*exp(-(0.5*pow(x[0]-4, 2) + pow(x[1]-0.7,2))/0.12) \
        + 0.7*exp(-(pow(x[0]-3.,2) + pow(x[1]-1.2,2))/0.12) \
        + 0.7*exp(-(pow(x[0]-2.,2) + pow(x[1]-1.8,2))/0.12) \
        + 0.7*exp(-(pow(x[0]-1.2,2) + pow(x[1]-0.7,2))/0.12) \
        + 0.7*exp(-(pow(x[0]-2.5,2) + pow(x[1]-1.2,2))/0.12)'), degree=2)
       
        self.ag_pos_0 = np.array([[3.3, 1.4]  ,[2.0, 0.7], [1.1, 1.0]])# ,[3,0.9]       
        self.ag_vel_0 = np.array([[0.5, -0.1] ,[0.6, 0.05], [0.8, -0.05]]) #,[0.5,0.0]        

class Example2(Example):

    def __init__(self):

        # Parameters
        self.eta = Constant(1.5)
        self.intensity = Constant(0.5)
        self.gamma = 1.e-1             # barrier parameter

        self.T = 12
        self.N = 300
        
        room = Rectangle(Point(-0.5,-0.5), Point(7.5,7.5))
        wall1 = Rectangle(Point(0.5,2), Point(0.9, 3.4))
        wall2 = Rectangle(Point(0.5,3.6), Point(0.9, 5))

        wall3 = Rectangle(Point(0.5,2), Point(4.4, 2.4))
        wall4 = Rectangle(Point(4,0.5), Point(4.4, 2.4))

        wall5 = Rectangle(Point(0.5,4.6), Point(4.4, 5))
        wall6 = Rectangle(Point(4,5), Point(4.4, 6.5))

        wall7 = Rectangle(Point(6,0.5), Point(6.4, 6.5))        

        geometry = room - wall1 - wall2 - wall3 - wall4 - wall5 - wall6 - wall7
        
        self.mesh = generate_mesh(geometry, 32)

        def wall_region(x):
            return (x[0] < 7.+DOLFIN_EPS and x[0] > 0.5-DOLFIN_EPS \
                    and x[1] < 7.0+DOLFIN_EPS and x[1] > 0.-DOLFIN_EPS)

        def inside_room(x):
            return (x[0] > 0.5 and x[0] < 4.4 and x[1] > 2 and x[1] < 5) \
                    or (x[0] > 4.0 and x[0] < 6 and x[1] > 1.0 and x[1] < 6.0)
        
        class Exits(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and not wall_region(x)

        class Walls(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and wall_region(x)
            
        self.exits = Exits()
        self.walls = Walls()
        self.inside_room = inside_room
        self.wall_region = wall_region

        self.rho_0 = Expression(('0.7*exp(-(pow(x[0]-1.2,2) + pow(x[1]-3.5,2))/0.1) \
                + 0.7*exp(-(pow(x[0]-2.2,2) + pow(x[1]-4.0,2))/0.1) \
                + 0.7*exp(-(pow(x[0]-2.5,2) + pow(x[1]-3.2,2))/0.1) \
                + 0.7*exp(-(pow(x[0]-3.0,2) + pow(x[1]-3.8,2))/0.1) \
                + 0.7*exp(-(pow(x[0]-2.3,2) + pow(x[1]-2.7,2))/0.1) \
                + 0.7*exp(-(pow(x[0]-3.3,2) + pow(x[1]-3.1,2))/0.1) \
                + 0.4*exp(-(pow(x[0]-3.5,2) + pow(x[1]-3.5,2))/0.01) \
                '), degree=2)

        # 2 Agents
        self.ag_pos_0 = np.array([[2.4,3.0],[2.7, 4.0]]) # ,[0.1,0.8]])
        self.ag_vel_0 = np.array([[0.15,-0.01],[0.15, 0.01]]) # ,[0.5,0.0]])

class Example3(Example):

    def __init__(self):

        # Parameters
        self.eta = Constant(1.5)
        self.intensity = Constant(1.0)
        self.gamma = 2.e-2             # barrier parameter
        
        self.T = 15        
        self.N = 300
        
        room = Rectangle(Point(0,-1.5), Point(9,5))
        wall1 = Rectangle(Point(1,1), Point(3.6, 1.4))
        wall1a = Rectangle(Point(4.6,1), Point(8, 1.4))

        # Slit left
        wall2 = Rectangle(Point(1,1), Point(1.4, 1.9))
        wall3 = Rectangle(Point(1,2.1), Point(1.4, 3))
        
        wall4 = Rectangle(Point(1, 2.6), Point(3.6, 3))
        wall5 = Rectangle(Point(4.6, 2.6), Point(8, 3))        
        
        wall6 = Rectangle(Point(3.2,3), Point(3.6, 4.5))
        wall7 = Rectangle(Point(3.6, 4.1),Point(4.6, 4.5))
        wall8 = Rectangle(Point(4.6, 3),Point(5, 4.5))

        wall9 = Rectangle(Point(3.2,-1), Point(3.6, 1))
        wall10 = Rectangle(Point(3.6, -0.6),Point(4.6, -1))
        wall11 = Rectangle(Point(4.6, -1),Point(5, 1))

        
        geometry = room - wall1 - wall1a - wall2 - wall3 - wall4 - wall5 - wall6 - wall7 - wall8 - wall9 - wall10 - wall11
        
        self.mesh = generate_mesh(geometry, 32)

        def wall_region(x):
            return (x[0] < 8.5+DOLFIN_EPS and x[0] > 1.0-DOLFIN_EPS \
                    and x[1] < 4.5+DOLFIN_EPS and x[1] > -1.0-DOLFIN_EPS)

        def inside_room(x):
            return (x[0] > 2 and x[0] < 6 and x[1] > -1.5 and x[1] < 4.5)
        
        class Exits(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and not wall_region(x)

        class Walls(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and wall_region(x)
            
        self.exits = Exits()
        self.walls = Walls()
        self.inside_room = inside_room
        self.wall_region = wall_region

        self.rho_0 = Expression(('(x[0] < 4.4 && x[0] > 3.8 && x[1] > 0.1 && x[1] < 3.8)? 0.8 : 0.'), degree=2)

        # 2 Agents
        self.ag_pos_0 = np.array([[4.7,1.7]]) # ,[0.1,0.8]])
        self.ag_vel_0 = np.array([[0.4,0.05]]) # ,[0.5,0.0]])
        

class Example4(Example):

    def __init__(self):

        # Parameters
        self.eta = Constant(1.)
        self.intensity = Constant(0.5)
        self.gamma = 5.e-2             # barrier parameter

        self.T = 10
        self.N = 300
        
        room = Rectangle(Point(0,0), Point(6.5,6.5))

        # Bottom
        wall1 = Rectangle(Point(1,1), Point(2.5, 1.4))
        wall2 = Rectangle(Point(2.8,1), Point(5.4, 1.4))

        # Left
        wall3 = Rectangle(Point(1,1), Point(1.4, 5.4))

        # Top
        wall4 = Rectangle(Point(1,5), Point(2.5, 5.4))
        wall5 = Rectangle(Point(3.5,5), Point(5.4, 5.4))

        # Right
        wall6 = Rectangle(Point(5,1), Point(5.4, 3.6))
        wall7 = Rectangle(Point(5,4.2), Point(5.4, 5.4))        
               
        geometry = room - wall1 - wall2 - wall3 - wall4 - wall5 - wall6 - wall7
        
        self.mesh = generate_mesh(geometry, 32)

        def wall_region(x):
            return (x[0] < 5.5 and x[0] > 0.5 \
                    and x[1] < 5.5 and x[1] > 0.5)

        def inside_room(x):
            return (x[0] > 1 and x[0] < 5.4 and x[1] > 1 and x[1] < 5.4)
        
        class Exits(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and not wall_region(x)

        class Walls(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and wall_region(x)
            
        self.exits = Exits()
        self.walls = Walls()
        self.inside_room = inside_room
        self.wall_region = wall_region

        self.rho_0 = Expression(('0.8*exp(-(pow(x[0]-1.5,2) + pow(x[1]-1.5,2))/0.2) \
        + 0.8*exp(-(pow(x[0]-2.5,2) + pow(x[1]-1.5,2))/0.2) \
        + 0.8*exp(-(pow(x[0]-2.3,2) + pow(x[1]-2.5,2))/0.2) \
        + 0.8*exp(-(pow(x[0]-3.9,2) + pow(x[1]-2.2,2))/0.2) \
        + 0.8*exp(-(pow(x[0]-3.7,2) + pow(x[1]-1.5,2))/0.2) \
        + 0.2*exp(-(pow(x[0]-3,2) + pow(x[1]-2.5,2))/2.)'), degree=2)

        # 2 Agents
        self.ag_pos_0 = np.array([[2.,3.0],[3.3,3.1]])
        self.ag_vel_0 = np.array([[0.10,0.1], [0.05,0.0]])

        
def create_wall(x1, y1, x2, y2):
    wall_thick = 0.1
    if x1 == x2:
        # vertical wall
        return Rectangle(Point(x1,y1), Point(x1+wall_thick,y2))
    elif y1 == y2:
        # horizontal wall
        return Rectangle(Point(x1,y1), Point(x2,y2+wall_thick))
    else:
        return Rectangle(Point(x1,y1), Point(x2,y2))
    
# Scaling: 0.1 == 1m in reality
class ExampleOrangerie(Example):
       
    def __init__(self):

        # Parameters
        self.eta = Constant(1.)
        self.intensity = Constant(0.5)
        self.gamma = 1.e-1             # barrier parameter      
    
        room = Rectangle(Point(-0.5, -0.5), Point(8.2, 6.5))

        walls = [];
        # Lower walls
        walls.append(create_wall(0,0,1.5,0))
        walls.append(create_wall(1.5,0,1.5,1))
        walls.append(create_wall(1.5,1,2.8,1))
        walls.append(create_wall(3,1,3.4,1))
        walls.append(create_wall(3.6,1,4,1))
        walls.append(create_wall(4.2,1,4.6,1))
        walls.append(create_wall(4.8,1,5.2,1))
        walls.append(create_wall(5.4,1,5.8,1))
        walls.append(create_wall(6,1,7.2,1))

        # Right walls
        walls.append(create_wall(7.2,1,7.2,2))
        walls.append(create_wall(7.2,2.4,7.2,2.6))
        walls.append(create_wall(7.2,3,7.2,3.2))
        walls.append(create_wall(7.2,3.6,7.2,5))

        # Upper walls
        walls.append(create_wall(7.2,5,7.7,5))
        walls.append(create_wall(7.7,5,7.7,6))
        walls.append(create_wall(7.8,6,0,6))

        # Left wall
        walls.append(create_wall(0,6,0,0))

        # Room 1
        walls.append(create_wall(1.2,0,1.2,0.2))
        walls.append(create_wall(1.2,0.4,1.2,1))
        walls.append(create_wall(1.2,1.2,1.2,1.6))
        walls.append(create_wall(0,1.6,1.3,1.6))        
        
        # Room 2
        walls.append(create_wall(0,1.9,1.7,1.9))
        walls.append(create_wall(1.6,2.2,1.6,2.6))
        walls.append(create_wall(1.6,2.8,1.6,4.0))
        walls.append(create_wall(0.3,3.9,1.6,3.9))
        walls.append(create_wall(0,4.3,1.6,4.3))

        walls.append(create_wall(1.6,2.2,2,2.2))
        walls.append(create_wall(2.2,2.2,2.4,2.2))
        walls.append(create_wall(2.4,2.2,2.4,4))
        walls.append(create_wall(1.6,3.9,2.4,3.9))
                
        # Room 3
        walls.append(create_wall(0,4.7,1.4,4.7))
        walls.append(create_wall(1.3,5,1.3,5.5))
        walls.append(create_wall(1.3,5.7,1.3,6))

        # Room 4 (stairs and lift)
        walls.append(create_wall(1.7,5,1.7,6))
        walls.append(create_wall(2.1,5,2.1,6))
        walls.append(create_wall(2.5,5,2.5,6))

        # Room 5 + 6
        walls.append(create_wall(2.5,5,2.7,5))
        walls.append(create_wall(2.9,5,4.3,5))
        walls.append(create_wall(3.6,5,3.6,6))
        walls.append(create_wall(4.5,5,4.6,5))
        walls.append(create_wall(4.6,5,4.6,6))

        # Room 7 + 8
        walls.append(create_wall(5.2,5,5.2,6))
        walls.append(create_wall(5.2,5,5.4,5))
        walls.append(create_wall(5.6,5,6.8,5))
        walls.append(create_wall(7,5,7.5,5))
        walls.append(create_wall(6.2,5,6.2,6))
        
        # Obstacles
        walls.append(create_wall(4,2,6,2.3))
        walls.append(create_wall(4,3.7,6,4))
    
        geometry = room
        for wall in walls:
            geometry = geometry - wall

        self.mesh = generate_mesh(geometry, 40)

        def wall_region(x):
            return (x[0] < 8.1 and x[0] > -DOLFIN_EPS \
                    and x[1] < 6.2 and x[1] > -DOLFIN_EPS)
        
        def inside_room(x):
            return (x[0] < 8.1 and x[0] > -DOLFIN_EPS \
                    and x[1] < 6.2 and x[1] > -DOLFIN_EPS)
        
        class Exits(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and not wall_region(x)

        class Walls(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and wall_region(x)            
        
        class Exits(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and not inside_room(x)
                
        self.exits = Exits()
        self.walls = Walls()
        self.inside_room = inside_room
        self.wall_region = wall_region
        
        # self.rho_0 = Expression(('(pow(x[0]-1.5,2.) + 3*pow(x[1]-1.2,2.) < 0.35 || pow(x[0]-4,2.) + 3*pow(x[1]-0.7,2.) < 0.35) ? 0.8 : 0.0'), degree=2)
        self.rho_0 = Expression(('0.8*exp(-(pow(x[0]-1,2) + pow(x[1]-3,2))/0.1)'
                                 '+ 0.8*exp(-(pow(x[0]-0.6,2) + pow(x[1]-0.7,2))/0.1)'
                                 '+ 0.8*exp(-(pow(x[0]-0.6,2) + pow(x[1]-5.5,2))/0.05)'
                                 '+ 0.8*exp(-(pow(x[0]-3.2,2) + pow(x[1]-5.5,2))/0.05)'
                                 '+ 0.8*exp(-(pow(x[0]-4.3,2) + pow(x[1]-5.5,2))/0.05)'
                                 '+ 0.8*exp(-(pow(x[0]-5.6,2) + pow(x[1]-5.5,2))/0.05)'
                                 '+ 0.8*exp(-(pow(x[0]-7.2,2) + pow(x[1]-5.5,2))/0.05)'
                                 '+ 0.9*exp(-(0.5*pow(x[0]-4, 2) + pow(x[1]-2.8,2))/0.1)'
                                 '+ 0.9*exp(-(0.5*pow(x[0]-3, 2) + pow(x[1]-4.4,2))/0.1)'
                                 '+ 0.9*exp(-(0.5*pow(x[0]-5, 2) + pow(x[1]-4.4,2))/0.1)'
                                 '+ 0.9*exp(-(0.5*pow(x[0]-6.2, 2) + pow(x[1]-3,2))/0.1)'
                                 '+ 0.8*exp(-(pow(x[0]-3.,2) + pow(x[1]-2.,2))/0.1)'), degree=2)
            
        self.ag_pos_0 = np.array([[2.5, 2.6],[3.5, 3.8]])
        self.ag_vel_0 = np.array([[0.03, -0.1],[0.2,-0.02]])

        self.ag_pos_0 = np.empty((0,2))

