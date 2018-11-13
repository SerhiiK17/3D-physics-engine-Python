import sys, pygame, numpy, time
from math import cos, sin

pygame.init()


class Object3D:
    
    def __init__(self, path = '', screen = pygame.display.set_mode((700, 650)), 
                 color = [255,255,255], vertices = 0, faces_v = 0, velocity = [0, 0, 0], 
                 ang_velocity = [0, 0, 0], mass = 0, center_of_mass = 0, 
                 inertia_tensor = 0, position_tensor =[[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        self.path = path
        self.screen = screen
        self.color = color
        self.vertices = vertices
        self.faces_v = faces_v
        self.velocity = velocity
        self.ang_velocity = ang_velocity
        self.mass = mass
        self.center_of_mass = center_of_mass
        self.inertia_tensor = inertia_tensor
        self.position_tensor = position_tensor
    

    def set_parameters_from_obj_file(self):
        obj_size = 1000  # 1000000  # For zooming object
        
        str = open(self.path, 'r').read().split("\n")
        del str[-1]

        vertices_str_format = [line.split(' ')[1:] for line in str if line[0] == 'v']
        self.vertices = [[obj_size * float(j) for j in i] for i in vertices_str_format]
        for i in self.vertices:
            i[2] += 7000

        faces = [line.split(' ')[1:] for line in str if line[0] == 'f']
        self.faces_v = [[int(''.join(j.split('//')[:1])) for j in i] for i in faces]

        return self.vertices, self.faces_v

    
    def draw(self):
        x_center = 370
        y_center = 350
        scale_factor = 1 / 1000 # Denominator here is the distance from the camera point to 
        # the projection plane

        for i in self.faces_v:
            if ((self.vertices[i[1] - 1][0] - self.vertices[i[0] - 1][0]) 
                * (self.vertices[i[2] - 1][1] - self.vertices[i[0] - 1][1]) - (
                    self.vertices[i[1] - 1][1] - self.vertices[i[0] - 1][1]) * (
                        self.vertices[i[2] - 1][0] - self.vertices[i[0] - 1][0])) <= 0:
                color = self.color
            else:
                color = (70, 70, 70)

            list_of_face_points = []
            for j in i:
                if (scale_factor * self.vertices[j - 1][2]) <= 0:
                    k = 0.1
                else:
                    k = scale_factor * self.vertices[j - 1][2]

                list_of_face_points.append((x_center + self.vertices[j - 1][0] 
                                / k, y_center - self.vertices[j - 1][1] / k))

            pygame.draw.lines(self.screen, color, True, list_of_face_points)

        pygame.display.update()


    def remove(self):
        color = (0, 0, 0)
        x_center = 370
        y_center = 350
        scale_factor = 1 / 1000  # Denominator here is the distance from the camera point to 
        # the projection plane

        for i in self.faces_v:
            list_of_face_points = []
            for j in i:
                if (scale_factor * self.vertices[j - 1][2]) <= 0:
                    k = 0.1
                else:
                    k = scale_factor * self.vertices[j - 1][2]

                list_of_face_points.append((x_center + self.vertices[j - 1][0]
                                            / k, y_center - self.vertices[j - 1][1] / k))

            pygame.draw.lines(self.screen, color, True, list_of_face_points)

        pygame.display.update()
    
    
    def change_point_of_view(self):
        angle = - 1 / 100
        ang_v = self.ang_velocity
        
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_d]:
            for i in range(len(self.vertices)):
                self.vertices[i] = numpy.dot(self.rotation_matrix(angle)[1], self.vertices[i])
            
            self.ang_velocity = numpy.dot(self.rotation_matrix(angle)[1], ang_v)
            self.velocity = numpy.dot(self.rotation_matrix(angle)[1], self.velocity)
            self.inertia_tensor = numpy.dot(self.rotation_matrix(angle)[1], 
                numpy.dot(self.inertia_tensor, numpy.transpose(self.rotation_matrix(angle)[1])))
        
        elif pressed[pygame.K_a]:
            for i in range(len(self.vertices)):
                self.vertices[i] = numpy.dot(self.rotation_matrix( - angle)[1], self.vertices[i])
            
            self.ang_velocity = numpy.dot(self.rotation_matrix( - angle)[1], ang_v)
            self.velocity = numpy.dot(self.rotation_matrix( - angle)[1], self.velocity)
            self.inertia_tensor = numpy.dot(self.rotation_matrix( - angle)[1], 
                numpy.dot(self.inertia_tensor, numpy.transpose(self.rotation_matrix( - angle)[1])))
        
        elif pressed[pygame.K_w]:
            for i in range(len(self.vertices)):
                self.vertices[i] = numpy.dot(self.rotation_matrix( - angle)[0], self.vertices[i])
            
            self.ang_velocity = numpy.dot(self.rotation_matrix( - angle)[0], ang_v)
            self.velocity = numpy.dot(self.rotation_matrix( - angle)[0], self.velocity)
            self.inertia_tensor = numpy.dot(self.rotation_matrix( - angle)[0], 
                numpy.dot(self.inertia_tensor, numpy.transpose(self.rotation_matrix( - angle)[0])))
        
        elif pressed[pygame.K_s]:
            for i in range(len(self.vertices)):
                self.vertices[i] = numpy.dot(self.rotation_matrix(angle)[0], self.vertices[i])

            self.ang_velocity = numpy.dot(self.rotation_matrix(angle)[0], ang_v)
            self.velocity = numpy.dot(self.rotation_matrix(angle)[0], self.velocity)
            self.inertia_tensor = numpy.dot(self.rotation_matrix(angle)[0], 
                numpy.dot(self.inertia_tensor, numpy.transpose(self.rotation_matrix(angle)[0])))
        
        elif pressed[pygame.K_LEFT]:
            for i in self.vertices:
                i[0] += 37
        elif pressed[pygame.K_RIGHT]:
            for i in self.vertices:
                i[0] -= 37
        elif pressed[pygame.K_UP]:
            for i in self.vertices:
                i[2] -= 125
        elif pressed[pygame.K_DOWN]:
            for i in self.vertices:
                i[2] += 125
        
        #print(self.inertia_tensor)
        return self.vertices
    
    
    # This supposed to work for every normal polygon, but it doesnt
    def get_mass_and_cm_and_ti(self):
        mass = 0
        mass_tmp = 0
        center_mass = 0
        center_mass_tmp = 0
        inertia_tensor = numpy.zeros((3, 3))
        v = numpy.array(self.vertices)
        
        for i in self.faces_v:
            face_mass_tmp = 0
            face_center_mass_tmp = 0
            
            for j in range(0, len(i) - 1):
                v1 = numpy.array(v[i[j] - 1]) - numpy.array(v[i[0] - 1])
                v2 = numpy.array(v[i[j + 1] - 1]) - numpy.array(v[i[0] - 1])
                #print(v1, v2)
                face_mass_tmp += 1/2 * numpy.cross(v1, v2)
                face_center_mass_tmp += 1/len(i) * v[i[j] - 1]
            
            mass_tmp = 1/2 * numpy.dot(face_mass_tmp, face_center_mass_tmp)
            center_mass_tmp = 3/4 * mass_tmp * face_center_mass_tmp
            mass += mass_tmp
            center_mass += center_mass_tmp
            
            cm_of_pyramid = 3/4 * face_center_mass_tmp
            for j in range(3):
                for k in range(3):
                    if k == j:
                        ind = [number for number in (0, 1, 2) if number != j]
                        inertia_tensor[j, k] += mass_tmp * (numpy.dot(cm_of_pyramid[ind[0]],
                        cm_of_pyramid[ind[0]]) + numpy.dot(cm_of_pyramid[ind[1]],
                        cm_of_pyramid[ind[1]]))
                    else:
                        ind = [number for number in (0, 1, 2) if number == j or number == k]
                        inertia_tensor[j, k] -= mass_tmp * (numpy.dot(cm_of_pyramid[ind[0]],
                        cm_of_pyramid[ind[1]]))
        
        center_mass = 1/mass * center_mass
        self.mass = mass
        self.center_of_mass = center_mass
        self.inertia_tensor = inertia_tensor
        
        return self.mass, self.center_of_mass, self.inertia_tensor
    
    
    #Work just for the simple polygons like a cube
    def get_center_of_mass(self):
        v = []
        vert = []        
        for i in self.faces_v:
            for vertex in i:
                if vertex not in v:
                    v.append(vertex)        
        for i in v:
            vert.append(self.vertices[i - 1])
        
        self.center_of_mass = [sum([j[i] / len(vert) for j in vert]) 
                               for i in (0, 1, 2)]
        return self.center_of_mass
    
    #Work just for a cube
    def get_mass(self):
        v = self.vertices
        f = self.faces_v[0]
        cube_length_vector = numpy.array(v[f[1] - 1]) - numpy.array(v[f[0] - 1])
        cube_length = (numpy.dot(cube_length_vector, cube_length_vector)) ** 0.5
        self.mass = cube_length ** 3
        return self.mass

    # Work just for a cube
    def get_inertia_tensor_initial(self):
        cube_tensor_inertia = numpy.zeros((3,3))
        v = self.vertices
        f = self.faces_v[0]
        cube_length_vector = numpy.array(v[f[1] - 1]) - numpy.array(v[f[0] - 1])
        cube_length = (numpy.dot(cube_length_vector, cube_length_vector)) ** 0.5
        
        dimensions_number = len(cube_tensor_inertia)
        for i in range(dimensions_number):
            for j in range(dimensions_number):
                if i == j:
                    cube_tensor_inertia[i, j] = self.mass * 1/6 * cube_length ** 2
        
        self.inertia_tensor = cube_tensor_inertia * 5 # Number here is to slow down the rotation 
        # after collision
        return self.inertia_tensor
    
    
    def get_inertia_tensor_current(self):
        for i in range(3):
            tensor = numpy.array(self.inertia_tensor)
            rotation_matrix = numpy.array(self.position_tensor)
            self.inertia_tensor = numpy.dot(rotation_matrix, numpy.dot(tensor, 
                numpy.transpose(rotation_matrix)))
        
        return self.inertia_tensor
    
    @staticmethod
    def rotation_matrix(angle):
        rotation_matrix = [0, 0, 0]
        rotation_matrix[0] = [[1, 0, 0], [0, cos(angle), - sin(angle)],
                              [0, sin(angle), cos(angle)]]
        rotation_matrix[1] = [[cos(angle), 0, sin(angle)], [0, 1, 0],
                              [- sin(angle), 0, cos(angle)]]
        rotation_matrix[2] = [[cos(angle), - sin(angle), 0],
                              [sin(angle), cos(angle), 0], [0, 0, 1]]
        
        return rotation_matrix
    
    
    def move(self):
        dt = 0.1
        v = numpy.array(self.velocity)
        ang_v = numpy.array(self.ang_velocity)      
        c_m = numpy.array(self.get_center_of_mass())
        
        for i in range(len(self.vertices)):
            radius_vector = numpy.array(self.vertices[i]) - c_m
            for k in range(3):
                radius_vector = numpy.dot(self.rotation_matrix(ang_v[k])[k], radius_vector)
            
            self.vertices[i] = c_m + radius_vector + v * dt
        
        return self.vertices


def collision_wall(pol):
    
    b = 15000
    ang_v = pol.ang_velocity
    v = numpy.array(pol.velocity)
    
    for i in pol.vertices:
        ec = 1 # one means no energy is lose after collision
        radius_vector = numpy.array(i) - numpy.array(pol.get_center_of_mass())
        tang_v = numpy.cross(ang_v, radius_vector)
        v_in_collision_point = v + tang_v
        
        screen_width = pol.screen.get_width()
        screen_height = pol.screen.get_height()
        walls_normals = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        collisions_conditions = [i[0] <= - b, i[0] >= b, i[1] <= - b, i[1] >= b, 
                                 i[2] <= b/3, i[2] >= b]
        
        for j in range(len(walls_normals)):
            normal = walls_normals[j]
            if collisions_conditions[j] and numpy.dot(v_in_collision_point, normal) < 0:
                inertia_tensor = pol.get_inertia_tensor_current()
                impulse = ( - (1 + ec) * numpy.array(normal) * numpy.dot(v_in_collision_point, normal)) \
                          / (1 / pol.mass + numpy.dot(normal, (numpy.linalg.inv(inertia_tensor)
                          * (numpy.cross((numpy.cross(radius_vector, normal)), radius_vector)))))
                pol.velocity += impulse / pol.mass
                pol.ang_velocity += numpy.dot(numpy.linalg.inv(inertia_tensor),
                                    numpy.cross(radius_vector, impulse))
        
    return pol.velocity, pol.ang_velocity


def collision_detection(pol1, pol2):
    # Here need to be found first - collision point, second - normal, along which collision occurs
       
    # But before need to calculate vertices which are in faces of each polygons
    v1 = []
    v2 = []
    vert1 = []
    vert2 = []
    
    for i in pol1.faces_v:
        for vertex in i:
            if vertex not in v1:
                v1.append(vertex)    
    for i in pol2.faces_v:
        for vertex in i:
            if vertex not in v2:
                v2.append(vertex)
     
    for i in v1:
        vert1.append(pol1.vertices[i - 1])    
    for i in v2:
        vert2.append(pol2.vertices[i - 1])
    
    # The calculation of the collision point, which will be the remaining element of vert2, 
    # is following
    
    axises = []
    
    for i in pol1.faces_v:
        x = numpy.array(pol1.vertices[i[1] - 1]) - numpy.array(pol1.vertices[i[0] - 1])
        y = numpy.array(pol1.vertices[i[2] - 1]) - numpy.array(pol1.vertices[i[0] - 1])
        axis = numpy.cross(x, y) / (numpy.dot(x, x) * numpy.dot(y, y)) ** 0.5
        
        axises.append(axis)
        
        vert1_proj = []        
        for j in vert1:
            vert1_proj.append(numpy.dot(numpy.array(j), axis))
        
        k = 0
        vert1_proj.sort()
        while k <= len(vert2) - 1:
            proj = numpy.dot(numpy.array(vert2[k]), axis)
            if proj < vert1_proj[0] or proj > vert1_proj[len(vert1_proj) - 1]:
                vert2.pop(k)
            else:
                k += 1
        
        if len(vert2) == 0:
            break   
    
    if len(vert2) != 0:
        # The calculation of the normal vector of the collision is following
        vert1_proj = []
        for j in vert1:
            vert1_proj.append(numpy.dot(numpy.array(j), axises[0]))
        
        vert1_proj.sort()
        normal = axises[0]
        depth = numpy.dot(numpy.array(vert2[0]), axises[0]) - vert1_proj[len(vert1_proj) - 1]
        
        for k in range(1, len(axises)):
            vert1_proj = []
            for j in vert1:
                vert1_proj.append(numpy.dot(numpy.array(j), axises[k]))
            
            vert1_proj.sort()
            d = numpy.dot(numpy.array(vert2[0]), axises[k]) - vert1_proj[len(vert1_proj) - 1]
            if d < depth:
                depth = d
                normal = axises[k]
        
        normal /= (numpy.dot(normal, normal)) ** 0.5
        print(vert2)
        collision_response(pol2, pol1, vert2[0], normal)
    
    return pol1.velocity, pol2.velocity, pol1.ang_velocity, pol2.ang_velocity


def collision_response(pol1, pol2, collision_point, normal):
    ec = 1
    ang_v1 = pol1.ang_velocity
    ang_v2 = pol2.ang_velocity
    radius_vector_1 = numpy.array(collision_point) - numpy.array(pol1.get_center_of_mass())
    radius_vector_2 = numpy.array(collision_point) - numpy.array(pol2.get_center_of_mass())
    tang_v1 = numpy.cross(ang_v1, radius_vector_1)
    tang_v2 = numpy.cross(ang_v2, radius_vector_2)
    v1 = numpy.array(pol1.velocity)
    v2 = numpy.array(pol2.velocity)

    v1_in_collision_point = v1 + numpy.array(tang_v1)
    v2_in_collision_point = v2 + numpy.array(tang_v2)
    inertia_tensor_1 = pol1.get_inertia_tensor_current()
    inertia_tensor_2 = pol2.get_inertia_tensor_current()

    print(numpy.dot(v1_in_collision_point - v2_in_collision_point, normal))
    if numpy.dot(v1_in_collision_point - v2_in_collision_point, normal) > 0:
        impulse = (- (1 + ec) * numpy.array(normal) * numpy.dot(v1_in_collision_point
             - v2_in_collision_point, normal)) \
             / (1 / pol1.mass + 1 / pol2.mass + numpy.dot(normal, (numpy.linalg.inv(inertia_tensor_1)
             * (numpy.cross(numpy.cross(radius_vector_1, normal), radius_vector_1)) +
             numpy.linalg.inv(inertia_tensor_2) * (numpy.cross(numpy.cross(radius_vector_2, normal),
                                                                               radius_vector_2)))))
        pol1.velocity += impulse / pol1.mass
        pol1.ang_velocity += numpy.dot(numpy.linalg.inv(inertia_tensor_1), 
                                    numpy.cross(radius_vector_1, impulse))
        pol2.velocity -= impulse / pol2.mass
        pol2.ang_velocity -= numpy.dot(numpy.linalg.inv(inertia_tensor_2), 
                                    numpy.cross(radius_vector_2, impulse))

    '''energy = pol1.mass / 2 * numpy.dot(pol1.velocity, pol1.velocity) \
             + numpy.dot(pol1.ang_velocity, pol1.ang_velocity) * numpy.linalg.inv(inertia_tensor_1) / 2 \
             + pol2.mass / 2 * numpy.dot(pol2.velocity, pol2.velocity) \
             + numpy.dot(pol2.ang_velocity, pol2.ang_velocity) * numpy.linalg.inv(inertia_tensor_2) / 2
    print(energy)'''

    return pol1.velocity, pol1.ang_velocity, pol2.velocity, pol2.ang_velocity


def main():
    win = pygame.display.set_mode((700, 650))
    x_center = 370
    y_center = 350
    
    cube = Object3D('C:\\obj.obj', win, [255, 255, 255], 0, 0, [0, 10, 10], [0, 0, 0])
    cube.set_parameters_from_obj_file()
    cube2 = Object3D('D:\\cube3.obj', win, [255, 255, 255], 0, 0, [0, 0, 100], [1/200, 0, 0])
    cube2.set_parameters_from_obj_file()
    m, cm, ti = cube.get_mass_and_cm_and_ti()
    print(m, cm, ti)
    
    obj_list = [cube, cube2]
    for obj in obj_list:
        obj.get_center_of_mass()
        obj.get_mass()
        obj.get_inertia_tensor_initial()
    
    working = True
    while working:
        pressed = pygame.key.get_pressed()
        win.fill((0, 0, 0))
        
        for obj in obj_list:
            obj.change_point_of_view()
        
        # these walls cant rotate with camera movement
        '''for obj in obj_list:
            collision_wall(obj)'''

        for i in range(len(obj_list)):
            for j in range(i + 1, len(obj_list)):
                collision_detection(obj_list[i], obj_list[j])
                collision_detection(obj_list[j], obj_list[i])
        
        for obj in obj_list:
            obj.move()
        
        for obj in obj_list:
            obj.draw()
        
        pygame.display.update()
        time.sleep(1/23)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                working = False
                pygame.display.quit()



main()

###___
###___
###___


