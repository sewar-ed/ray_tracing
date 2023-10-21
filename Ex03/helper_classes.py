import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    # TODO:
    v = normalize(vector - 2 * np.dot(vector, axis) * axis)
    return v


"""
Parameters:
- intersection: a tuple consisting of the intersected object and its distance t from the origin
- ray: the ray that intersects the object/point
"""
def get_intersection_point(intersection, ray):
    _ ,t = intersection
    return ray.origin + t * ray.direction
        

## Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = direction

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection_point):
        return Ray(intersection_point, self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.inf
    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection):
        return Ray(intersection,normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = position
        self.direction = normalize(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        #TODO
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        #TODO
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        #TODO
        d = self.get_distance_from_light(intersection)
        att = (self.kc + self.kl * d + self.kq * (d**2))
        v = normalize(self.position - intersection)
        return (self.intensity * np.dot(self.direction, v)) / att


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = normalize(direction)

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        intersections = None
        nearest_object = None
        min_distance = np.inf

        for obj in objects:
            intersection = obj.intersect(self)
            t, obj = intersection if intersection else (np.inf, None)
            if t < min_distance:
                min_distance = t
                nearest_object = obj

        return nearest_object, min_distance


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = (np.dot(v, self.normal) / np.dot(self.normal, ray.direction))
        if t > 0:
            return t, self
        else:
            return None


class Rectangle(Object3D):
    """
        A rectangle is defined by a list of vertices as follows:
        a _ _ _ _ _ _ _ _ d
         |               |  
         |               |  
         |_ _ _ _ _ _ _ _|
        b                 c
        This function gets the vertices and creates a rectangle object
    """
    def __init__(self, a, b, c, d):
        """
            ul -> bl -> br -> ur
        """
        self.abcd = [np.asarray(v) for v in [a, b, c, d]]
        self.normal = self.compute_normal()

    def compute_normal(self):
        # TODO
        a, b, c, d = self.abcd
        n = np.cross(d - c, b - c)
        return normalize(n)

    # Intersect returns both distance and nearest object.
    # Keep track of both.
    def intersect(self, ray: Ray):
        #TODO
        rect_plane = Plane(self.normal, self.abcd[0])
        intersection = rect_plane.intersect(ray)
        if not intersection: return None
        t, _ = intersection
        p = ray.origin + t * ray.direction # refactor
        for i in range(len(self.abcd)):
            v = self.abcd[i - 1] - p
            u = self.abcd[i] - p
            if np.dot(self.normal, np.cross(v, u)) <= 0:
                return None
            
        return t, self


class Cuboid(Object3D):
    def __init__(self, a, b, c, d, e, f):
        """ 
              g+---------+f
              /|        /|
             / |  E C  / |
           a+--|------+d |
            |Dh+------|B +e
            | /  A    | /
            |/     F  |/
           b+--------+/c
        """

        h = (np.asarray(b) + (np.asarray(e) - np.asarray(c)))
        g = (np.asarray(a) + (np.asarray(f) - np.asarray(d)))

        A = Rectangle(a,b,c,d)
        B = Rectangle(d,c,e,f)
        C = Rectangle(f,e,h,g)
        D = Rectangle(g,h,b,a)
        E = Rectangle(g,a,d,f)
        F = Rectangle(e,c,b,h)
        self.face_list = [A,B,C,D,E,F]

    def apply_materials_to_faces(self):
        for t in self.face_list:
            t.set_material(self.ambient,self.diffuse,self.specular,self.shininess,self.reflection)

    # Hint: Intersect returns both distance and nearest object.
    # Keep track of both
    def intersect(self, ray: Ray):
        #TODO
        ray_int = []
        for f in self.face_list:
            res_tuple = f.intersect(ray)
            if res_tuple: 
                ray_int.append(res_tuple)
        #return the closest intersection and the corresponding plane
        if ray_int != []:
            result = min(ray_int, key=lambda x: x[0])
            return result[0],result[1]
        return None

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def compute_normal(self, point):
        self.normal = normalize(point - np.asarray(self.center))

    def intersect(self, ray: Ray):
        #TODO
        a = np.sum(np.square(ray.direction))
        b = 2 * np.sum((ray.origin - self.center) * ray.direction)
        c = np.sum(np.square(ray.origin - self.center)) - (self.radius)**2
        #check for intersection using quadratic formula
        discriminant= b**2 - 4*a*c
        if discriminant < 0:
            return None
        elif discriminant == 0:
            t = -b / 2*a
            if t < 0: return None
            p = ray.origin + t * ray.direction
            self.compute_normal(p)
            return t,self
        else:
            #we want to return the closest intersection point 
            t_1 = (-b + np.sqrt(discriminant))/ 2*a 
            p_1 = ray.origin + t_1 * ray.direction
            t_2 = (-b - np.sqrt(discriminant))/ 2*a
            p_2 = ray.origin + t_2 * ray.direction 

            if t_1 < 0 and t_2 < 0: return None

            elif t_1 < 0 and t_2 >= 0:
                self.compute_normal(p_2)
                return t_2, self
            
            elif t_1 >= 0 and t_2 < 0:
                self.compute_normal(p_1)
                return t_1, self
            #use distance from origin to determine which is closer
            if t_1 > t_2:
                self.compute_normal(p_2)
                return t_2,self
            
            self.compute_normal(p_1)
            return t_1,self