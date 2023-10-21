from helper_classes import *
import matplotlib.pyplot as plt

"""
Parameters:
- objects: list of objects in the scene
- ray: Ray object

returns: a tuple consisting of the intersected object and its distance from ray.origin 
"""
def find_intersection(objects, ray: Ray):
    return ray.nearest_intersected_object(objects)

"""
Parameters:
- ambient: ambient light
- obj: the object on which the point lies

returns: np.ndarray of RGB values for the ambient color of the point 
"""
def calc_ambient(obj: Object3D, ambient, coef=1):
    return obj.ambient * (coef * ambient)

"""
Parameters:
- lights: list of all light sources in the scence
- intersection: a tuple of the intersected object and its distance from ray.origin
- data: a dictionary that maps between a light source and relevant PointToLightData object

returns: diffusion color as an np.ndarray
"""

def calc_diffusion(intensity, obj, light_ray, coef=1):

    diffusion = np.array([0,0,0], dtype=np.float64)
    
    
    diffusion = coef * (obj.diffuse * intensity * np.dot(obj.normal, normalize(light_ray.direction)))
                    
        # TODO: add support for other 3DObjects

    return diffusion

"""
Parameters:
- lights: list of all light sources in the scence
- intersection: a tuple of the intersected object and its distance from ray.origin
- ray: the ray shot towards the point (used for computing the reflection vector)
- data: a dictionary that maps between a light source and relevant PointToLightData object

returns: specular color as an np.ndarray
"""

def calc_specular(intensity, obj, ray: Ray, light_ray, coef=1):

    specular = np.array([0,0,0], dtype=np.float64)
    
    refl_vector = reflected(light_ray.direction, obj.normal)
    specular = coef * (obj.specular * intensity * (np.dot(ray.direction, refl_vector)**(obj.shininess)))
            # TODO: add support for other 3DObjects

    return specular

"""
TODO: Impelement

returns: reflection ray
"""

def get_reflective_ray(point, obj, ray):
    v = reflected(ray.direction, obj.normal)
    return Ray(point, v)

"""
Parameters:
- objects: objects in the scence
- ray: ray cast towards the point
- intersection: a tuple consisting of the object and the distance from ray.origin
- ambient: ambient light
- max_depth: maximum recursion depth
- level: current recusion depth
"""

def get_color(objects, ray, intersection, lights, ambient, max_depth, epsilon=0.045, level=1, coef=1):
    obj, _ = intersection
    if not obj: return np.array([0,0,0])
    point = get_intersection_point(intersection, ray) + epsilon * obj.normal
    #data = {light:PointToLightData(light, point, obj.normal, objects) for light in lights}
    ambient_refl = calc_ambient(obj, ambient)
    diffuse_refl = np.array([0,0,0], dtype=np.float64)
    specular_refl =  np.array([0,0,0], dtype=np.float64)

    for light in lights:
        light_ray = light.get_light_ray(point)
        blocking_obj, t = light_ray.nearest_intersected_object(objects)
        distance = light.get_distance_from_light(point)
        intensity = light.get_intensity(point)
        
        if not (blocking_obj and t < distance):
            diffuse_refl += calc_diffusion(intensity, obj, light_ray)
            specular_refl += calc_specular(intensity, obj, ray, light_ray)

    color = ambient_refl + diffuse_refl + specular_refl
    refl_ray = get_reflective_ray(point, obj, ray)
    refl_ray_intersection = find_intersection(objects, refl_ray)
    
    # TODO: Implement Recursion
    if (level >= max_depth):
        return color


    return color + coef * obj.reflection * get_color(objects, refl_ray, refl_ray_intersection, lights, ambient, max_depth, level=level + 1)

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            color = np.zeros(3)

            # This is the main loop where each pixel color is computed.
            # TODO
            intersection = find_intersection(objects, ray)
            color = get_color(objects, ray, intersection, lights, ambient,max_depth)

            
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image


# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    light = PointLight(intensity= np.array([1, 1, 1]),position=np.array([1,1.5,1]),kc=0.1,kl=0.1,kq=0.1)
    lights = [light,DirectionalLight(intensity= np.array([1, 1, 1]),direction=np.array([1,1,1]))]
    bear_body=Sphere([-0.5, 0.2, -1],0.5)
    bear_body.set_material([1, 0, 0], [1, 0, 0], [0.3, 0.3, 0.3], 100, 1)
    bear_head=Sphere([-0.5, 0.9, -1],0.4)
    bear_head.set_material([1, 0, 0], [1, 0, 0], [0.3, 0.3, 0.3], 100, 1)

    r_ear=Sphere( [-0.5-0.35, 0.9 + 0.4, -1],0.1)
    r_ear.set_material([1, 0, 0], [1, 0, 0], [0.3, 0.3, 0.3], 100, 1)
    l_ear=Sphere(  [-0.5+0.35, 0.9 + 0.4, -1],0.1)
    l_ear.set_material([1, 0, 0], [1, 0, 0], [0.3, 0.3, 0.3], 100, 1)

    r_hand=Sphere([-0.5 - 0.4, 0.2, -1],0.2)
    r_hand.set_material([1, 0, 0], [1, 0, 0], [0.3, 0.3, 0.3], 100, 1)
    l_hand=Sphere(  [-0.5 + 0.4, 0.2, -1],0.2)
    l_hand.set_material([1, 0, 0], [1, 0, 0], [0.3, 0.3, 0.3], 100, 1)

    plane = Plane([0,1,0], [0,-0.3,0])
    plane.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [1, 1, 1], 10, 0.5)
    background = Plane([0,0,1], [0,0,-3])
    background.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], 10, 0.5)
    objects = [bear_body,bear_head,plane,background,r_ear,l_ear,r_hand,l_hand]

    return camera,lights, objects