#[macro_use]
extern crate approx; // For the macro relative_eq!
extern crate nalgebra as na;
extern crate image;


use na::{Vector3, Rotation3, Point3};
use image::{GenericImage, ImageBuffer};
use std::fs::File;
use std::collections::LinkedList;

type Color = [u8; 3];

fn vec3_to_rgb8(v : Vector3<f32>) -> Color {
    [v[0] as u8, v[1] as u8, v[2] as u8]
}

const red : Color = [255, 0, 0];

struct RaylorSwift {
    origin : Point3<f32>,
    direction : Vector3<f32>
}

// translate a point, it's still a point!
fn point_at_param(r : &RaylorSwift, t : f32) -> Point3<f32> {
    r.origin + t*r.direction
}

fn point_to_vec(pt : &Point3<f32>) -> Vector3<f32> {
    Vector3::new( pt[0], pt[1],pt[2] )
}

type Ray = RaylorSwift;

fn bg(ray : &Ray) -> Color {
    let unit_dir = ray.direction.normalize();
    let t = 0.5*(unit_dir[1] + 1.0);
    let white = Vector3::new(1.0, 1.0, 1.0)*255.0;
    let bloop = Vector3::new(0.5, 0.7, 1.0)*255.0;
    vec3_to_rgb8( (1.0-t)*white + t*bloop )
}


struct HitRecord {
    t : f32,
    p : Point3<f32>,
    normal : Vector3<f32>
}

struct Sphere {
    center : Point3<f32>,
    radius : f32
}

trait Hit {
    fn hit(&self, r : &Ray, t_min : f32, f_max : f32) -> Option<HitRecord>;
}

enum Hittable {
    Sphere(Sphere),
}


impl Hit for Sphere {
    fn hit(&self, r : &Ray, t_min : f32, t_max : f32) -> Option<HitRecord> {
        let oc = r.origin - self.center;
        let a = na::norm(&r.direction).powi(2);
        let b = 2.0 * na::dot(&oc, &r.direction);
        let c = na::norm(&oc).powi(2) - self.radius*self.radius;
        let discrim = b*b - 4.0*a*c; // factor 2*2 absorbed
        if discrim > 0.0 {
            let temp = (-b - (b*b -a*c).sqrt()) / a;
            if temp < t_max && temp > t_min {
                Some( HitRecord{ t:temp, 
                                 p:point_at_param(&r, temp), 
                                 normal:(point_to_vec(&point_at_param(&r, temp)) - point_to_vec(&self.center)) / self.radius
                              })
            } else {
                let temp = (-b + (b*b -a*c).sqrt()) / a; // sign flip
                Some( HitRecord{ t:temp, 
                                 p:point_at_param(&r, temp), 
                                 normal:(point_to_vec(&point_at_param(&r, temp)) - point_to_vec(&self.center)) / self.radius
                              })
            }
        } else {
            None
        }
    }
}


fn hit_world(r : &Ray, t_min : f32, t_max : f32, world : &LinkedList::<Hittable>) -> Option<HitRecord> {
    let mut closest_so_far = t_max;
    let mut nearest = None;
    for hittable in world {
        match hittable {
            &Hittable::Sphere(ref s) =>  match s.hit(&r, t_min, closest_so_far) {
                                            Some(hitrec) => {closest_so_far = hitrec.t;
                                                             nearest = Some(hitrec);
                                                            }
                                            None => {}
                    }

        }
    }
    nearest
}

fn get_pixel_color(ray : &Ray) -> Color {
    
    let mut hittables = LinkedList::<Hittable>::new();
    hittables.push_back( Hittable::Sphere(Sphere { center:Point3::new(0.0,0.0,-1.0), radius:0.5 }) );
    hittables.push_back( Hittable::Sphere(Sphere { center:Point3::new(1.0,1.0,-2.5), radius:1.0 }) );

    // let s = Sphere { center:Point3::new(0.0,0.0,-1.0), radius:0.5 };
    match hit_world(&ray, 0.0, 100.0, &hittables) {
        Some(hitrec) => {
            let pt = point_to_vec(&point_at_param(ray, hitrec.t));
            let N = (pt + Vector3::z()).normalize();
            vec3_to_rgb8( (N+Vector3::from_element(1.0))*128.0 )
        }
        None => bg(&ray)
    }
}

fn main() {    
    let nx = 200;
    let ny = 100;
    
    let lower_left_corner = Vector3::new(-2.0, -1.0, -1.0);
    let hor = Vector3::new(4.0, 0.0, 0.0);
    let vert = Vector3::new(0.0, 2.0, 0.0);

    
    let img = ImageBuffer::from_fn(nx, ny, |x, y| {
        let u = x as f32 / nx as f32;
        let v = y as f32 / ny as f32;
        let r = Ray{ origin: Point3::origin(), direction: lower_left_corner + u*hor + v*vert };
        image::Rgb( get_pixel_color(&r) )
    });
    
    let ref mut fout = File::create("test.png").unwrap();
    image::ImageRgb8(img).save(fout, image::PNG).unwrap();
}
