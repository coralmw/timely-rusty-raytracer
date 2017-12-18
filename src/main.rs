extern crate approx; // For the macro relative_eq!
extern crate nalgebra as na;
extern crate image;
extern crate rand;
extern crate timely;
#[macro_use] extern crate itertools;

use na::{Vector3, Point3};
use image::{ImageBuffer};
use std::fs::File;
use std::collections::LinkedList;
use std::f32;

use std::iter;

use rand::{random, Open01};

use timely::dataflow::{InputHandle};
use timely::dataflow::operators::{Map, Accumulate, Inspect, Probe, Input};

type Color = [u8; 3];

#[derive(Copy, Clone)]
struct screenspec {
    lower_left_corner : Vector3<f32>,
    hor : Vector3<f32>,
    vert : Vector3<f32>,
    nx : u32,
    ny : u32,

}

fn vec3_to_rgb8(v : Vector3<f32>) -> Color {
    [v[0] as u8, v[1] as u8, v[2] as u8]
}

fn random_unit() -> Vector3<f32> {
    let mut v : Vector3<f32>;
    while {
        v = Vector3::new( random::<f32>(), 
                          random::<f32>(), 
                          random::<f32>() );
        v = (v * 2.0) - Vector3::new(1.0, 1.0, 1.0);
        na::dot(&v, &v) >= 1.0 // condition, passed back to while as test
    } {} // do-nothing body, emulates do-while
    v
}

const RED : Color = [255, 0, 0];

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
    // Rect(Rect),
    // Ngon(Ngon), // Hybrid modes!
    // RichardSpencer() // is a Hittable
}

impl Hit for Sphere {
    fn hit(&self, r : &Ray, t_min : f32, t_max : f32) -> Option<HitRecord> {
        let oc = r.origin - self.center;
        let a = na::dot(&r.direction, &r.direction);
        let b = na::dot(&oc, &r.direction);
        let c = na::dot(&oc, &oc) - self.radius*self.radius;
        let discrim = b*b - a*c; // factor 2*2 absorbed
        if discrim > 0.0 {
            let temp = (-b - (b*b -a*c).sqrt()) / a;
            if temp < t_max && temp > t_min {
                Some( HitRecord{ t:temp, 
                                 p:point_at_param(&r, temp), 
                                 normal:(point_to_vec(&point_at_param(&r, temp)) - point_to_vec(&self.center)) / self.radius
                              })
            } else {
                None
            }
        } else {
            None
        }
    }
}


fn hit_world(r : &Ray, t_min : f32, t_max : f32, world : &LinkedList<Hittable>) -> Option<HitRecord> {
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
    
    let mut Hittables = LinkedList::<Hittable>::new();
    Hittables.push_back( Hittable::Sphere(Sphere { center:Point3::new(0.0,0.0,-1.0), radius:0.5 }) );
    Hittables.push_back( Hittable::Sphere(Sphere { center:Point3::new(1.0,100.5,-1.0), radius:100.0 }) );

    // let s = Sphere { center:Point3::new(0.0,0.0,-1.0), radius:0.5 };
    match hit_world(&ray, 0.0, f32::MAX, &Hittables) {
        Some(hitrec) => {
            let target = hitrec.p + hitrec.normal + random_unit();
            let pt = point_to_vec(&point_at_param(ray, hitrec.t));
            let normal = (pt + Vector3::z()).normalize();
            get_pixel_color( &Ray{origin:hitrec.p, direction:target-hitrec.p} )
        }
        None => bg(&ray)
    }
}

fn shuffle((x, y) : (f32, f32)) -> (f32, f32) { 
    let Open01(vx) = random::<Open01<f32>>(); 
    let Open01(vy) = random::<Open01<f32>>(); 
    ((x+vx), (y+vy)) 
}

fn blend(samples : Vec<Color>) -> Color {
    let l = samples.len() as f32;
    let summed = samples.into_iter().fold([0.0, 0.0, 0.0], |slist, c| {[slist[0]+(c[0] as f32), slist[1]+(c[1] as f32), slist[2]+(c[2] as f32)]});
    [(summed[0]/l) as u8, (summed[1]/l) as u8, (summed[2]/l) as u8]
}

fn get_screenspace_aa_pixel_color(point : (f32, f32), screen : &screenspec) -> Color {
    let points : Vec<(f32, f32)> = iter::repeat(point)
                                        .take(50)
                                        .map(|pt| {shuffle(pt)}) // move the samples about for AA
                                        .map(|(x, y)| {(x/screen.nx as f32, y/screen.ny as f32)}) // scale to screen space
                                        .collect();
    let samples : Vec<Color> = points.into_iter()
                                     .map(|(u, v)| { get_pixel_color(&Ray{ origin: Point3::origin(), direction: screen.lower_left_corner + u*screen.hor + v*screen.vert })})
                                     .collect();
    blend(samples)

}


fn main() {    
    
    let lower_left_corner = Vector3::new(-2.0, -1.0, -1.0);
    let hor = Vector3::new(4.0, 0.0, 0.0);
    let vert = Vector3::new(0.0, 2.0, 0.0);
    
    let screen = screenspec{ lower_left_corner : Vector3::new(-2.0, -1.0, -1.0),
                             hor : Vector3::new(4.0, 0.0, 0.0),
                             vert : Vector3::new(0.0, 2.0, 0.0),
                             nx : 200,
                             ny : 100, };
    
    
    timely::execute_from_args(std::env::args(), move |worker| {
        let mut input = InputHandle::new();
        let pixlocs = iproduct!(0..screen.nx, 0..screen.ny);
        
        let probe = worker.dataflow(|scope| {
        
            scope.input_from(&mut input)
                 .map(move |(x, y)| { 
                    ((x, y), get_screenspace_aa_pixel_color((x as f32, y as f32), &screen)) 
                 })
                 .accumulate(vec![vec![[0u8, 0u8, 0u8]; screen.ny as usize]; screen.nx as usize], 
                             |pixels, data| { 
                                 println!("accumulating!");
                                 for &((x, y), color) in data.iter() {pixels[x as usize][y as usize] = color;} 
                             })
                 .inspect(move |imgbuf| {
                      let img = ImageBuffer::from_fn(screen.nx, screen.ny, |x, y| {
                          image::Rgb(imgbuf[x as usize][y as usize])
                      });
                      
                      let ref mut fout = File::create("test.png").unwrap();
                      image::ImageRgb8(img).save(fout, image::PNG).unwrap();
                 })
                 .probe();
        });
        
        for pixloc in pixlocs {
            if worker.index() == 0 {
                input.send(pixloc);
            }
        }
        input.advance_to(1);

    }).unwrap();
        
}
