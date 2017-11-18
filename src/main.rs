#[macro_use]
extern crate approx; // For the macro relative_eq!
extern crate nalgebra as na;
extern crate image;

use na::{Vector3, Rotation3, Point3};
use image::{GenericImage, ImageBuffer};
use std::fs::File;

type Color = [u8; 3];

fn vec3_to_rgb8(v : Vector3<f32>) -> Color {
    [v[0] as u8, v[1] as u8, v[2] as u8]
}

fn bg(ray : Vector3<f32>) -> Color {
    let unit_dir = ray.normalize();
    let t = 0.5*(unit_dir[1] + 1.0);
    let white = Vector3::new(1.0, 1.0, 1.0)*255.0;
    let bloop = Vector3::new(0.5, 0.7, 1.0)*255.0;
    vec3_to_rgb8( (1.0-t)*white + t*bloop )
}

struct RaylorSwift {
    origin : Point3<f32>,
    direction : Vector3<f32>
}

// translate a point, it's still a point!
fn point_at_param(r : RaylorSwift, t : f32) -> Point3<f32> {
    r.origin + t*r.direction
}

type Ray = RaylorSwift;

fn main() {
    // let axis  = Vector3::x_axis();
    // let angle = 1.57;
    // let b     = Rotation3::from_axis_angle(&axis, angle);

    // relative_eq!(b.axis().unwrap(), axis);
    // relative_eq!(b.angle(), angle);
    
    let nx = 100;
    let ny = 200;
    
    let lower_left_corner = Vector3::new(-2.0, -1.0, -1.0);
    let hor = Vector3::new(4.0, 0.0, 0.0);
    let vert = Vector3::new(0.0, 2.0, 0.0);

    
    let img = ImageBuffer::from_fn(nx, ny, |x, y| {
        let u = (x as f32 / nx as f32);
        let v = (y as f32 / ny as f32);
        let r = Ray{ origin: Point3::origin(), direction: lower_left_corner + u*hor + v*vert };
        image::Rgb(vec3_to_rgb8(r.direction))
    });
    
    let ref mut fout = File::create("test.png").unwrap();
    image::ImageRgb8(img).save(fout, image::PNG).unwrap();



}
