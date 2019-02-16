use cgmath::{
    perspective, vec3, vec4, Angle, Deg, InnerSpace, Matrix4, Point3, Transform, Vector3,
};
use env_logger;
use image;
use image::ImageBuffer;
use log::debug;

type Vec3 = Vector3<f32>;
type Mat4 = Matrix4<f32>;

const NEAR_PLANE: f32 = 0.5f32;
const FAR_PLANE: f32 = 10_000_000_000.0f32;

fn compact_color(input: [f32; 4]) -> u32 {
    let r: u32 = (input[0].max(0.0).min(1.0) * 255.0) as u32;
    let g: u32 = (input[1].max(0.0).min(1.0) * 255.0) as u32;
    let b: u32 = (input[2].max(0.0).min(1.0) * 255.0) as u32;
    let a: u32 = (input[3].max(0.0).min(1.0) * 255.0) as u32;
    let r = r << 24;
    let g = g << 16;
    let b = b << 8;

    r + g + b + a
}

struct Ray {
    origin: Vec3,
    direction: Vec3,
}

struct Light {
    position: Vec3,
}

struct Geometry {
    vertices: Vec<Vec3>,
    indices: Vec<i32>,
    color: [f32; 4],
}

struct Face {
    vertices: [Vec3; 3],
    normal: Vec3,
    start_index: usize,
}

struct FaceIter<'a> {
    geometry: &'a Geometry,
    face_idx: u32,
}

impl Iterator for FaceIter<'_> {
    type Item = Face;

    fn next(&mut self) -> Option<Face> {
        fn calc_normal(vertices: &[Vec3; 3]) -> Vec3 {
            let u = vertices[1] - vertices[0];
            let v = vertices[2] - vertices[0];

            vec3(
                (u.y * v.z) - (u.z * v.y),
                (u.z * v.x) - (u.x * v.z),
                (u.x * v.y) - (u.y * v.x),
            )
        }

        self.face_idx += 1;

        let geometry = self.geometry;

        let indices_idx = (self.face_idx * 3) as usize;
        if indices_idx < geometry.indices.len() - 2 {
            let index0 = geometry.indices[indices_idx];
            let index1 = geometry.indices[indices_idx + 1];
            let index2 = geometry.indices[indices_idx + 2];
            let v0 = geometry.vertices[index0 as usize];
            let v1 = geometry.vertices[index1 as usize];
            let v2 = geometry.vertices[index2 as usize];
            let vertices = [v0, v1, v2];
            let normal = calc_normal(&vertices);
            Some(Face {
                vertices,
                normal,
                start_index: indices_idx,
            })
        } else {
            None
        }
    }
}

impl Geometry {
    fn face_iter(&self) -> FaceIter {
        FaceIter {
            geometry: self,
            face_idx: 0,
        }
    }
}

struct Camera {
    view: Mat4,
    projection: Mat4,
    ortho_width: f32,
    ortho_height: f32,
}

struct Scene {
    camera: Camera,
    geometries: Vec<Geometry>,
    lights: Vec<Light>,
}

struct RenderSettings {
    resolution: (u32, u32),
    background_color: u32,
}

fn create_cube() -> Geometry {
    let vertices = vec![
        vec3(-1.0, -1.0, -1.0),
        vec3(-1.0, -1.0, 1.0),
        vec3(1.0, -1.0, 1.0),
        vec3(1.0, -1.0, -1.0),
        vec3(-1.0, 1.0, -1.0),
        vec3(-1.0, 1.0, 1.0),
        vec3(1.0, 1.0, 1.0),
        vec3(1.0, 1.0, -1.0),
    ];

    let indices = vec![
        0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 5, 4, 1, 4, 0, 2, 6, 5, 2, 5, 1,
        2, 6, 7, 2, 7, 3,
    ];

    Geometry {
        vertices,
        indices,
        color: [1.0, 0.0, 0.0, 1.0],
    }
}

fn create_scene() -> Scene {
    let view = Mat4::look_at(
        Point3::new(0.0f32, 0.0f32, -5.0f32),
        Point3::new(0.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
    );
    let projection = perspective(Deg(75.0), 1.0, NEAR_PLANE, FAR_PLANE);

    Scene {
        camera: Camera {
            view,
            projection,
            ortho_width: 5.0f32,
            ortho_height: 5.0f32,
        },
        geometries: vec![create_cube()],
        lights: vec![Light {
            position: vec3(1.0, 1.0, -1.2),
        }],
    }
}

fn intersect(ray: &Ray, face: &Face, view: &Mat4) -> Option<(f32, Vec3)> {
    const TAG: &str = "intersection";

    let Ray { origin, direction } = ray;
    let Face {
        vertices, /*normal,*/
        ..
    } = face;
    let (v0, v1, v2) = (vertices[0], vertices[1], vertices[2]);
    let (v0, v1, v2) = (v0.extend(1.0), v1.extend(1.0), v2.extend(1.0));
    let (v0, v1, v2) = (view * v0, view * v1, view * v2);
    let (v0, v1, v2) = (v0.xyz(), v1.xyz(), v2.xyz());

    debug!(
        target: TAG,
        "Intersect test:\nRay origin={:?} & direction={:?}\nv0={:?}\nv1={:?}\nv2={:?}",
        origin,
        direction,
        v0,
        v1,
        v2
    );

    let v0_v1 = v1 - v0;
    let v0_v2 = v2 - v0;
    let normal = v0_v1.cross(v0_v2);

    let normal_dot_ray_dir = normal.dot(*direction);
    if normal_dot_ray_dir.abs() < core::f32::EPSILON {
        debug!(target: TAG, "1 parallel");
        return None;
    }

    let d = normal.dot(v0);

    let t = (normal.dot(*origin) + d) / normal_dot_ray_dir;
    if t < 0.0 {
        debug!(target: TAG, "2 triangle behind t={}", t);
        return None;
    }

    let p = origin + t * direction;

    let edge0 = v0_v1;
    let vp0 = p - v0;
    let c = edge0.cross(vp0);
    if normal.dot(c) < 0.0 {
        debug!(target: TAG, "3 P on right of edge0");
        return None;
    }

    let edge1 = v2 - v1;
    let vp1 = p - v1;
    let c = edge1.cross(vp1);
    if normal.dot(c) < 0.0 {
        debug!(target: TAG, "4 P on right of edge1");
        return None;
    }

    let edge2 = v0 - v2;
    let vp2 = p - v2;
    let c = edge2.cross(vp2);
    if normal.dot(c) < 0.0 {
        debug!(target: TAG, "5 P on right of edge2");
        return None;
    }

    Some((t, p))
}

fn trace(scene: &Scene, ray: &Ray, background_color: u32) -> u32 {
    let mut hit_dist = FAR_PLANE + 1.0f32;
    let mut hit_point = vec3(0.0f32, 0.0f32, 0.0f32);
    let mut hit_geom_idx = 0;
    let mut hit_start_idx = 0;
    let mut face_normal = vec3(0.0f32, 0.0f32, 0.0f32);

    for (geom_idx, geom) in scene.geometries.iter().enumerate() {
        for face in geom.face_iter() {
            if let Some((dist, p)) = intersect(&ray, &face, &scene.camera.view) {
                if dist < hit_dist {
                    hit_dist = dist;
                    hit_point = p;
                    hit_geom_idx = geom_idx;
                    hit_start_idx = face.start_index;
                    face_normal = face.normal;
                }
            }
        }
    }

    if hit_dist < FAR_PLANE {
        debug!(target: "hitidx", "Hit vertex index: {} raydir={:?}", hit_start_idx, ray.direction);
        (&scene.lights)
            .iter()
            .map(|light| {
                let light_pos = scene.camera.view * light.position.extend(1.0);
                let light_pos = light_pos.xyz();
                let hit_to_light = hit_point - light_pos;
                let hit_to_light = hit_to_light.normalize();
                let angle = hit_to_light.angle(face_normal);
                let shade = angle.cos();
                debug!(target: "shade", "shade={} P={:?} angle={:?} light={:?} raydir={:?}", shade, hit_point, angle, light_pos, ray.direction);

                let diffuse_color = scene.geometries[hit_geom_idx].color;

                let final_color = [
                    diffuse_color[0] * shade,
                    diffuse_color[1] * shade,
                    diffuse_color[2] * shade,
                    diffuse_color[3],
                ];

                debug!(target: "shade", "color={}", final_color[0]);
                debug!(target: "shade", "color={}", final_color[1]);
                debug!(target: "shade", "color={}", final_color[2]);
                debug!(target: "shade", "color={}", final_color[3]);

                let final_color = compact_color(final_color);

                debug!(target: "shade", "color={}", final_color);

                final_color
            })
            .sum()
    } else {
        background_color
    }
}

fn render(scene: &Scene, render_settings: &RenderSettings) {
    let (res_w, res_h) = render_settings.resolution;
    let ortho_width = scene.camera.ortho_width;
    let ortho_height = scene.camera.ortho_height;
    let imgbuf = ImageBuffer::from_fn(res_w, res_h, |x, y| {
        let x = x as f32;
        let y = y as f32;
        let res_w = res_w as f32;
        let res_h = res_h as f32;
        let ortho_x = -(x / res_w) * ortho_width + (ortho_width / 2.0);
        let ortho_y = (y / res_h) * ortho_height - (ortho_height / 2.0);
        let ray_origin = vec3(0.0, 0.0, 0.0f32);
        let ray_direction = vec4(
            ortho_x / ortho_width,
            ortho_y / ortho_height,
            0.0f32,
            1.0f32,
        );
        let inverse_projection = scene.camera.projection.inverse_transform().unwrap();
        let ray_direction = inverse_projection * ray_direction;
        debug!(target: "ray", "ray: {:?}", ray_direction);

        let ray = Ray {
            origin: ray_origin,
            direction: ray_direction.xyz(),
        };
        let color = trace(&scene, &ray, render_settings.background_color);
        let red = (color & 0xFF_00_00_00) >> 24;
        let green = (color & 0x00_FF_00_00) >> 16;
        let blue = (color & 0x00_00_FF_00) >> 8;
        image::Rgb([red as u8, green as u8, blue as u8])
    });

    imgbuf.save("image.png").unwrap();
}

fn main() {
    env_logger::init();

    let scene = create_scene();
    let render_settings = RenderSettings {
        resolution: (2000, 2000),
        background_color: 0,
    };
    render(&scene, &render_settings);
}
