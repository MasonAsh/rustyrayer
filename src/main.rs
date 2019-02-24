use cgmath::{
    perspective, vec2, vec3, vec4, Angle, Array, Deg, EuclideanSpace, InnerSpace, Matrix4, Point3,
    Quaternion, Rad, Rotation, Rotation3, SquareMatrix, Transform, Vector2, Vector3, Zero,
};
use env_logger;
use image;
use image::{DynamicImage, GenericImageView, ImageBuffer, Pixel};
use log::{debug, info};
use rand;
use sdl2;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::pixels::PixelFormatEnum;
use std::collections::HashMap;

macro_rules! debug_if(
    ( $enable_debug:expr, target: $target:expr, $($debug_params:tt)* ) => (
        if $enable_debug {
            debug!(target: $target, $( $debug_params )* );
        }
    );
);

type Vec2 = Vector2<f32>;
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

fn transform_vec3(transform: Mat4, vec: Vec3) -> Vec3 {
    (transform * vec.extend(1.0)).xyz()
}

#[allow(clippy::many_single_char_names)]
fn bary_coords(p: Vec3, [a, b, c]: [Vec3; 3]) -> [f32; 3] {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;

    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    [u, v, w]
}

#[derive(Copy, Clone)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
    inv_dir: Vec3,
    inv_dir_sign: [usize; 3],
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3) -> Ray {
        let inv_dir = 1.0 / direction;
        // This is a precomputation of the signs to optimize intersect_aabb
        let inv_dir_sign = [
            if inv_dir.x < 0.0 { 1 } else { 0 },
            if inv_dir.y < 0.0 { 1 } else { 0 },
            if inv_dir.z < 0.0 { 1 } else { 0 },
        ];

        Ray {
            origin,
            direction,
            inv_dir,
            inv_dir_sign,
        }
    }
}

struct Light {
    position: Vec3,
}

struct AABB {
    bounds: [Vec3; 2],
}

impl AABB {
    fn min(&self) -> &Vec3 {
        &self.bounds[0]
    }

    fn max(&self) -> &Vec3 {
        &self.bounds[1]
    }
}

fn aabb(points: &[Vec3]) -> AABB {
    let infinite = Vec3::from_value(core::f32::INFINITY);
    let min = points.iter().fold(infinite, |acc, &point| {
        vec3(acc.x.min(point.x), acc.y.min(point.y), acc.z.min(point.z))
    });

    let max = points.iter().fold(Vec3::zero(), |acc, &point| {
        vec3(acc.x.max(point.x), acc.y.max(point.y), acc.z.max(point.z))
    });

    debug!(target: "bb", "min {:?} max {:?}", min, max);

    AABB { bounds: [min, max] }
}

struct Geometry {
    id: u32,
    vertices: Vec<Vec3>,
    uvs: Option<Vec<Vec2>>,
    indices: Vec<i32>,
    color: [f32; 4],
    diffuse_texture_id: Option<u64>,
    transform: Mat4,
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry {
            id: rand::random(),
            vertices: Vec::new(),
            uvs: None,
            indices: Vec::new(),
            color: [0.0; 4],
            diffuse_texture_id: None,
            transform: Mat4::identity(),
        }
    }
}

struct Texture {
    id: u64,
    image: DynamicImage,
}

struct TextureStorage {
    textures: Vec<Texture>,
}

impl TextureStorage {
    fn new() -> Self {
        TextureStorage {
            textures: Vec::new(),
        }
    }

    fn load(&mut self, path: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let image = image::open(path).unwrap();
        let mut hasher = DefaultHasher::new();
        Hash::hash(path, &mut hasher);
        let id = hasher.finish();
        self.textures.push(Texture { id, image });

        id
    }

    fn fetch(&self, id: u64) -> Option<&DynamicImage> {
        self.textures
            .iter()
            .find(|texture| texture.id == id)
            .map(|tex| &tex.image)
    }
}

#[derive(Debug)]
struct Face {
    vertices: [Vec3; 3],
    uvs: Option<[Vec2; 3]>,
    normal: Vec3,
    start_index: usize,
}

struct FaceIter<'a> {
    geometry: &'a Geometry,
    cache: &'a SceneCache,
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

        let FaceIter {
            cache, geometry, ..
        } = self;

        let indices_idx = (self.face_idx * 3) as usize;
        self.face_idx += 1;
        if indices_idx < geometry.indices.len() {
            let pretransformed_verts = &cache.world_space_verts[&geometry.id];
            let index0 = geometry.indices[indices_idx] as usize;
            let index1 = geometry.indices[indices_idx + 1] as usize;
            let index2 = geometry.indices[indices_idx + 2] as usize;
            let v0 = pretransformed_verts[index0];
            let v1 = pretransformed_verts[index1];
            let v2 = pretransformed_verts[index2];
            let uvs = geometry
                .uvs
                .as_ref()
                .map(|uvs| [uvs[index0], uvs[index1], uvs[index2]]);
            let vertices = [v0, v1, v2];
            let normal = calc_normal(&vertices);
            Some(Face {
                vertices,
                uvs,
                normal,
                start_index: indices_idx,
            })
        } else {
            None
        }
    }
}

impl Geometry {
    fn face_iter<'a>(&'a self, cache: &'a SceneCache) -> FaceIter<'a> {
        FaceIter {
            geometry: self,
            cache,
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
    ambient_light: [f32; 3],
    lights: Vec<Light>,
    texture_storage: TextureStorage,
}

impl Scene {
    fn get_geometry(&self, id: u32) -> Option<&Geometry> {
        self.geometries.iter().find(|geom| geom.id == id)
    }
}

/// The Scene structure is intended to be a "pure" representation of the scene.
/// SceneCache holds all the hot data that should be precomputed before a render.
struct SceneCache {
    world_space_verts: HashMap<u32, Vec<Vec3>>,
    bounding_boxes: HashMap<u32, AABB>,
}

fn build_cache(scene: &Scene) -> SceneCache {
    let world_space_verts: HashMap<u32, Vec<Vec3>> = scene
        .geometries
        .iter()
        .map(|geom| {
            let mut transformed_verts = Vec::with_capacity(geom.vertices.len());
            for vert in geom.vertices.iter() {
                transformed_verts.push(transform_vec3(geom.transform, *vert));
            }
            (geom.id, transformed_verts)
        })
        .collect();

    let bounding_boxes: HashMap<u32, AABB> = world_space_verts
        .iter()
        .map(|(id, verts)| (*id, aabb(verts)))
        .collect();

    SceneCache {
        world_space_verts,
        bounding_boxes,
    }
}

struct RenderSettings {
    resolution: (u32, u32),
    background_color: u32,
    debug_coord: Option<(u32, u32)>,
}

fn create_cube(diffuse_texture_id: Option<u64>) -> Geometry {
    let vertices = vec![
        vec3(-1.0, 1.0, -1.0),
        vec3(1.0, 1.0, 1.0),
        vec3(1.0, 1.0, -1.0),
        vec3(1.0, 1.0, 1.0),
        vec3(-1.0, -1.0, 1.0),
        vec3(1.0, -1.0, 1.0),
        vec3(-1.0, 1.0, 1.0),
        vec3(-1.0, -1.0, -1.0),
        vec3(-1.0, -1.0, 1.0),
        vec3(1.0, -1.0, -1.0),
        vec3(-1.0, -1.0, 1.0),
        vec3(-1.0, -1.0, -1.0),
        vec3(1.0, 1.0, -1.0),
        vec3(1.0, -1.0, 1.0),
        vec3(1.0, -1.0, -1.0),
        vec3(-1.0, 1.0, -1.0),
        vec3(1.0, -1.0, -1.0),
        vec3(-1.0, -1.0, -1.0),
        vec3(-1.0, 1.0, -1.0),
        vec3(-1.0, 1.0, 1.0),
        vec3(1.0, 1.0, 1.0),
        vec3(1.0, 1.0, 1.0),
        vec3(-1.0, 1.0, 1.0),
        vec3(-1.0, -1.0, 1.0),
        vec3(-1.0, 1.0, 1.0),
        vec3(-1.0, 1.0, -1.0),
        vec3(-1.0, -1.0, -1.0),
        vec3(1.0, -1.0, -1.0),
        vec3(1.0, -1.0, 1.0),
        vec3(-1.0, -1.0, 1.0),
        vec3(1.0, 1.0, -1.0),
        vec3(1.0, 1.0, 1.0),
        vec3(1.0, -1.0, 1.0),
        vec3(-1.0, 1.0, -1.0),
        vec3(1.0, 1.0, -1.0),
        vec3(1.0, -1.0, -1.0),
    ];

    let uvs = vec![
        vec2(0.625, 0.0),
        vec2(0.375, 0.25),
        vec2(0.375, 0.0),
        vec2(0.625, 0.25),
        vec2(0.375, 0.5),
        vec2(0.375, 0.25),
        vec2(0.625, 0.5),
        vec2(0.375, 0.75),
        vec2(0.375, 0.5),
        vec2(0.625, 0.75),
        vec2(0.375, 1.0),
        vec2(0.375, 0.75),
        vec2(0.375, 0.5),
        vec2(0.125, 0.75),
        vec2(0.125, 0.5),
        vec2(0.875, 0.5),
        vec2(0.625, 0.75),
        vec2(0.625, 0.5),
        vec2(0.625, 0.0),
        vec2(0.625, 0.25),
        vec2(0.375, 0.25),
        vec2(0.625, 0.25),
        vec2(0.625, 0.5),
        vec2(0.375, 0.5),
        vec2(0.625, 0.5),
        vec2(0.625, 0.75),
        vec2(0.375, 0.75),
        vec2(0.625, 0.75),
        vec2(0.625, 1.0),
        vec2(0.375, 1.0),
        vec2(0.375, 0.5),
        vec2(0.375, 0.75),
        vec2(0.125, 0.75),
        vec2(0.875, 0.5),
        vec2(0.875, 0.75),
        vec2(0.625, 0.75),
    ];

    let uvs = Some(uvs);
    let indices = (0..36).collect();
    let transform = Mat4::from_axis_angle(vec3(0.0, 1.0, 0.0), Deg(50.0));

    Geometry {
        vertices,
        uvs,
        indices,
        color: [1.0, 0.0, 0.0, 1.0],
        diffuse_texture_id,
        transform,
        ..Default::default()
    }
}

fn create_scene() -> Scene {
    let view = Mat4::look_at(
        Point3::new(0.0f32, 0.0f32, -5.0f32),
        Point3::new(0.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
    );
    let projection = perspective(Deg(75.0), 1.0, NEAR_PLANE, FAR_PLANE);
    let mut texture_storage = TextureStorage::new();
    let diffuse_texture_id = texture_storage.load("test.png");

    Scene {
        camera: Camera {
            view,
            projection,
            ortho_width: 5.0f32,
            ortho_height: 5.0f32,
        },
        geometries: vec![create_cube(Some(diffuse_texture_id))],
        ambient_light: [0.02, 0.02, 0.02],
        lights: vec![
            Light {
                position: vec3(1.5, 1.5, 1.5),
            },
            Light {
                position: vec3(0.0, 1.2, 0.0),
            },
        ],
        texture_storage,
    }
}

fn intersect(ray: &Ray, face: &Face, debug: bool) -> Option<(f32, Vec3)> {
    const TAG: &str = "intersection";

    let Ray {
        origin, direction, ..
    } = ray;
    let Face {
        vertices, /*normal,*/
        ..
    } = face;
    let (v0, v1, v2) = (vertices[0], vertices[1], vertices[2]);
    let (v0, v1, v2) = (v0.extend(1.0), v1.extend(1.0), v2.extend(1.0));
    let (v0, v1, v2) = (v0.xyz(), v1.xyz(), v2.xyz());

    debug_if!(
        debug,
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
        debug_if!(debug, target: TAG, "1 parallel");
        return None;
    }

    let d = normal.dot(v0);

    let t = -(normal.dot(*origin) + d) / normal_dot_ray_dir;
    if t < 0.0 {
        debug_if!(debug, target: TAG, "2 triangle behind t={}", t);
        return None;
    }

    let p = origin + t * direction;

    let edge0 = v0_v1;
    let vp0 = p - v0;
    let c = edge0.cross(vp0);
    if normal.dot(c) < 0.0 {
        debug_if!(debug, target: TAG, "3 P on right of edge0");
        return None;
    }

    let edge1 = v2 - v1;
    let vp1 = p - v1;
    let c = edge1.cross(vp1);
    if normal.dot(c) < 0.0 {
        debug_if!(debug, target: TAG, "4 P on right of edge1");
        return None;
    }

    let edge2 = v0 - v2;
    let vp2 = p - v2;
    let c = edge2.cross(vp2);
    if normal.dot(c) < 0.0 {
        debug_if!(debug, target: TAG, "5 P on right of edge2");
        return None;
    }

    // FIXME: why is p.z the wrong sign?
    let p = vec3(p.x, p.y, -p.z);

    Some((t, p))
}

fn min_vec3(a: &Vec3, b: &Vec3) -> Vec3 {
    vec3(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
}

fn max_vec3(a: &Vec3, b: &Vec3) -> Vec3 {
    vec3(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
}

fn transform_aabb(transform: &Mat4, bb: &AABB) -> AABB {
    let right = transform.x;
    let up = transform.y;
    let back = transform.z;
    let xa = right * bb.min().x;
    let xb = right * bb.max().x;
    let ya = up * bb.min().y;
    let yb = up * bb.max().y;
    let za = back * bb.min().z;
    let zb = back * bb.max().z;

    let xa = xa.xyz();
    let xb = xb.xyz();
    let ya = ya.xyz();
    let yb = yb.xyz();
    let za = za.xyz();
    let zb = zb.xyz();

    let min = min_vec3(&xa, &xb) + min_vec3(&ya, &yb) + min_vec3(&za, &zb);
    let max = max_vec3(&xa, &xb) + max_vec3(&ya, &yb) + max_vec3(&za, &zb);
    AABB { bounds: [min, max] }
}

fn transform_dir(transform: &Mat4, dir: &Vec3) -> Vec3 {
    use cgmath::Matrix3;
    let rot = Matrix3::from_cols(transform.x.xyz(), transform.y.xyz(), transform.z.xyz());

    let rot: Quaternion<f32> = rot.into();

    let res = rot.rotate_vector(*dir);

    vec3(res.x, res.y, res.z)
}

fn intersect_aabb(ray: &Ray, bb: &AABB, view: &Mat4) -> bool {
    let Ray {
        origin,
        inv_dir,
        inv_dir_sign,
        ..
    } = ray;

    let bb = transform_aabb(view, bb);

    let tmin = (bb.bounds[inv_dir_sign[0]].x - origin.x) * inv_dir.x;
    let tmax = (bb.bounds[1 - inv_dir_sign[0]].x - origin.x) * inv_dir.x;

    let tymin = (bb.bounds[inv_dir_sign[1]].y - origin.y) * inv_dir.y;
    let tymax = (bb.bounds[1 - inv_dir_sign[1]].y - origin.y) * inv_dir.y;

    if tmin > tymax || tymin > tmax {
        return false;
    }

    let (tmin, tmax) = (tmin.max(tymin), tmax.min(tymax));

    let tzmin = (bb.bounds[inv_dir_sign[2]].z - origin.z) * inv_dir.z;
    let tzmax = (bb.bounds[1 - inv_dir_sign[2]].z - origin.z) * inv_dir.z;

    if tmin > tzmax || tzmin > tmax {
        return false;
    }

    true
}

#[allow(clippy::many_single_char_names)]
fn sample(
    hit_point: Vec3,     // in view space
    vertices: [Vec3; 3], // in view space
    [uv0, uv1, uv2]: [Vec2; 3],
    image: &DynamicImage,
    debug: bool,
) -> [f32; 4] {
    let [b0, b1, b2] = bary_coords(hit_point, vertices);

    let uv = (b0 * uv0) + (b1 * uv1) + (b2 * uv2);

    debug_if!(debug, target: "sample", "uv0={:?} uv1={:?} uv1={:?}", uv0, uv1, uv2);
    debug_if!(debug, target: "sample", "sample: vertices={:?} hit={:?} bary={} {} {} uv={:?}", vertices, hit_point, b0, b1, b2, uv);

    let (width, height) = image.dimensions();
    let (width, height) = (width as f32, height as f32);
    let (x, y) = (width * uv.x, height * uv.y);
    let y = height - y;
    let (x, y) = (x.min(width - 1.0).max(0.0), y.min(height - 1.0).max(0.0));
    let (x, y) = (x as u32, y as u32);
    let pixel = image.get_pixel(x, y);

    let channels = pixel.channels();
    let (r, g, b, a) = (channels[0], channels[1], channels[2], channels[3]);
    let (r, g, b, a) = (f32::from(r), f32::from(g), f32::from(b), f32::from(a));
    let (r, g, b, a) = (r / 255.0, g / 255.0, b / 255.0, a / 255.0);

    debug_if!(debug, target: "sample", "sampled color is {:?}", (r, g, b, a));

    [r as f32, g as f32, b as f32, a as f32]
}

struct Hit {
    face: Face,
    geometry_id: u32,
    point: Vec3,
    //distance: f32,
    ray: Ray,
}

fn trace(scene: &Scene, cache: &SceneCache, ray: &Ray, min_dist: f32, debug: bool) -> Option<Hit> {
    let mut hit_dist = FAR_PLANE + 1.0f32;
    let mut hit_point = vec3(0.0f32, 0.0f32, 0.0f32);
    let mut hit_geom_id = 0;
    let mut hit_face = None;

    for geom in scene.geometries.iter() {
        let bounding_box = &cache.bounding_boxes[&geom.id];
        let hit_bounding_box = intersect_aabb(&ray, bounding_box, &scene.camera.view);
        debug_if!(debug, target: "intersect_aabb", "AABB intersect: {}", hit_bounding_box);
        if !hit_bounding_box {
            continue;
        }

        for face in geom.face_iter(cache) {
            if let Some((dist, p)) = intersect(&ray, &face, debug) {
                if dist < hit_dist && dist > min_dist {
                    hit_dist = dist;
                    hit_point = p;
                    hit_geom_id = geom.id;
                    hit_face = Some(face);
                }
            }
        }
    }

    hit_face.and_then(|face| {
        Some(Hit {
            face,
            geometry_id: hit_geom_id,
            point: hit_point,
            //distance: hit_dist,
            ray: *ray,
        })
    })
}

fn shade(
    scene: &Scene,
    cache: &SceneCache,
    hit: Option<&Hit>,
    background_color: u32,
    debug: bool,
) -> u32 {
    if let Some(hit) = hit {
        (&scene.lights)
            .iter()
            .map(|light| {
                let light_pos = light.position;
                let hit_to_light = light_pos - hit.point;
                let hit_to_light = hit_to_light.normalize();

                // Cast a ray towards the light
                // TODO: The minimum distance is the trace was determined by trial and error to minimize
                // self intersecting artifacts but isn't perfect.
                let ray = Ray::new(hit.point, hit_to_light);
                let is_shadowed = trace(scene, cache, &ray, 0.000_005, debug).is_some();
                debug_if!(debug, target: "shade", "shadowed: {}", is_shadowed);

                let angle = hit_to_light.angle(hit.face.normal);
                let shade = if !is_shadowed { angle.cos().abs() } else { 0.0 };

                debug_if!(debug, target: "shade", "shade={} P={:?} angle={:?} light={:?} raydir={:?}", shade, hit.point, angle, light_pos, hit.ray.direction);

                debug_if!(debug, target: "shade", "FACE={:?}", hit.face);

                let geometry = &scene.get_geometry(hit.geometry_id).unwrap();

                let texture_id = geometry.diffuse_texture_id;
                let image = texture_id.and_then(|texture_id| {
                    scene.texture_storage.fetch(texture_id)
                });

                let diffuse_color =
                    match (image, hit.face.uvs) {
                        (Some(image), Some(uvs)) => {
                            sample(hit.point, hit.face.vertices, uvs, &image, debug)
                        },
                        _ => {
                            geometry.color
                        }
                    };

                let ambient_light = scene.ambient_light;

                let final_color = [
                    (ambient_light[0]) + (diffuse_color[0] * shade),
                    (ambient_light[1]) + (diffuse_color[1] * shade),
                    (ambient_light[2]) + (diffuse_color[2] * shade),
                    diffuse_color[3],
                ];

                compact_color(final_color)
            })
            .sum()
    } else {
        background_color
    }
}

fn render_one_pixel(
    (x, y): (u32, u32),
    scene: &Scene,
    cache: &SceneCache,
    render_settings: &RenderSettings,
) -> image::Rgb<u8> {
    let (res_w, res_h) = render_settings.resolution;
    let ortho_width = scene.camera.ortho_width;
    let ortho_height = scene.camera.ortho_height;

    let debug = render_settings
        .debug_coord
        .map_or(false, |coord| coord.0 == x && coord.1 == y);

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
        -1.0f32,
        1.0f32,
    );
    let inverse_projection = scene.camera.projection.inverse_transform().unwrap();
    let inverse_view = scene.camera.view.inverse_transform().unwrap();
    let ray_origin = transform_vec3(inverse_view, ray_origin);
    let ray_direction = inverse_projection * ray_direction;
    let mut ray_direction = ray_direction.xyz().normalize();
    ray_direction.y *= -1.0;
    let ray_direction = transform_dir(&inverse_view, &ray_direction);
    debug_if!(debug, target: "ray", "ray: {:?}", ray_direction);

    let ray = Ray::new(ray_origin, ray_direction);
    let hit = trace(&scene, &cache, &ray, 0.0, debug);
    let color = shade(
        scene,
        cache,
        hit.as_ref(),
        render_settings.background_color,
        debug,
    );
    let red = (color & 0xFF_00_00_00) >> 24;
    let green = (color & 0x00_FF_00_00) >> 16;
    let blue = (color & 0x00_00_FF_00) >> 8;
    image::Rgb([red as u8, green as u8, blue as u8])
}

fn render(
    scene: &Scene,
    cache: &SceneCache,
    render_settings: &RenderSettings,
) -> ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let (res_w, res_h) = render_settings.resolution;

    let begin = std::time::Instant::now();

    let imgbuf = ImageBuffer::from_fn(res_w, res_h, |x, y| {
        render_one_pixel((x, y), scene, cache, render_settings)
    });

    let end = std::time::Instant::now();
    let elapsed = end.duration_since(begin);

    info!(target: "perf", "Render finished in {}s{}ms", elapsed.as_secs(), elapsed.subsec_millis());

    imgbuf
}

fn interactive_loop(scene: &mut Scene, render_settings: &mut RenderSettings) {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let (res_w, res_h) = render_settings.resolution;

    let window = video_subsystem
        .window("rustyrayer", res_w, res_h)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();
    let texture_creator = canvas.texture_creator();
    let mut texture = texture_creator
        .create_texture_streaming(PixelFormatEnum::RGB24, res_w, res_h)
        .unwrap();

    canvas.clear();
    canvas.present();

    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut needs_redraw = true;

    let mut mouse_down = false;

    // Camera orbit parameters
    let mut orbit_dist = 5.0f32;
    let mut yaw = Rad(0.0f32);
    let mut pitch = Rad(0.0f32);
    let mut cam_offset = Vec3::zero();

    let mut model_theta = Rad(0.0f32);

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::KeyDown {
                    keycode: Some(Keycode::W),
                    ..
                } => {
                    orbit_dist *= 0.7;
                    needs_redraw = true;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::S),
                    ..
                } => {
                    orbit_dist *= 1.3;
                    needs_redraw = true;
                }
                Event::MouseButtonDown {
                    mouse_btn: MouseButton::Left,
                    x,
                    y,
                    ..
                } if x > 0 && y > 0 && x < res_w as i32 && y < res_h as i32 => {
                    mouse_down = true;
                    needs_redraw = true;
                    render_settings.debug_coord = Some((x as u32, y as u32));
                    info!(target: "dbgcoord", "dbgcoord {} {}", x, y);
                }
                Event::MouseButtonUp {
                    mouse_btn: MouseButton::Left,
                    ..
                } => {
                    mouse_down = false;
                    render_settings.debug_coord = None;
                }
                Event::MouseMotion { xrel, yrel, .. } if mouse_down => {
                    yaw -= Rad((xrel as f32) * 0.008);
                    pitch += Rad((yrel as f32) * 0.008);
                    needs_redraw = true;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::A),
                    ..
                } => {
                    needs_redraw = true;
                    cam_offset.x -= 1.0;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::D),
                    ..
                } => {
                    needs_redraw = true;
                    cam_offset.x += 1.0;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::Left),
                    ..
                } => {
                    needs_redraw = true;
                    model_theta += Rad::from(Deg(5.0));
                }
                Event::KeyDown {
                    keycode: Some(Keycode::Right),
                    ..
                } => {
                    needs_redraw = true;
                    model_theta -= Rad::from(Deg(5.0));
                }
                _ => {}
            }
        }

        if needs_redraw {
            // Orbit the camera with our yaw and pitch angles
            let camera_position = Quaternion::from_angle_y(-yaw)
                * Quaternion::from_angle_x(pitch)
                * vec3(0.0, 0.0, -orbit_dist)
                + cam_offset;

            // Mat4::look_at requires a Point3
            let camera_position = Point3::from_vec(camera_position);

            scene.camera.view = Mat4::look_at(
                camera_position,
                Point3::from_vec(cam_offset),
                vec3(0.0, 1.0, 0.0),
            );

            scene.geometries[0].transform = Mat4::from_angle_y(model_theta);

            let cache = build_cache(scene);

            canvas.clear();

            texture
                .with_lock(None, |buffer: &mut [u8], _pitch: usize| {
                    let imgbuf = render(scene, &cache, render_settings);
                    buffer.copy_from_slice(&*imgbuf);
                })
                .unwrap();

            canvas.copy(&texture, None, None).unwrap();

            canvas.present();

            needs_redraw = false;
        }
    }
}

fn main() {
    env_logger::init();

    let mut scene = create_scene();
    let mut args_iter = std::env::args();
    let single_debug_coord = args_iter.position(|arg| arg == "--sdc").and_then(|_| {
        let debug_coord = args_iter.next();
        debug_coord.and_then(|debug_coord| {
            let split: Vec<_> = debug_coord.split(',').collect();
            let map_str_number = |input: &&str| input.parse::<u32>().ok();
            split.get(0).and_then(map_str_number).and_then(|val0| {
                split
                    .get(1)
                    .and_then(map_str_number)
                    .map(|val1| (val0, val1))
            })
        })
    });

    let mut render_settings = RenderSettings {
        resolution: (640, 480),
        background_color: 0,
        debug_coord: single_debug_coord,
    };

    let interactive_mode = std::env::args().any(|arg| arg == "--interactive");

    let cache = build_cache(&scene);

    if interactive_mode {
        interactive_loop(&mut scene, &mut render_settings);
    } else if let Some(coord) = single_debug_coord {
        let color = render_one_pixel(coord, &scene, &cache, &render_settings);
        println!("Pixel color is {:?}", color);
    } else {
        let imgbuf = render(&scene, &cache, &render_settings);
        imgbuf.save("image.png").unwrap();
    }
}
