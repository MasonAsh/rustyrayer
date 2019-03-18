use cgmath::{
    vec2, vec3, Angle, Array, EuclideanSpace, InnerSpace, Matrix, Matrix4, Point3, Quaternion, Rad,
    Rotation3, SquareMatrix, Transform, Vector2, Vector3, Zero,
};
use env_logger;
use gltf;
use image;
use image::{DynamicImage, GenericImageView, ImageBuffer, Pixel};
use log::{debug, info, warn};
use rand;
use rayon::prelude::*;
use sdl2;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::pixels::PixelFormatEnum;
use std::path::Path;

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

const FAR_PLANE: f32 = 10_000_000_000.0f32;
const TILE_SIZE: u32 = 8;

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

struct Material {
    color: [f32; 4],
    diffuse_texture_id: Option<u64>,
    metallic: f32,
    roughness: f32,
}

struct Geometry {
    id: u32,
    vertices: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Option<Vec<Vec2>>,
    indices: Vec<u32>,
    material: Material,
    transform: Mat4,
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry {
            id: rand::random(),
            vertices: Vec::new(),
            normals: Vec::new(),
            uvs: None,
            indices: Vec::new(),
            material: Material {
                color: [1.0; 4],
                diffuse_texture_id: None,
                metallic: 0.0,
                roughness: 0.0,
            },
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

    fn load_gltf_texture(&mut self, image: &gltf::image::Data) -> Option<u64> {
        let buf = ImageBuffer::from_raw(image.width, image.height, image.pixels.clone());
        buf.map(|buf| {
            let image = DynamicImage::ImageRgba8(buf);
            let id = rand::random();
            self.textures.push(Texture { id, image });

            id
        })
    }

    fn fetch(&self, id: u64) -> Option<&DynamicImage> {
        self.textures
            .iter()
            .find(|texture| texture.id == id)
            .map(|tex| &tex.image)
    }
}

#[derive(Debug, Clone)]
struct Face {
    vertices: [Vec3; 3],
    normals: [Vec3; 3],
    uvs: Option<[Vec2; 3]>,
    normal: Vec3,
    start_index: usize,
}

struct FaceIter<'a> {
    geometry: &'a Geometry,
    geometry_idx: usize,
    cache: &'a SceneCache,
    face_idx: u32,
}

impl Iterator for FaceIter<'_> {
    type Item = Face;

    fn next(&mut self) -> Option<Face> {
        fn calc_face_normal(normals: &[Vec3; 3]) -> Vec3 {
            normals.iter().sum::<Vec3>() / 3.0
        }

        let FaceIter {
            cache,
            geometry,
            geometry_idx,
            ..
        } = self;

        let indices_idx = (self.face_idx * 3) as usize;
        self.face_idx += 1;
        if indices_idx < geometry.indices.len() {
            let pretransformed_verts = &cache.world_space_verts[*geometry_idx];
            let pretransformed_normals = &cache.world_space_normals[*geometry_idx];
            let index0 = geometry.indices[indices_idx] as usize;
            let index1 = geometry.indices[indices_idx + 1] as usize;
            let index2 = geometry.indices[indices_idx + 2] as usize;
            let v0 = pretransformed_verts[index0];
            let v1 = pretransformed_verts[index1];
            let v2 = pretransformed_verts[index2];
            let n0 = pretransformed_normals[index0];
            let n1 = pretransformed_normals[index1];
            let n2 = pretransformed_normals[index2];
            let uvs = geometry
                .uvs
                .as_ref()
                .map(|uvs| [uvs[index0], uvs[index1], uvs[index2]]);
            let vertices = [v0, v1, v2];
            let normals = [n0, n1, n2];
            let normal = calc_face_normal(&normals);
            Some(Face {
                vertices,
                normals,
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
    fn face_iter<'a>(&'a self, cache: &'a SceneCache, geometry_idx: usize) -> FaceIter<'a> {
        FaceIter {
            geometry: self,
            cache,
            face_idx: 0,
            geometry_idx,
        }
    }
}

struct Camera {
    camera_to_world: Mat4,
    ortho_width: f32,
    ortho_height: f32,
}

struct Scene {
    camera: Camera,
    geometries: Vec<Geometry>,
    ambient_light: Vec3,
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
    world_space_verts: Vec<Vec<Vec3>>,
    world_space_normals: Vec<Vec<Vec3>>,
    bounding_boxes: Vec<AABB>,
}

fn build_cache(scene: &Scene) -> SceneCache {
    let world_space_verts: Vec<Vec<Vec3>> = scene
        .geometries
        .iter()
        .map(|geom| {
            let mut transformed_verts = Vec::with_capacity(geom.vertices.len());
            for vert in geom.vertices.iter() {
                transformed_verts.push(transform_vec3(geom.transform, *vert));
            }
            transformed_verts
        })
        .collect();

    let world_space_normals: Vec<Vec<Vec3>> = scene
        .geometries
        .iter()
        .map(|geom| {
            let normal_matrix = geom.transform.inverse_transform().unwrap().transpose();
            let mut transformed_normals = Vec::with_capacity(geom.normals.len());
            for normal in geom.normals.iter() {
                let normal = normal_matrix * normal.extend(0.0);
                transformed_normals.push(normal.xyz().normalize());
            }
            transformed_normals
        })
        .collect();

    let bounding_boxes: Vec<AABB> = world_space_verts.iter().map(|verts| aabb(verts)).collect();

    SceneCache {
        world_space_verts,
        world_space_normals,
        bounding_boxes,
    }
}

struct RenderSettings {
    resolution: (u32, u32),
    background_color: Vec3,
    debug_coord: Option<(u32, u32)>,
    max_ray_bounces: u32,
}

fn generate_smooth_normals(indices: &[u32], positions: &[Vec3]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::zero(); positions.len()];

    for triangle in indices.chunks(3) {
        let (i0, i1, i2) = (triangle[0], triangle[1], triangle[2]);
        let a = positions[i0 as usize];
        let b = positions[i1 as usize];
        let c = positions[i2 as usize];
        let p = (b - a).cross(c - a);
        normals[i0 as usize] += p;
        normals[i1 as usize] += p;
        normals[i2 as usize] += p;
    }

    normals
}

fn read_gltf_mesh(
    mesh: &gltf::Mesh,
    transform: &Mat4,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    texture_storage: &mut TextureStorage,
) -> Vec<Geometry> {
    use gltf::mesh::Mode;
    use gltf::mesh::Semantic;

    let valid_primitives = mesh
        .primitives()
        .filter(|primitives| {
            if primitives.mode() != Mode::Triangles {
                warn!("found non-triangle primitives which will be ignored");
                false
            } else {
                true
            }
        })
        .filter(|primitives| {
            if primitives.indices().is_none() {
                warn!("found primitives without indices which will be ignored");
                false
            } else {
                true
            }
        })
        .filter(|primitives| {
            if primitives.get(&Semantic::Positions).is_none() {
                warn!("found primitives without positions which will be ignored");
                false
            } else {
                true
            }
        });

    valid_primitives
        .map(|prim| {
            let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions = reader.read_positions().unwrap();
            let positions: Vec<_> = positions
                .map(|position| vec3(position[0], position[1], position[2]))
                .collect();

            let indices = reader.read_indices().unwrap();
            let indices = indices.into_u32().collect::<Vec<_>>();

            let normals = if let Some(normals) = reader.read_normals() {
                normals
                    .map(|normal| vec3(normal[0], normal[1], normal[2]))
                    .collect::<Vec<_>>()
            } else {
                generate_smooth_normals(&indices, &positions)
            };

            let material = prim.material();
            let pbr = material.pbr_metallic_roughness();
            let color = pbr.base_color_factor();
            let mut diffuse_texture_id = None;
            let mut uvs = None;
            if let Some(texinfo) = pbr.base_color_texture() {
                let tex_coord_idx = texinfo.tex_coord();

                let gltf_tex = texinfo.texture();

                diffuse_texture_id = texture_storage.load_gltf_texture(&images[gltf_tex.index()]);

                let in_uvs = reader.read_tex_coords(tex_coord_idx);
                uvs = in_uvs.map(|uvs| {
                    uvs.into_f32()
                        .map(|uv| vec2(uv[0], uv[1]))
                        .collect::<Vec<_>>()
                });
            }

            let metallic = pbr.metallic_factor();
            let roughness = pbr.roughness_factor();

            debug!(target: "material", "base color: {:?} diffuse_texture: {} metallic: {} roughness: {}", color, diffuse_texture_id.is_some(), metallic, roughness);

            Geometry {
                vertices: positions,
                normals,
                indices,
                uvs,
                transform: *transform,
                material: Material {
                    color,
                    diffuse_texture_id,
                    metallic,
                    roughness,
                },
                ..Default::default()
            }
        })
        .collect()
}

fn read_gltf_node_and_children(
    node: &gltf::Node,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    texture_storage: &mut TextureStorage,
    geometries: &mut Vec<Geometry>,
    camera: &mut Camera,
) {
    let transform: Mat4 = node.transform().matrix().into();
    let transform = parent_transform * transform;

    if let Some(mesh) = node.mesh() {
        geometries.extend(read_gltf_mesh(
            &mesh,
            &transform,
            buffers,
            images,
            texture_storage,
        ));
    }

    if node.camera().is_some() {
        camera.camera_to_world = transform;
    }

    for child in node.children() {
        read_gltf_node_and_children(
            &child,
            transform,
            buffers,
            images,
            texture_storage,
            geometries,
            camera,
        );
    }
}

/// Loads a gltf scene from a specified path.
/// Currently lights are hardcoded, as lights are not a core part of the gltf file format.
/// Lights are available in gltf through the KHR_punctual_lights extensions, but for the
/// time being there is no way to access the extension data through the gltf crate.
fn load_scene(path: &Path) -> Result<Scene, gltf::Error> {
    let (gltf, buffers, images) = gltf::import(path)?;

    let mut geometries: Vec<Geometry> = Vec::new();

    let mut texture_storage = TextureStorage::new();

    let default_view = Mat4::look_at(
        Point3::new(0.0f32, 0.0f32, -5.0f32),
        Point3::new(0.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
    );
    let camera_to_world = default_view.inverse_transform().unwrap();

    let mut camera = Camera {
        camera_to_world,
        ortho_width: 5.0f32,
        ortho_height: 5.0f32,
    };

    for scene in gltf.scenes() {
        for node in scene.nodes() {
            read_gltf_node_and_children(
                &node,
                Mat4::identity(),
                &buffers,
                &images,
                &mut texture_storage,
                &mut geometries,
                &mut camera,
            );
        }
    }

    Ok(Scene {
        camera,
        geometries,
        ambient_light: vec3(0.04, 0.04, 0.04),
        lights: vec![Light {
            position: vec3(2.0, 2.0, 2.0),
        }],
        texture_storage,
    })
}

#[allow(clippy::many_single_char_names)]
/// Checks if a ray hits a triangle using the Möller–Trumbore algorithm
fn intersect(ray: &Ray, face: &Face) -> Option<(f32, Vec3)> {
    debug_assert!(ray.direction.magnitude() < 1.0 + 0.01);
    debug_assert!(ray.direction.magnitude() > 1.0 - 0.01);

    let Ray {
        origin, direction, ..
    } = ray;
    let [v0, v1, v2] = face.vertices;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = direction.cross(edge2);
    let a = edge1.dot(h);
    if a > -core::f32::EPSILON && a < -core::f32::EPSILON {
        return None;
    }
    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * s.dot(h);
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = s.cross(edge1);
    let v = f * direction.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * edge2.dot(q);
    if t > core::f32::EPSILON {
        let p = origin + direction * t;
        Some((t, p))
    } else {
        None
    }
}

fn transform_dir(transform: &Mat4, dir: &Vec3) -> Vec3 {
    //use cgmath::Matrix3;
    //let rot = Matrix3::from_cols(transform.x.xyz(), transform.y.xyz(), transform.z.xyz());

    //let rot: Quaternion<f32> = rot.into();

    //let res = rot.rotate_vector(*dir);

    //vec3(res.x, res.y, res.z)
    (transform.inverse_transform().unwrap().transpose() * dir.extend(0.0))
        .xyz()
        .normalize()
}

fn intersect_aabb(ray: &Ray, bb: &AABB) -> bool {
    let Ray {
        origin,
        inv_dir,
        inv_dir_sign,
        ..
    } = ray;

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

fn sample_normal(
    hit_point: Vec3,     // in view space
    vertices: [Vec3; 3], // in view space
    [n0, n1, n2]: [Vec3; 3],
) -> Vec3 {
    let [b0, b1, b2] = bary_coords(hit_point, vertices);

    ((b0 * n0) + (b1 * n1) + (b2 * n2)).normalize()
}

#[allow(clippy::many_single_char_names)]
fn sample_image(
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

#[derive(Clone)]
struct Hit {
    face: Face,
    geometry_id: u32,
    point: Vec3,
    ray: Ray,
}

fn trace(scene: &Scene, cache: &SceneCache, ray: &Ray, min_dist: f32, debug: bool) -> Option<Hit> {
    let mut hit_dist = FAR_PLANE + 1.0f32;
    let mut hit_point = vec3(0.0f32, 0.0f32, 0.0f32);
    let mut hit_geom_id = 0;
    let mut hit_face = None;

    for (geom_idx, geom) in scene.geometries.iter().enumerate() {
        let bounding_box = &cache.bounding_boxes[geom_idx];
        let hit_bounding_box = intersect_aabb(&ray, bounding_box);
        debug_if!(debug, target: "intersect_aabb", "AABB intersect: {}", hit_bounding_box);
        if !hit_bounding_box {
            continue;
        }

        for face in geom.face_iter(cache, geom_idx) {
            if let Some((dist, p)) = intersect(&ray, &face) {
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
    background_color: Vec3,
    debug: bool,
) -> Vec3 {
    if let Some(hit) = hit {
        (&scene.lights)
            .par_iter()
            .map(|light| {
                let light_pos = light.position;
                let hit_to_light = light_pos - hit.point;
                let hit_light_dist2 = hit_to_light.magnitude2();
                let hit_to_light = hit_to_light.normalize();

                // Cast a ray towards the light
                // TODO: The minimum distance is the trace was determined by trial and error to minimize
                // self intersecting artifacts but isn't perfect.
                let ray = Ray::new(hit.point, hit_to_light);
                let is_shadowed = trace(scene, cache, &ray, 0.005, debug).filter(|shadow_hit| (shadow_hit.point - hit.point).magnitude2()  < hit_light_dist2).is_some();
                debug_if!(debug, target: "shade", "shadowed: {}", is_shadowed);

                let normal = sample_normal(hit.point, hit.face.vertices, hit.face.normals);
                let angle = hit_to_light.angle(normal);
                let shade = if !is_shadowed { angle.cos().abs() } else { 0.0 };

                debug_if!(debug, target: "shade", "shade={} P={:?} angle={:?} light={:?} raydir={:?}", shade, hit.point, angle, light_pos, hit.ray.direction);

                debug_if!(debug, target: "shade", "FACE={:?}", hit.face);

                let geometry = &scene.get_geometry(hit.geometry_id).unwrap();

                let texture_id = geometry.material.diffuse_texture_id;
                let image = texture_id.and_then(|texture_id| {
                    scene.texture_storage.fetch(texture_id)
                });

                let diffuse_color =
                    match (image, hit.face.uvs) {
                        (Some(image), Some(uvs)) => {
                            sample_image(hit.point, hit.face.vertices, uvs, &image, debug)
                        },
                        _ => {
                            geometry.material.color
                        }
                    };

                let ambient_light = scene.ambient_light;

                vec3(
                    (ambient_light[0]) + (diffuse_color[0] * shade),
                    (ambient_light[1]) + (diffuse_color[1] * shade),
                    (ambient_light[2]) + (diffuse_color[2] * shade),
                )
            })
            .sum()
    } else {
        background_color
    }
}

fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    incident - (2.0 * incident.dot(normal) * normal)
}

fn cast_reflection_rays(
    primary_ray: &Ray,
    hit: &Hit,
    scene: &Scene,
    cache: &SceneCache,
    render_settings: &RenderSettings,
    debug: bool,
) -> Vec3 {
    fn reflect_ray(hit: &Hit, ray_direction: Vec3) -> Ray {
        let normal = sample_normal(hit.point, hit.face.vertices, hit.face.normals);
        let reflection_origin = hit.point;
        let reflect_direction = reflect(ray_direction, normal);
        let reflection_origin = reflection_origin + reflect_direction * 0.1;
        Ray::new(reflection_origin, reflect_direction)
    }

    let RenderSettings {
        background_color,
        max_ray_bounces,
        ..
    } = render_settings;

    let mut hit = Some(hit.clone());
    let mut ray_direction = primary_ray.direction;
    let mut color = Vec3::zero();
    let hit_geom = scene
        .get_geometry(hit.as_ref().unwrap().geometry_id)
        .unwrap();
    if hit_geom.material.metallic < core::f32::EPSILON {
        return color;
    }

    for _ in 0..*max_ray_bounces {
        if hit.is_none() {
            break;
        }

        let hit_geom = scene
            .get_geometry(hit.as_ref().unwrap().geometry_id)
            .unwrap();
        let inverse_roughness = (1.0 - hit_geom.material.roughness.max(0.000_001)).abs();
        debug_if!(debug, target: "shade", "roughness {} 1/roughness {}", hit_geom.material.roughness, inverse_roughness);

        let reflection_ray = reflect_ray(&hit.unwrap(), ray_direction);
        hit = trace(scene, cache, &reflection_ray, 0.00001, debug);

        if hit.is_none() {
            color += inverse_roughness * background_color;
            break;
        }

        color += inverse_roughness * shade(scene, cache, hit.as_ref(), *background_color, debug);
        ray_direction = reflection_ray.direction;

        if hit_geom.material.metallic < core::f32::EPSILON {
            break;
        }
    }

    color
}

fn cast_primary_ray(
    ray: &Ray,
    scene: &Scene,
    cache: &SceneCache,
    render_settings: &RenderSettings,
    debug: bool,
) -> Vec3 {
    let hit = trace(scene, cache, ray, 0.0, debug);

    let background_color = render_settings.background_color;

    let mut color: Vec3 = background_color;

    if let Some(hit) = hit {
        color = shade(
            scene,
            cache,
            Some(&hit),
            render_settings.background_color,
            debug,
        );

        color += cast_reflection_rays(ray, &hit, scene, cache, render_settings, debug);
    }

    color
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
    let ortho_x = (x / res_w) * ortho_width - (ortho_width / 2.0);
    let ortho_y = -(y / res_h) * ortho_height + (ortho_height / 2.0);
    let ray_origin = vec3(0.0, 0.0, 0.0f32);
    let ray_direction = vec3(ortho_x / ortho_width, ortho_y / ortho_height, -1.0f32);
    let camera_to_world = scene.camera.camera_to_world;
    let ray_origin = transform_vec3(camera_to_world, ray_origin);
    let ray_direction = transform_dir(&camera_to_world, &ray_direction);
    let ray_direction = ray_direction.normalize();
    debug_if!(debug, target: "ray", "ray: {:?}", ray_direction);

    let ray = Ray::new(ray_origin, ray_direction);
    let color = cast_primary_ray(&ray, scene, cache, render_settings, debug);

    // Clamp the color to the 0.0 to 1.0 range.
    let color = vec3(
        color.x.min(1.0).max(0.0),
        color.y.min(1.0).max(0.0),
        color.z.min(1.0).max(0.0),
    );

    let red = color.x * 255.0;
    let green = color.y * 255.0;
    let blue = color.z * 255.0;
    let red = red as u8;
    let green = green as u8;
    let blue = blue as u8;
    image::Rgb([red, green, blue])
}

fn render_tile(
    start_x: u32,
    start_y: u32,
    end_x: u32,
    end_y: u32,
    scene: &Scene,
    cache: &SceneCache,
    render_settings: &RenderSettings,
) -> Vec<image::Rgb<u8>> {
    let mut colors = Vec::with_capacity(((end_x - start_x) * (end_y - start_y)) as usize);

    for y in start_y..end_y {
        for x in start_x..end_x {
            let color = render_one_pixel((x, y), scene, cache, render_settings);
            colors.push(color);
        }
    }

    colors
}

fn render(
    scene: &Scene,
    cache: &SceneCache,
    render_settings: &RenderSettings,
    target: &mut ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) {
    let (res_w, res_h) = render_settings.resolution;

    let begin = std::time::Instant::now();

    // Render tiles in  parallel
    let tiles = (0..((res_h + TILE_SIZE - 1) / TILE_SIZE))
        .into_par_iter()
        .flat_map(|y_chunk| {
            (0..((res_w + TILE_SIZE - 1) / TILE_SIZE))
                .into_par_iter()
                .map_with(y_chunk, |y_chunk, x_chunk| {
                    let x = x_chunk * TILE_SIZE;
                    let y = *y_chunk * TILE_SIZE;
                    let end_x = (x + TILE_SIZE).min(res_w - 1);
                    let end_y = (y + TILE_SIZE).min(res_h - 1);
                    render_tile(x, y, end_x, end_y, scene, cache, render_settings)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Combine them into the final image.
    let tile_cols = res_w / TILE_SIZE;
    let tile_cols = tile_cols as usize;
    for (i, tile) in tiles.iter().enumerate() {
        let x_chunk = i % tile_cols;
        let y_chunk = i / tile_cols;
        let x_chunk = x_chunk as u32;
        let y_chunk = y_chunk as u32;
        let start_x = x_chunk * TILE_SIZE;
        let start_y = y_chunk * TILE_SIZE;
        let end_x = (start_x + TILE_SIZE).min(res_w - 1);
        let end_y = (start_y + TILE_SIZE).min(res_h - 1);
        for y in start_y..end_y {
            for x in start_x..end_x {
                let tile_x = x - start_x;
                let tile_y = y - start_y;
                let tile_width = end_x - start_x;
                let index = (tile_width * tile_y + tile_x) as usize;
                target.put_pixel(x, y, tile[index]);
            }
        }
    }

    let end = std::time::Instant::now();
    let elapsed = end.duration_since(begin);

    info!(target: "perf", "Render finished in {}s{}ms", elapsed.as_secs(), elapsed.subsec_millis());
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

    let mut render_buffer = ImageBuffer::new(res_w, res_h);

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
    let mut orbit_dist = 10.0f32;
    let mut yaw = Rad(0.0f32);
    let mut pitch = Rad(0.0f32);
    let mut cam_offset = Vec3::zero();

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
                _ => {}
            }
        }

        if needs_redraw {
            // Orbit the camera with our yaw and pitch angles
            let camera_position = Quaternion::from_angle_y(yaw)
                * Quaternion::from_angle_x(-pitch)
                * vec3(0.0, 0.0, orbit_dist)
                + cam_offset;

            // Mat4::look_at requires a Point3
            let camera_position = Point3::from_vec(camera_position);

            scene.camera.camera_to_world = Mat4::look_at(
                camera_position,
                Point3::from_vec(cam_offset),
                vec3(0.0, 1.0, 0.0),
            )
            .inverse_transform()
            .unwrap();

            let cache = build_cache(scene);

            canvas.clear();

            texture
                .with_lock(None, |buffer: &mut [u8], _pitch: usize| {
                    render(scene, &cache, render_settings, &mut render_buffer);
                    buffer.copy_from_slice(&*render_buffer);
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

    let mut args_iter = std::env::args();
    let scene_file = args_iter
        .position(|arg| arg == "--scene")
        .and_then(|_| args_iter.next());

    if scene_file.is_none() {
        eprintln!("No scene file specified. Please specify a scene file using --scene");
        std::process::exit(-1);
    }

    let mut scene = load_scene(Path::new(scene_file.unwrap().as_str()))
        .expect("failed to load specified scene file");

    let mut render_settings = RenderSettings {
        resolution: (640, 480),
        background_color: vec3(0.1, 0.1, 0.1),
        debug_coord: single_debug_coord,
        max_ray_bounces: 4,
    };

    let interactive_mode = std::env::args().any(|arg| arg == "--interactive");

    let cache = build_cache(&scene);

    if interactive_mode {
        interactive_loop(&mut scene, &mut render_settings);
    } else if let Some(coord) = single_debug_coord {
        let color = render_one_pixel(coord, &scene, &cache, &render_settings);
        println!("Pixel color is {:?}", color);
    } else {
        let mut imgbuf =
            ImageBuffer::new(render_settings.resolution.0, render_settings.resolution.1);
        render(&scene, &cache, &render_settings, &mut imgbuf);
        imgbuf.save("image.png").unwrap();
    }
}
