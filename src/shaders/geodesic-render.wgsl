// Photon geodesic path visualization
// Renders traced photon paths as glowing lines

struct Uniforms {
  viewProj: mat4x4<f32>,
  color: vec4<f32>,     // xyz=color, w=alpha
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) alpha: f32,
  @location(1) glow: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
  let point = points[vid];
  var out: VSOut;
  out.pos = u.viewProj * vec4(point.xyz, 1.0);
  out.alpha = point.w; // fade along path
  out.glow = point.w;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let color = u.color.rgb * (1.0 + in.glow * 2.0);
  return vec4(color, in.alpha * u.color.w);
}
