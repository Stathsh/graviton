// Final composite pass — combine scene with bloom + vignette + tone mapping

struct CompositeParams {
  bloom_intensity: f32,
  vignette_strength: f32,
  exposure: f32,
  _pad: f32,
}

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> params: CompositeParams;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VSOut {
  let verts = array<vec2<f32>, 3>(
    vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0),
  );
  var out: VSOut;
  out.pos = vec4(verts[i], 0.0, 1.0);
  out.uv = verts[i] * 0.5 + 0.5;
  return out;
}

// ACES filmic tone mapping
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let scene = textureSample(scene_tex, tex_sampler, in.uv).rgb;
  let bloom = textureSample(bloom_tex, tex_sampler, in.uv).rgb;

  // Combine scene + bloom
  var color = scene + bloom * params.bloom_intensity;

  // Exposure
  color *= params.exposure;

  // ACES tone mapping (cinematic look)
  color = aces_tonemap(color);

  // Vignette
  let uv_centered = in.uv * 2.0 - 1.0;
  let vignette_dist = length(uv_centered);
  let vignette = 1.0 - smoothstep(0.5, 1.5, vignette_dist) * params.vignette_strength;
  color *= vignette;

  // Subtle film grain
  let grain = fract(sin(dot(in.uv * 1000.0, vec2(12.9898, 78.233))) * 43758.5453) * 0.015 - 0.0075;
  color += grain;

  return vec4(color, 1.0);
}
