// Final composite — ACES tonemapping + vignette + subtle glow
// Reads HDR scene texture, outputs to screen

struct Params {
  exposure: f32,
  vignette: f32,
  glow_amount: f32,
  time: f32,
}

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> params: Params;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VSOut {
  let v = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var out: VSOut;
  out.pos = vec4(v[i], 0.0, 1.0);
  out.uv = v[i] * 0.5 + 0.5;
  return out;
}

fn aces(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let dims = vec2<f32>(textureDimensions(scene));
  let texel = 1.0 / dims;

  // Center sample
  var color = textureSample(scene, samp, in.uv).rgb;

  // Cheap bloom: sample surrounding area and add glow
  let glow_radius = 3.0;
  var glow = vec3(0.0);
  glow += textureSample(scene, samp, in.uv + vec2(texel.x * glow_radius, 0.0)).rgb;
  glow += textureSample(scene, samp, in.uv - vec2(texel.x * glow_radius, 0.0)).rgb;
  glow += textureSample(scene, samp, in.uv + vec2(0.0, texel.y * glow_radius)).rgb;
  glow += textureSample(scene, samp, in.uv - vec2(0.0, texel.y * glow_radius)).rgb;
  glow += textureSample(scene, samp, in.uv + texel * glow_radius * 0.7).rgb;
  glow += textureSample(scene, samp, in.uv - texel * glow_radius * 0.7).rgb;
  glow += textureSample(scene, samp, in.uv + vec2(texel.x, -texel.y) * glow_radius * 0.7).rgb;
  glow += textureSample(scene, samp, in.uv + vec2(-texel.x, texel.y) * glow_radius * 0.7).rgb;
  glow /= 8.0;

  // Wider glow pass
  let glow_radius2 = 8.0;
  var glow2 = vec3(0.0);
  glow2 += textureSample(scene, samp, in.uv + vec2(texel.x * glow_radius2, 0.0)).rgb;
  glow2 += textureSample(scene, samp, in.uv - vec2(texel.x * glow_radius2, 0.0)).rgb;
  glow2 += textureSample(scene, samp, in.uv + vec2(0.0, texel.y * glow_radius2)).rgb;
  glow2 += textureSample(scene, samp, in.uv - vec2(0.0, texel.y * glow_radius2)).rgb;
  glow2 /= 4.0;

  // Even wider
  let glow_radius3 = 20.0;
  var glow3 = vec3(0.0);
  glow3 += textureSample(scene, samp, in.uv + vec2(texel.x * glow_radius3, 0.0)).rgb;
  glow3 += textureSample(scene, samp, in.uv - vec2(texel.x * glow_radius3, 0.0)).rgb;
  glow3 += textureSample(scene, samp, in.uv + vec2(0.0, texel.y * glow_radius3)).rgb;
  glow3 += textureSample(scene, samp, in.uv - vec2(0.0, texel.y * glow_radius3)).rgb;
  glow3 /= 4.0;

  // Only bloom bright areas (threshold)
  let lum = dot(glow, vec3(0.2126, 0.7152, 0.0722));
  let lum2 = dot(glow2, vec3(0.2126, 0.7152, 0.0722));
  let lum3 = dot(glow3, vec3(0.2126, 0.7152, 0.0722));
  let bloom = glow * smoothstep(0.3, 1.0, lum) * 0.4
            + glow2 * smoothstep(0.2, 0.8, lum2) * 0.25
            + glow3 * smoothstep(0.15, 0.6, lum3) * 0.15;

  color += bloom * params.glow_amount;

  // Exposure
  color *= params.exposure;

  // ACES filmic tonemapping
  color = aces(color);

  // Vignette — cinematic darkening at edges
  let uv_c = in.uv * 2.0 - 1.0;
  let vig_d = length(uv_c);
  let vignette = 1.0 - smoothstep(0.4, 1.6, vig_d) * params.vignette;
  color *= vignette;

  // Very subtle warm color grade
  color = mix(color, color * vec3(1.05, 0.98, 0.92), 0.15);

  // Film grain
  let grain = fract(sin(dot(in.uv * 1000.0 + params.time, vec2(12.9898, 78.233))) * 43758.5) * 0.012 - 0.006;
  color += grain;

  return vec4(color, 1.0);
}
