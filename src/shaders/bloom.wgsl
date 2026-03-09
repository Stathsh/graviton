// Post-processing bloom — dual Kawase blur + additive composite
// Makes hot accretion disk and jets glow cinematically

struct BloomParams {
  texel_size: vec2<f32>,  // 1/width, 1/height
  intensity: f32,
  threshold: f32,
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: BloomParams;

// Extract bright pixels (threshold pass)
@compute @workgroup_size(8, 8)
fn threshold_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(input_tex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let color = textureLoad(input_tex, vec2<i32>(gid.xy), 0);
  let luminance = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));

  let soft_threshold = params.threshold * 0.8;
  let contribution = smoothstep(soft_threshold, params.threshold * 1.5, luminance);
  let bright = color.rgb * contribution;

  textureStore(output_tex, vec2<i32>(gid.xy), vec4(bright, 1.0));
}

// Kawase blur (downsample) — better quality than gaussian for bloom
@compute @workgroup_size(8, 8)
fn blur_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(input_tex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let uv = vec2<i32>(gid.xy);

  // 9-tap tent filter
  var color = textureLoad(input_tex, uv, 0).rgb * 4.0;
  color += textureLoad(input_tex, uv + vec2(1, 0), 0).rgb * 2.0;
  color += textureLoad(input_tex, uv + vec2(-1, 0), 0).rgb * 2.0;
  color += textureLoad(input_tex, uv + vec2(0, 1), 0).rgb * 2.0;
  color += textureLoad(input_tex, uv + vec2(0, -1), 0).rgb * 2.0;
  color += textureLoad(input_tex, uv + vec2(1, 1), 0).rgb;
  color += textureLoad(input_tex, uv + vec2(-1, 1), 0).rgb;
  color += textureLoad(input_tex, uv + vec2(1, -1), 0).rgb;
  color += textureLoad(input_tex, uv + vec2(-1, -1), 0).rgb;
  color /= 16.0;

  textureStore(output_tex, uv, vec4(color, 1.0));
}

// Composite: read bloom texture, blend with scene
// This is done as a full-screen render pass instead
