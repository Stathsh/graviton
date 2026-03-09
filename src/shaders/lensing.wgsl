// Gravitational lensing + warped accretion disk + nebula starfield
// This shader does the heavy visual lifting — Interstellar-style black hole

struct Uniforms {
  invViewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,
  params: vec4<f32>,    // x=mass, y=spin, z=time, w=colorScheme
  metrics: vec4<f32>,   // x=r_horizon, y=r_photon, z=r_isco, w=0
}

@group(0) @binding(0) var<uniform> u: Uniforms;

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

// --- Noise functions for nebula ---
fn hash21(p: vec2<f32>) -> f32 {
  var p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn hash31(p: vec3<f32>) -> f32 {
  var p3 = fract(p * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}

fn noise3(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);

  return mix(
    mix(mix(hash31(i), hash31(i + vec3(1,0,0)), u.x),
        mix(hash31(i + vec3(0,1,0)), hash31(i + vec3(1,1,0)), u.x), u.y),
    mix(mix(hash31(i + vec3(0,0,1)), hash31(i + vec3(1,0,1)), u.x),
        mix(hash31(i + vec3(0,1,1)), hash31(i + vec3(1,1,1)), u.x), u.y),
    u.z
  );
}

fn fbm(p: vec3<f32>) -> f32 {
  var value = 0.0;
  var amplitude = 0.5;
  var pos = p;
  for (var i = 0; i < 5; i++) {
    value += amplitude * noise3(pos);
    pos *= 2.1;
    amplitude *= 0.5;
  }
  return value;
}

// --- Rich starfield with nebula ---
fn starfield(dir: vec3<f32>, time: f32) -> vec3<f32> {
  var color = vec3(0.0);

  // Layer 1: bright individual stars
  let p1 = dir * 120.0;
  let g1 = floor(p1);
  let f1 = fract(p1);
  let h1 = hash31(g1);
  let h2 = hash31(g1 + 7.0);
  let h3 = hash31(g1 + 13.0);
  let star_pos = vec3(h1, h2, h3);
  let d1 = length(f1 - star_pos);
  let twinkle = 0.75 + 0.25 * sin(time * (1.0 + h1 * 4.0) + h1 * 80.0);
  let brightness = pow(smoothstep(0.08, 0.0, d1), 1.5) * twinkle;

  // Star spectral types
  var star_color = vec3(1.0);
  if (h3 < 0.15) { star_color = vec3(0.55, 0.65, 1.0); }      // O/B hot blue
  else if (h3 < 0.3) { star_color = vec3(0.75, 0.82, 1.0); }   // A blue-white
  else if (h3 < 0.5) { star_color = vec3(1.0, 0.98, 0.9); }    // F white
  else if (h3 < 0.7) { star_color = vec3(1.0, 0.92, 0.7); }    // G yellow
  else if (h3 < 0.85) { star_color = vec3(1.0, 0.75, 0.45); }  // K orange
  else { star_color = vec3(1.0, 0.5, 0.3); }                    // M red
  color += star_color * brightness * (0.4 + h2 * 1.2);

  // Layer 2: dense dim star field
  let p2 = dir * 300.0;
  let g2 = floor(p2);
  let f2 = fract(p2);
  let d2 = length(f2 - vec3(hash31(g2), hash31(g2 + 5.0), hash31(g2 + 11.0)));
  let dim_star = pow(smoothstep(0.12, 0.0, d2), 2.0) * 0.25;
  color += vec3(0.85, 0.9, 1.0) * dim_star;

  // Layer 3: ultra-faint star dust
  let p3 = dir * 600.0;
  let g3 = floor(p3);
  let f3 = fract(p3);
  let d3 = length(f3 - vec3(hash31(g3 + 1.0), hash31(g3 + 3.0), hash31(g3 + 7.0)));
  color += vec3(0.7, 0.75, 0.9) * pow(smoothstep(0.15, 0.0, d3), 2.0) * 0.1;

  // Milky Way band — thick galactic plane
  let galactic_angle = asin(dir.y);
  let galactic_density = exp(-galactic_angle * galactic_angle * 6.0);
  let galactic_detail = fbm(dir * 8.0 + vec3(0.0, time * 0.01, 0.0));
  let galactic_color = mix(
    vec3(0.15, 0.08, 0.2),  // deep purple
    vec3(0.25, 0.18, 0.12), // warm dust
    galactic_detail,
  );
  color += galactic_color * galactic_density * galactic_detail * 0.5;

  // Nebula clouds — colored gas
  let nebula1 = fbm(dir * 3.0 + vec3(1.5, 0.3, 0.7));
  let nebula2 = fbm(dir * 4.5 + vec3(3.2, 1.1, 2.5));
  let nebula_mask = smoothstep(0.45, 0.7, nebula1) * smoothstep(0.4, 0.65, nebula2);
  let nebula_color = mix(
    vec3(0.3, 0.05, 0.15),  // deep red/magenta
    vec3(0.05, 0.1, 0.35),  // deep blue
    nebula2,
  );
  color += nebula_color * nebula_mask * 0.15;

  // Second nebula layer
  let nebula3 = fbm(dir * 2.0 + vec3(5.0, 2.0, 8.0));
  let nebula4 = fbm(dir * 6.0 + vec3(0.0, 3.0, 1.0));
  let nebula_mask2 = smoothstep(0.5, 0.75, nebula3) * smoothstep(0.35, 0.6, nebula4);
  color += vec3(0.1, 0.2, 0.05) * nebula_mask2 * 0.08; // green emission nebula

  return color;
}

// --- Accretion disk emission (for ray-disk intersection) ---
fn disk_emission(r: f32, phi: f32, time: f32) -> vec3<f32> {
  let isco = u.metrics.z;
  let mass = u.params.x;
  let rs = 2.0 * mass;

  if (r < isco * 0.8 || r > mass * 25.0) { return vec3(0.0); }

  // Radial temperature profile: T ~ r^(-3/4) for standard thin disk
  let r_ratio = isco / r;
  let temp_profile = pow(r_ratio, 0.75) * (1.0 - sqrt(isco / r));
  let temperature = 4000.0 + temp_profile * 50000.0;

  // Blackbody approximation
  let t = clamp(temperature / 1000.0, 1.0, 60.0);
  var cr: f32; var cg: f32; var cb: f32;
  if (t <= 6.6) { cr = 1.0; } else { cr = 1.29 * pow(t - 6.0, -0.133); }
  if (t <= 6.6) { cg = 0.39 * log(t) - 0.63; } else { cg = 1.13 * pow(t - 6.0, -0.076); }
  if (t >= 6.6) { cb = 1.0; } else if (t <= 1.9) { cb = 0.0; } else { cb = 0.543 * log(t - 1.0) - 1.196; }
  var bb = clamp(vec3(cr, cg, cb), vec3(0.0), vec3(1.0));

  // Color scheme
  let scheme = i32(u.params.w);
  if (scheme == 1) { bb *= vec3(1.2, 0.6, 1.4); }
  else if (scheme == 2) { bb = vec3(bb.b * 0.6, bb.g * 0.8 + bb.b * 0.3, bb.b * 1.5); }
  else if (scheme == 3) { bb *= vec3(1.4, 0.95, 0.5); } // warm interstellar

  // Spiral arm structure
  let spiral = sin(phi * 3.0 - r * 0.8 + time * 0.3) * 0.3 + 0.7;
  let spiral2 = sin(phi * 7.0 + r * 1.5 - time * 0.5) * 0.15 + 0.85;

  // Turbulent detail
  let turb = noise3(vec3(r * 2.0, phi * 5.0, time * 0.2));
  let detail = 0.7 + turb * 0.6;

  // Intensity: peaks near ISCO, falls off with r
  let radial_brightness = pow(r_ratio, 1.5) * exp(-(r - isco) * 0.02);
  let intensity = radial_brightness * spiral * spiral2 * detail * 3.0;

  return bb * max(intensity, 0.0);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let mass = u.params.x;
  let spin = u.params.y;
  let time = u.params.z;
  let rs = 2.0 * mass;
  let r_h = u.metrics.x;
  let r_ph = u.metrics.y;
  let isco = u.metrics.z;

  // Reconstruct ray
  let ndc = vec2(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0);
  let near = u.invViewProj * vec4(ndc, -1.0, 1.0);
  let far = u.invViewProj * vec4(ndc, 1.0, 1.0);
  let rayDir = normalize(far.xyz / far.w - near.xyz / near.w);
  let rayOrigin = u.cameraPos.xyz;

  // --- Ray-march through curved spacetime ---
  var pos = rayOrigin;
  var dir = rayDir;
  var absorbed = false;
  var closest = 999.0;
  var total_deflection = 0.0;
  var disk_color = vec3(0.0);
  var disk_hits = 0;
  var prev_y = pos.y;

  let steps = 256;

  for (var i = 0; i < steps; i++) {
    let to_bh = -pos;
    let r = length(to_bh);
    closest = min(closest, r);

    if (r < r_h * 0.85) {
      absorbed = true;
      break;
    }

    // --- Check disk plane crossing (y=0) ---
    if (prev_y * pos.y < 0.0 && disk_hits < 3) {
      // Interpolate exact crossing point
      let t_cross = prev_y / (prev_y - pos.y + 0.0001);
      let cross_r = length(mix(pos - dir * 0.3, pos, t_cross).xz);

      if (cross_r > isco * 0.8 && cross_r < mass * 25.0) {
        let cross_pos = mix(pos - dir * 0.3, pos, t_cross);
        let phi = atan2(cross_pos.z, cross_pos.x);

        // Doppler shift: approaching side brighter
        let orbital_dir = vec3(-sin(phi), 0.0, cos(phi));
        let doppler = 1.0 + dot(dir, orbital_dir) * 0.4;

        // Gravitational redshift
        let grav_shift = sqrt(max(1.0 - rs / cross_r, 0.01));

        var emission = disk_emission(cross_r, phi, time);
        emission *= doppler * doppler * doppler * doppler; // relativistic beaming D^4
        emission *= grav_shift;

        // Attenuation for secondary/tertiary images
        let atten = select(0.5, select(0.25, 1.0, disk_hits == 0), disk_hits == 1);
        disk_color += emission * atten;
        disk_hits++;
      }
    }
    prev_y = pos.y;

    // Gravitational deflection
    let r_hat = normalize(to_bh);
    let deflect_strength = rs / (r * r) * 1.5;

    // Frame-dragging
    let drag_dir = normalize(vec3(-pos.z, 0.0, pos.x));
    let drag_strength = spin * rs * rs / (r * r * r) * 0.4;

    dir = normalize(dir + r_hat * deflect_strength * 0.25 + drag_dir * drag_strength * 0.25);
    total_deflection += deflect_strength * 0.25;

    let step_size = select(0.4, select(0.15, 0.06, r < r_h * 2.5), r < r_h * 6.0);
    pos += dir * step_size;

    if (r > 150.0 && dot(dir, r_hat) < 0.0) { break; }
  }

  // --- Final color composition ---

  if (absorbed) {
    // Event horizon — deep black with subtle edge glow
    let rim = closest / r_h;
    let edge_glow = exp(-pow(rim - 1.0, 2.0) * 80.0) * 0.08;
    let hawking = vec3(edge_glow * 0.4, edge_glow * 0.2, edge_glow * 0.8);
    return vec4(hawking + disk_color * 0.3, 1.0);
  }

  // Lensed background
  var sky = starfield(normalize(dir), time);

  // Photon ring — bright thin ring at photon sphere
  let photon_dist = abs(closest - r_ph);
  let ring1 = exp(-photon_dist * photon_dist * 8.0) * 0.5;
  let ring2 = exp(-photon_dist * photon_dist * 0.8) * 0.15; // broader glow
  let ring_color = vec3(1.0, 0.7, 0.3) * ring1 + vec3(1.0, 0.5, 0.2) * ring2;

  // Einstein ring glow
  let er = rs * 2.6;
  let einstein_dist = abs(closest - er);
  let einstein_glow = exp(-einstein_dist * einstein_dist * 1.5) * 0.06;
  let einstein_color = vec3(0.6, 0.75, 1.0) * einstein_glow;

  // Inner shadow darkening
  let shadow_factor = smoothstep(r_h, r_h * 3.0, closest);

  // Gravitational reddening of background light
  let rshift = max(1.0 - total_deflection * 0.15, 0.0);

  var final_color = sky * rshift * shadow_factor + ring_color + einstein_color + disk_color;

  return vec4(final_color, 1.0);
}
