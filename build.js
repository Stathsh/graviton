const esbuild = require('esbuild');
const fs = require('fs');
const { execSync } = require('child_process');

const wgslPlugin = {
  name: 'wgsl-loader',
  setup(build) {
    build.onLoad({ filter: /\.wgsl$/ }, async (args) => {
      const code = await fs.promises.readFile(args.path, 'utf8');
      return { contents: `export default ${JSON.stringify(code)};`, loader: 'ts' };
    });
  },
};

async function build() {
  // 1. Compile Zig to WASM
  console.log('Compiling Zig → WASM...');
  try {
    execSync('cd zig && zig build', { stdio: 'inherit' });
  } catch {
    console.log('Zig not available, using pre-built WASM');
  }

  // 2. Bundle TypeScript
  console.log('Bundling TypeScript...');
  await esbuild.build({
    entryPoints: ['src/main.ts'],
    bundle: true,
    outfile: 'dist/app.js',
    format: 'esm',
    minify: process.env.NODE_ENV === 'production',
    sourcemap: true,
    target: 'es2022',
    plugins: [wgslPlugin],
  });

  // 3. Copy static assets
  fs.cpSync('public', 'dist', { recursive: true });

  // 4. Copy WASM binary
  fs.mkdirSync('dist/wasm', { recursive: true });
  fs.copyFileSync('zig/zig-out/bin/kerr.wasm', 'dist/wasm/kerr.wasm');

  console.log('Build complete → dist/');
}

build().catch(e => { console.error(e); process.exit(1); });
