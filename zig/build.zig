const std = @import("std");

pub fn build(b: *std.Build) void {
    const exe = b.addExecutable(.{
        .name = "kerr",
        .root_source_file = b.path("src/kerr.zig"),
        .target = b.resolveTargetQuery(.{
            .cpu_arch = .wasm32,
            .os_tag = .freestanding,
        }),
        .optimize = .ReleaseFast,
    });

    // Export all pub functions
    exe.entry = .disabled;
    exe.rdynamic = true;

    // Minimize binary size
    exe.root_module.strip = true;

    b.installArtifact(exe);
}
