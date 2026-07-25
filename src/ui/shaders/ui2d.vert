#version 450
// The whole 2D layer: panel fills, rounded-corner fans, and glyph quads out of
// the HarfBuzz-fed coverage atlas. Positions arrive in PIXELS with the origin
// at the top-left, which is the coordinate system every widget is laid out in.

layout(push_constant) uniform Push {
    vec2 invHalfSize;   // 2 / framebuffer extent
} pc;

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUv;    // negative = untextured (solid fill)
layout(location = 2) in uvec4 inCol;

layout(location = 0) out vec2 vUv;
layout(location = 1) out vec4 vCol;

void main()
{
    gl_Position = vec4(inPos * pc.invHalfSize - 1.0, 0.0, 1.0);
    vUv = inUv;
    vCol = vec4(inCol) * (1.0 / 255.0);
}
