#version 450
// Solid geometry: obstacle surfaces, cut-cell boxes, slice quads.

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 lightDir;   // xyz world-space direction TO the light, w unused
    vec4 tint;       // multiplied into the vertex colour (w scales alpha)
} pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNrm;
layout(location = 2) in uvec4 inCol;   // R8G8B8A8_UNORM, expanded by the format

layout(location = 0) out vec3 vNrm;
layout(location = 1) out vec4 vCol;

void main()
{
    gl_Position = pc.mvp * vec4(inPos, 1.0);
    vNrm = inNrm;
    vCol = vec4(inCol) * (1.0 / 255.0) * pc.tint;
}
