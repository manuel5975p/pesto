#version 450
// Grid lines, domain box, axes. Unlit: a mesh line must read as exactly its
// assigned colour, or "is this row fine or coarse?" becomes a lighting
// question.

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 lightDir;
    vec4 tint;
} pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in uvec4 inCol;

layout(location = 0) out vec4 vCol;

void main()
{
    gl_Position = pc.mvp * vec4(inPos, 1.0);
    vCol = vec4(inCol) * (1.0 / 255.0) * pc.tint;
}
