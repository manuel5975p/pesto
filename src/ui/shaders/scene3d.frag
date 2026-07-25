#version 450
// Two-sided hemispheric lambert plus a fixed key light. Two-sided because the
// cut-cell boxes and slice quads are viewed from both faces and back faces
// must not go black.

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 lightDir;
    vec4 tint;
} pc;

layout(location = 0) in vec3 vNrm;
layout(location = 1) in vec4 vCol;

layout(location = 0) out vec4 outCol;

void main()
{
    vec3 n = normalize(vNrm);
    if (dot(n, n) < 0.5) n = vec3(0.0, 0.0, 1.0);
    float key = abs(dot(n, normalize(pc.lightDir.xyz)));
    // Sky/ground hemispheric term keeps unlit faces readable on a dark
    // background instead of collapsing to the clear colour.
    float sky = 0.5 + 0.5 * n.z;
    float lit = 0.30 + 0.45 * key + 0.25 * sky;
    outCol = vec4(vCol.rgb * lit, vCol.a);
}
