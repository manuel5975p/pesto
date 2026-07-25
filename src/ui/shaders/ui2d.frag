#version 450

layout(set = 0, binding = 0) uniform sampler2D glyphAtlas;

layout(location = 0) in vec2 vUv;
layout(location = 1) in vec4 vCol;

layout(location = 0) out vec4 outCol;

void main()
{
    // The atlas holds coverage only (R8): the glyph's colour is the vertex
    // colour, so one atlas serves every text colour in the UI.
    float cov = vUv.x < 0.0 ? 1.0 : texture(glyphAtlas, vUv).r;
    outCol = vec4(vCol.rgb, vCol.a * cov);
}
