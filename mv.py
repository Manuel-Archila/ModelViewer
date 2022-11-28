import numpy
import random
import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import *
import glm
from Obj import *

pygame.init()

screen = pygame.display.set_mode(
    (800, 600),
    pygame.OPENGL | pygame.DOUBLEBUF
)
# dT = pygame.time.Clock()



vertex_shader = """
#version 460
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 vertexColor;
uniform mat4 amatrix;
out vec3 ourColor;
out vec2 fragCoord;
void main()
{
    gl_Position = amatrix * vec4(position, 1.0f);
    fragCoord =  gl_Position.xy;
    ourColor = vertexColor;
}
"""

fragment_shader_1 = """
#version 460

layout (location = 0) out vec4 fragColor;

uniform vec3 color;


in vec3 ourColor;

void main()
{
    fragColor = vec4(color, 1.0f);
}
"""



fragment_shader_2 = """
#version 460
#define PI 3.14159265358979
layout (location = 0) out vec4 fragColor;

uniform vec3 color;
uniform float iTime;
in vec2 fragCoord;
in vec3 ourColor;


vec3 a = vec3(0.5, 0.5, 0.5);
vec3 b = vec3(0.5, 0.5, 0.5);
vec3 c = vec3(1.0, 1.0, 1.0);
vec3 d = vec3(0.00, 0.33, 0.67);

// iq color mapper
vec3 colorMap(float t) {
	return (a + b * cos(2. * PI * (c * t + d)));
}

void main()
{
    vec2 iResolution = vec2(1, 1);
    vec2 uv = fragCoord / iResolution.xy;
    uv -= 0.5;
    uv.x *= iResolution.x / iResolution.y;
    
    float r = length(uv);
    float a = atan(uv.y, uv.x);
    
    float ring = 1.5 + 0.8 * sin(PI * 0.25 * iTime);
    
    float kr = 0.5 - 0.5 * cos(7. * PI * r); 
    vec3 kq = 0.5 - 0.5 * sin(ring*vec3(30., 29.3, 28.6) * r - 6.0 * iTime + PI * vec3(-0.05, 0.5, 1.0));
    vec3 c = kr * (0.1 + kq * (1. - 0.5* colorMap(a / PI))) * (0.5 + 0.5 * sin(11.*a + 22.5*r));

    // Output to screen
    fragColor.rgb = mix(vec3(0.0, 0.0, 0.2), c, 0.85);
}
"""
fragment_shader_3 = """
#version 460

layout (location = 0) out vec4 fragColor;
uniform vec3 color;
uniform float iTime;
in vec2 fragCoord;
in vec3 ourColor;

vec3 JuliaFractal(vec2 c, vec2 c2, float animparam, float anim2 ) {	
	vec2 z = c;
    
	float ci = 0.0;
	float mean = 0.0;
    
	for(int i = 0;i < 64; i++)
    {
		vec2 a = vec2(z.x,abs(z.y));
		
        float b = atan(a.y*(0.99+animparam*9.0), a.x+.110765432+animparam);
		
        if(b > 0.0) b -= 6.303431307+(animparam*3.1513);
		
        z = vec2(log(length(a*(0.98899-(animparam*2.70*anim2)))),b) + c2;

        if (i>0) mean+=length(z/a*b);

        mean+=a.x-(b*77.0/length(a*b));

        mean = clamp(mean, 111.0, 99999.0);
	}
    
	mean/=131.21;
	ci =  1.0 - fract(log2(.5*log2(mean/(0.57891895-abs(animparam*141.0)))));

	return vec3( .5+.5*cos(6.*ci+0.0),.5+.75*cos(6.*ci + 0.14),.5+.5*cos(6.*ci +0.7) );
}


void main()
{
    vec2 iResolution = vec2(1, 1);
    float animWings = 0.004 * cos(iTime*0.5);
    float animFlap = 0.011 * sin(iTime*1.0);    
    float timeVal = 56.48-20.1601;
	vec2 uv = fragCoord.xy - iResolution.xy*.5;
	uv /= iResolution.x*1.5113*abs(sin(timeVal));
    uv.y -= animWings*5.0; 
	vec2 tuv = uv*125.0;
	float rot=3.141592654*0.5;
  
	uv.x = tuv.x*cos(rot)-tuv.y*sin(rot);
	uv.y =1.05* tuv.x*sin(rot)+tuv.y*cos(rot);
	float juliax = tan(timeVal) * 0.011 + 0.02/(fragCoord.y*0.19531*(1.0-animFlap));
	float juliay = cos(timeVal * 0.213) * (0.022+animFlap) + 5.66752-(juliax*1.5101);//+(fragCoord.y*0.0001);// or 5.7
    
 
    float tapU = (1.0/ float(iResolution.x))*25.5;//*cos(animFlap);
    float tapV = (1.0/ float(iResolution.y))*25.5;//*cos(animFlap);
    
  
	fragColor = vec4( JuliaFractal(uv+vec2(0.0,0.0), vec2(juliax, juliay), animWings, animFlap ) ,1.0);
    
    fragColor += vec4( JuliaFractal(uv+vec2(tapU,tapV), vec2(juliax, juliay), animWings, animFlap ) ,1.0);
//    fragColor += vec4( JuliaFractal(uv+vec2(tapU,-tapV), vec2(juliax, juliay), animWings, animFlap ) ,1.0);
//    fragColor += vec4( JuliaFractal(uv+vec2(-tapU,tapV), vec2(juliax, juliay), animWings, animFlap ) ,1.0);
    fragColor += vec4( JuliaFractal(uv+vec2(-tapU,-tapV), vec2(juliax, juliay), animWings, animFlap ) ,1.0);  
    fragColor *= 0.3333;
    
    fragColor.xyz = fragColor.zyx;
	fragColor.xyz = vec3(1)-fragColor.xyz;

}
"""

fragment_shader_4 = """
#version 460
layout (location = 0) out vec4 fragColor;
const float pi = 3.14159;

uniform vec3 color;
uniform float iTime;
in vec2 fragCoord;
in vec3 ourColor;

float sigmoid(float x){
 	return x/(1.+abs(x));   
}

float iter(vec2 p, vec4 a, vec4 wt, vec4 ws, float t, float m, float stereo){
    float wp = .2;
    vec4 phase = vec4(mod(t, wp), mod(t+wp*.25, wp), mod(t+wp*.5, wp), mod(t+wp*.75, wp))/wp;
    float zoom = 1./(1.+.5*(p.x*p.x+p.y*p.y));
    vec4 scale = zoom*pow(vec4(2.), -4.*phase);
    vec4 ms = .5-.5*cos(2.*pi*phase);
    vec4 pan = stereo/scale*(1.-phase)*(1.-phase);
    vec4 v = ms*sin( wt*(t+m) + (m+ws*scale)*((p.x+pan) * cos((t+m)*a) + p.y * sin((t+m)*a)));
    return sigmoid(v.x+v.y+v.z+v.w+m);
}

vec3 scene(float gt, vec2 uv, vec4 a0, vec4 wt0, vec4 ws0, float blur){
    //time modulation
    float tm = mod(.0411*gt, 1.);
    tm = sin(2.*pi*tm*tm);
    float t = (.04*gt + .05*tm);
    
    float stereo = 1.*(sigmoid(2.*(sin(1.325*t*cos(.5*t))+sin(-.7*t*sin(.77*t)))));//+sin(-17.*t)+sin(10.*t))));
    //t = 0.;
    //also apply spatial offset
    uv+= .5*sin(.33*t)*vec2(cos(t), sin(t));
    
    //wildly iterate and divide
    float p0 = iter(uv, a0, wt0, ws0, t, 0., stereo);
    
   	float p1 = iter(uv, a0, wt0, ws0, t, p0, stereo);
    
    float p2 = sigmoid(p0/(p1+blur));
    
    float p3 = iter(uv, a0, wt0, ws0, t, p2, stereo);
    
    float p4 = sigmoid(p3/(p2+blur));
    
    float p5 = iter(uv, a0, wt0, ws0, t, p4, stereo);
    
    float p6 = sigmoid(p4/(p5+blur));
    
    float p7 = iter(uv, a0, wt0, ws0, t, p6, stereo);
    
    float p8 = sigmoid(p4/(p2+blur));
    
    float p9 = sigmoid(p8/(p7+blur));
    
    float p10 = iter(uv, a0, wt0, ws0, t, p8, stereo);
    
    float p11 = iter(uv, a0, wt0, ws0, t, p9, stereo);
    
    float p12 = sigmoid(p11/(p10+blur));
    
    float p13 = iter(uv, a0, wt0, ws0, t, p12, stereo);
    
    //colors
    vec3 accent_color = vec3(1.,0.2,0.);//vec3(0.99,0.5,0.2);
    /*float r = sigmoid(-1.+2.*p0+p1-max(1.*p3,0.)+p5+p7+p10+p11+p13);
    float g = sigmoid(-1.+2.*p0-max(1.*p1,0.)-max(2.*p3,0.)-max(2.*p5,0.)+p7+p10+p11+p13);
    float b = sigmoid(0.+1.5*p0+p1+p3+-max(2.*p5,0.)+p7+p10+p11+p13);
    */
    float r = sigmoid(p0+p1+p5+p7+p10+p11+p13);
    float g = sigmoid(p0-p1+p3+p7+p10+p11);
    float b = sigmoid(p0+p1+p3+p5+p11+p13);
    
    
    vec3 c = max(vec3(0.), .4+.6*vec3(r,g,b));
    
    float eps = .4;
    float canary = min(abs(p1), abs(p2));
    canary = min(canary, abs(p5));
    //canary = min(canary, abs(p6));
    canary = min(canary, abs(p7));
    canary = min(canary, abs(p10));
    float m = max(0.,eps-canary)/eps;
    m = sigmoid((m-.5)*700./(1.+10.*blur))*.5+.5;
    //m = m*m*m*m*m*m*m*m*m*m;
    vec3 m3 = m*(1.-accent_color);
    c *= .8*(1.-m3)+.3;//mix(c, vec3(0.), m);
    
    return c;
}

void main()
{
    vec2 iResolution = vec2(1, 1);
    float s = min(iResolution.x, iResolution.y);
   	vec2 uv = (2.*fragCoord.xy - vec2(iResolution.xy)) / s;
    
    float blur = .5*(uv.x*uv.x+uv.y*uv.y);
    
    //angular, spatial and temporal frequencies
    vec4 a0 = pi*vec4(.1, -.11, .111, -.1111); 
    vec4 wt0 = 2.*pi*vec4(.3);//.3333, .333, .33, .3);
    vec4 ws0 = 2.5*vec4(11., 13., 11., 5.);

    //aa and motion blur
    float mb = 1.;
    float t = 1100.+iTime;
    vec3 c = scene(t, uv, a0, wt0, ws0, blur)
        + scene(t-mb*.00185, uv+(1.+blur)*vec2(.66/s, 0.), a0, wt0, ws0, blur)
        + scene(t-mb*.00370, uv+(1.+blur)*vec2(-.66/s, 0.), a0, wt0, ws0, blur)
        + scene(t-mb*.00555, uv+(1.+blur)*vec2(0., .66/s), a0, wt0, ws0, blur)
        + scene(t-mb*.00741, uv+(1.+blur)*vec2(0., -.66/s), a0, wt0, ws0, blur)
        + scene(t-mb*.00926, uv+(1.+blur)*vec2(.5/s, .5/s), a0, wt0, ws0, blur)
        + scene(t-mb*.01111, uv+(1.+blur)*vec2(-.5/s, .5/s), a0, wt0, ws0, blur)
        + scene(t-mb*.01296, uv+(1.+blur)*vec2(-.5/s, -.5/s), a0, wt0, ws0, blur)
        + scene(t-mb*.01481, uv+(1.+blur)*vec2(.5/s, -.5/s), a0, wt0, ws0, blur)

        ;
    c/=9.;
    
    fragColor = vec4(c,1.0);
}

"""
compiled_vertex_shader = compileShader(vertex_shader, GL_VERTEX_SHADER)
compiled_fragment_shader = compileShader(fragment_shader_1, GL_FRAGMENT_SHADER)
shader = compileProgram(
    compiled_vertex_shader,
    compiled_fragment_shader
)

compiled_fragment_shader2 = compileShader(fragment_shader_2, GL_FRAGMENT_SHADER)
shader2 = compileProgram(
    compiled_vertex_shader,
    compiled_fragment_shader2
)

compiled_fragment_shader3 = compileShader(fragment_shader_3, GL_FRAGMENT_SHADER)
shader3 = compileProgram(
    compiled_vertex_shader,
    compiled_fragment_shader3
)

compiled_fragment_shader4 = compileShader(fragment_shader_4, GL_FRAGMENT_SHADER)
shader4 = compileProgram(
    compiled_vertex_shader,
    compiled_fragment_shader4
)

shaders = {1: shader2, 2: shader3, 3: shader4}

vertices = []
modelo = Obj('./model.obj')
for element in modelo.vertices:
    for vertex in element:
        vertices.append(vertex)
vertex_data = numpy.array(vertices, dtype=numpy.float32)
vertex_array_object = glGenVertexArrays(1)
glBindVertexArray(vertex_array_object)

vertex_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
glBufferData(
    GL_ARRAY_BUFFER,  # tipo de datos
    vertex_data.nbytes,  # tamaño de da data en bytes    
    vertex_data, # puntero a la data
    GL_STATIC_DRAW
)

glVertexAttribPointer(
    0,
    3,
    GL_FLOAT,
    GL_FALSE,
    3 * 4,
    ctypes.c_void_p(0)
)
glEnableVertexAttribArray(0)

caras = []
for face in modelo.faces:
    f1 = face[0][0] - 1
    f2 = face[1][0] - 1
    f3 = face[2][0] - 1
    caras.append(f1)
    caras.append(f2)
    caras.append(f3)

caras = numpy.array(caras, dtype=numpy.int32)
faces_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faces_buffer_object)
glBufferData(
    GL_ELEMENT_ARRAY_BUFFER,
    caras.nbytes,
    caras,
    GL_STATIC_DRAW)






def calculateMatrix(angle, movement):
    i = glm.mat4(1)
    translate = glm.translate(i, glm.vec3(0, 0, 0))
    rotate = glm.rotate(i, glm.radians(angle), movement)
    scale = glm.scale(i, glm.vec3(1, 1, 1))

    model = translate * rotate * scale

    view = glm.lookAt(
        glm.vec3(0, 0, 5),
        glm.vec3(0, 0, 0),
        glm.vec3(0, 1, 0)
    )

    projection = glm.perspective(
        glm.radians(45),
        800/600,
        0.1,
        1000.0
    )

    amatrix = projection * view * model

    glUniformMatrix4fv(
        glGetUniformLocation(shader, 'amatrix'),
        1,
        GL_FALSE,
        glm.value_ptr(amatrix)
    )

glViewport(0, 0, 800, 600)



active_shader = shaders[random.randint(1, 3)]
glUseProgram(active_shader)
running = True

glClearColor(0.5, 1.0, 0.5, 1.0)

r = 0
change = False
shader_key = 0
movement = glm.vec3(0, 0, 1)
while running:
    glClear(GL_COLOR_BUFFER_BIT)

    glUniform1f(glGetUniformLocation(active_shader,'iTime'), pygame.time.get_ticks() / 1000)

    color1 = random.random()
    color2 = random.random()
    color3 = random.random()

    color = glm.vec3(color1, color2, color3)

    if change == True:
        active_shader = shaders[shader_key]
        glUseProgram(active_shader)
        change = False


    glUniform3fv(
        glGetUniformLocation(active_shader,'color'),
        1,
        glm.value_ptr(color)
    )

    calculateMatrix(r, movement)

    pygame.time.wait(50)


    glDrawElements(GL_TRIANGLES, len(caras), GL_UNSIGNED_INT, None)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        camera_keys = pygame.key.get_pressed()
        if camera_keys[pygame.K_LEFT]:
            movement = glm.vec3(0, 1, 0)
            r -= 10
        if camera_keys[pygame.K_RIGHT]:
            movement = glm.vec3(0, 1, 0)
            r += 10
        if camera_keys[pygame.K_UP]:
            movement = glm.vec3(1, 0, 0)
            r += 10
        if camera_keys[pygame.K_DOWN]:
            r -= 10
            movement = glm.vec3(1, 0, 0)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                shader_key = 1
                change = True
            if event.key == pygame.K_2:
                shader_key = 2
                change = True
            if event.key == pygame.K_3:
                shader_key = 3
                change = True
            