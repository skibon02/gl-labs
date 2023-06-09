use std::ffi::{CStr, CString};
use std::num::NonZeroU32;
use std::ops::Deref;

use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::window::WindowBuilder;

use raw_window_handle::HasRawWindowHandle;

use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::SwapInterval;

use glutin_winit::{self, DisplayBuilder, GlWindow};

pub mod gl {
    #![allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));
}

pub fn main(event_loop: winit::event_loop::EventLoop<()>) {
    let window_builder = Some(WindowBuilder::new().with_inner_size(LogicalSize::new(800, 600)).with_transparent(true));

    let template = ConfigTemplateBuilder::new().with_alpha_size(8);

    let display_builder = DisplayBuilder::new().with_window_builder(window_builder);

    let (mut window, gl_config) = display_builder.build(&event_loop, template, |configs| {
            configs.reduce(|accum, config| {
                let transparency_check = config.supports_transparency().unwrap_or(false)
                    & !accum.supports_transparency().unwrap_or(false);

                if transparency_check || config.num_samples() > accum.num_samples() {
                    config
                } else {
                    accum
                }
            }).unwrap()
        }).unwrap();

    println!("Picked a config with {} samples", gl_config.num_samples());

    let raw_window_handle = window.as_ref().map(|window| window.raw_window_handle());

    // XXX The display could be obtained from the any object created by it, so we
    // can query it from the config.
    let gl_display = gl_config.display();

    // The context creation part. It can be created before surface and that's how
    // it's expected in multithreaded + multiwindow operation mode, since you
    // can send NotCurrentContext, but not Surface.
    let context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(Some(Version::new(4,6))))
        .build(raw_window_handle);

    let mut not_current_gl_context = Some(unsafe {
        gl_display.create_context(&gl_config, &context_attributes).expect("Failed to create context!")
    });

    let mut state = None;
    let mut renderer = None;
    event_loop.run(move |event, window_target, control_flow| {
        control_flow.set_poll();

        // println!("{:?}", event);
        
        match event {
            Event::Resumed => {
                let window = window.take().unwrap_or_else(|| {
                    let window_builder = WindowBuilder::new().with_transparent(true);
                    glutin_winit::finalize_window(window_target, window_builder, &gl_config)
                        .unwrap()
                });

                let attrs = window.build_surface_attributes(<_>::default());
                let gl_surface = unsafe {
                    gl_config.display().create_window_surface(&gl_config, &attrs).unwrap()
                };

                // Make it current.
                let gl_context =
                    not_current_gl_context.take().unwrap().make_current(&gl_surface).unwrap();

                // The context needs to be current for the Renderer to set up shaders and
                // buffers. It also performs function loading, which needs a current context on
                // WGL.
                renderer.get_or_insert_with(|| Renderer::new(&gl_display));

                // Try setting vsync.
                if let Err(res) = gl_surface
                    .set_swap_interval(&gl_context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()))
                {
                    eprintln!("Error setting vsync: {res:?}");
                }

                assert!(state.replace((gl_context, gl_surface, window)).is_none());
            },
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(size) => {
                    if size.width != 0 && size.height != 0 {
                        // Some platforms like EGL require resizing GL surface to update the size
                        // Notable platforms here are Wayland and macOS, other don't require it
                        // and the function is no-op, but it's wise to resize it for portability
                        // reasons.
                        if let Some((gl_context, gl_surface, _)) = &state {
                            gl_surface.resize(
                                gl_context,
                                NonZeroU32::new(size.width).unwrap(),
                                NonZeroU32::new(size.height).unwrap(),
                            );
                            let renderer = renderer.as_ref().unwrap();
                            renderer.resize(size.width as i32, size.height as i32);
                        }
                    }
                },
                WindowEvent::CloseRequested => {
                    control_flow.set_exit();
                },
                _ => (),
            },
            Event::RedrawEventsCleared => {
                if let Some((gl_context, gl_surface, window)) = &state {
                    let renderer = renderer.as_ref().unwrap();
                    renderer.draw();
                    window.request_redraw();

                    gl_surface.swap_buffers(gl_context).unwrap();
                }
            },
            _ => (),
        }
    })
}

pub struct Renderer {
    program: gl::types::GLuint,
    vao: gl::types::GLuint,
    vbo: gl::types::GLuint,
    gl: gl::Gl,

    resolution_location: gl::types::GLint,
    time_location: gl::types::GLint,

    start_time: std::time::Instant,


}

impl Renderer {
    pub fn new<D: GlDisplay>(gl_display: &D) -> Self {
        unsafe {
            let gl = gl::Gl::load_with(|symbol| {
                let symbol = CString::new(symbol).unwrap();
                gl_display.get_proc_address(symbol.as_c_str()).cast()
            });

            if let Some(renderer) = get_gl_string(&gl, gl::RENDERER) {
                println!("Running on {}", renderer.to_string_lossy());
            }
            if let Some(version) = get_gl_string(&gl, gl::VERSION) {
                println!("OpenGL Version {}", version.to_string_lossy());
            }

            if let Some(shaders_version) = get_gl_string(&gl, gl::SHADING_LANGUAGE_VERSION) {
                println!("Shaders version on {}", shaders_version.to_string_lossy());
            }

            let vertex_shader = create_shader(&gl, gl::VERTEX_SHADER, VERTEX_SHADER_SOURCE);
            let fragment_shader = create_shader(&gl, gl::FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE);

            //check errors
            Renderer::check_shader_errors(&gl, [vertex_shader, fragment_shader]);

            let program = gl.CreateProgram();

            gl.AttachShader(program, vertex_shader);
            gl.AttachShader(program, fragment_shader);

            gl.LinkProgram(program);

            gl.UseProgram(program);

            gl.DeleteShader(vertex_shader);
            gl.DeleteShader(fragment_shader);

            let mut vao = std::mem::zeroed();
            gl.GenVertexArrays(1, &mut vao);
            gl.BindVertexArray(vao);

            let mut vbo = std::mem::zeroed();
            gl.GenBuffers(1, &mut vbo);
            gl.BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl.BufferData(
                gl::ARRAY_BUFFER,
                (VERTEX_DATA.len() * std::mem::size_of::<f32>()) as gl::types::GLsizeiptr,
                VERTEX_DATA.as_ptr() as *const _,
                gl::STATIC_DRAW,
            );

            let pos_attrib = gl.GetAttribLocation(program, b"position\0".as_ptr() as *const _);
            gl.VertexAttribPointer(
                pos_attrib as gl::types::GLuint,
                2,
                gl::FLOAT,
                0,
                2 * std::mem::size_of::<f32>() as gl::types::GLsizei,
                std::ptr::null(),
            );
            gl.EnableVertexAttribArray(pos_attrib as gl::types::GLuint);

            let time_location = gl.GetUniformLocation(program, b"iTime\0".as_ptr() as *const _);
            let resolution_location =
                gl.GetUniformLocation(program, b"iResolution\0".as_ptr() as *const _);
            gl.Uniform2f(resolution_location, 800.0, 600.0);

            let start_time = std::time::Instant::now();
            Self { program, vao, vbo, gl, time_location, start_time, resolution_location }
        }
    }

    pub fn draw(&self) {
        unsafe {
            self.gl.UseProgram(self.program);

            self.gl.BindVertexArray(self.vao);
            self.gl.BindBuffer(gl::ARRAY_BUFFER, self.vbo);

            let time = self.start_time.elapsed().as_secs_f32();
            self.gl.Uniform1f(self.time_location, time);

            self.gl.ClearColor(0.1, 0.1, 0.1, 0.9);
            self.gl.Clear(gl::COLOR_BUFFER_BIT);
            self.gl.DrawArrays(gl::TRIANGLES, 0, 6);
        }
    }

    pub fn resize(&self, width: i32, height: i32) {
        unsafe {
            self.gl.Viewport(0, 0, width, height);
            self.gl
                .Uniform2f(self.resolution_location, width as f32, height as f32);
        }
    }
    pub fn check_shader_errors( gl: &gl::Gl, [vertex_shader, fragment_shader]: [gl::types::GLuint; 2]) {
        let mut success = gl::FALSE as gl::types::GLint;


        unsafe { gl.GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut success) };
        if success != gl::TRUE as gl::types::GLint {
            let mut info_log = [0u8; 512];
            let mut len = 0;
            unsafe { gl.GetShaderInfoLog(vertex_shader, 512, &mut len, info_log.as_mut_ptr() as *mut _) };
            let info_log = std::str::from_utf8(&info_log[..len as usize]).unwrap();
            println!("Vertex shader compilation failed: {}", info_log);
        }

        unsafe { gl.GetShaderiv(fragment_shader, gl::COMPILE_STATUS, &mut success) };
        if success != gl::TRUE as gl::types::GLint {
            let mut info_log = [0u8; 512];
            let mut len = 0;
            unsafe { gl.GetShaderInfoLog(fragment_shader, 512, &mut len, info_log.as_mut_ptr() as *mut _) };
            let info_log = std::str::from_utf8(&info_log[..len as usize]).unwrap();
            println!("Fragment shader compilation failed: {}", info_log);
        }
    }
}

impl Deref for Renderer {
    type Target = gl::Gl;

    fn deref(&self) -> &Self::Target {
        &self.gl
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.gl.DeleteProgram(self.program);
            self.gl.DeleteBuffers(1, &self.vbo);
            self.gl.DeleteVertexArrays(1, &self.vao);
        }
    }
}

unsafe fn create_shader(
    gl: &gl::Gl,
    shader: gl::types::GLenum,
    source: &[u8],
) -> gl::types::GLuint {
    let shader = gl.CreateShader(shader);
    gl.ShaderSource(shader, 1, [source.as_ptr().cast()].as_ptr(), std::ptr::null());
    gl.CompileShader(shader);
    shader
}

fn get_gl_string(gl: &gl::Gl, variant: gl::types::GLenum) -> Option<&'static CStr> {
    unsafe {
        let s = gl.GetString(variant);
        (!s.is_null()).then(|| CStr::from_ptr(s.cast()))
    }
}

#[rustfmt::skip]
static VERTEX_DATA: [f32; 12] = [

    -1.0, -1.0,
    1.0, 1.0,
    -1.0, 1.0,

    -1.0, -1.0,
    1.0, -1.0,
    1.0, 1.0,
];

const VERTEX_SHADER_SOURCE: &[u8] = b"
#version 460 core

in vec2 position;

out vec2 fragCoord;

void main() {
gl_Position = vec4(position, 0.0, 1.0);
fragCoord = position;
}
\0";

const FRAGMENT_SHADER_SOURCE: &[u8] = b"
#version 460 core

in vec2 fragCoord;
out vec4 fragColor;

uniform vec2 iResolution;
uniform float iTime;

vec3 hsb2rgb(in vec3 c)
{
    vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),
                             6.0)-3.0)-1.0,
                     0.0,
                     1.0 );
    rgb = rgb*rgb*(3.0-2.0*rgb);
    return c.z * mix( vec3(1.0), rgb, c.y);
}

void main() {
    vec2 newCoord = (fragCoord + vec2(0.5, 0.5))*iResolution.xy;
    vec2 p = (2.0*newCoord-iResolution.xy)/iResolution.y;
    
    float r = length(p) * 0.9;
	vec3 color = hsb2rgb(vec3(iTime/13, 0.7, 0.4));
    
    float a = pow(r, 2.0);
    float b = sin(r * 0.8 - 1.6);
    float c = sin(r - 0.010);
    float s = sin(a - iTime * 3.0 + b) * c;
    
    color *= abs(1.0 / (s * 10.8)) - 0.01;
	fragColor = vec4(color, 1.);
}
\0";
