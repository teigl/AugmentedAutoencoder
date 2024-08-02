# -*- coding: UTF-8 -*-
from glfw.GLFW import *
from OpenGL.GL import *
from OpenGL.GL.NV.bindless_texture import *

class Window(object):

    def __init__(self, window_width, window_height, samples=1, window_title='', monitor=1, show_at_center=True, offscreen=False):
        self.window_title = window_title
        assert glfwInit(), 'Glfw Init failed!'
        glfwWindowHint(GLFW_SAMPLES, samples)
        if offscreen:
            glfwWindowHint(GLFW_VISIBLE, False);
        mon = glfwGetMonitors()[monitor] if monitor!=None else None
        self.windowID = glfwCreateWindow(window_width, window_height, self.window_title, mon)
        assert self.windowID, 'Could not create Window!'
        glfwMakeContextCurrent(self.windowID)

        if not glInitBindlessTextureNV():
            raise RuntimeError('Bindless Textures not supported')

        self.framebuf_width, self.framebuf_height = glfwGetFramebufferSize(self.windowID)
        self.framebuffer_size_callback = []
        def framebuffer_size_callback(window, w, h):
            self.framebuf_width, self.framebuf_height = w, h
            for callback in self.framebuffer_size_callback:
                callback(w,h)
        glfwSetFramebufferSizeCallback(self.windowID, framebuffer_size_callback)
        
        self.key_callback = []
        def key_callback(window, key, scancode, action, mode):
            if action == GLFW_PRESS:
                if key == GLFW_KEY_ESCAPE:
                    glfwSetWindowShouldClose(window, True)
            for callback in  self.key_callback:
                callback(key, scancode, action, mode)
        glfwSetKeyCallback(self.windowID, key_callback)

        self.mouse_callback = []
        def mouse_callback(window, xpos, ypos):
            for callback in self.mouse_callback:               
                callback(xpos, ypos)
        glfwSetCursorPosCallback(self.windowID, mouse_callback)

        self.mouse_button_callback = []
        def mouse_button_callback(window, button, action, mods):
            for callback in self.mouse_button_callback:
                callback(button, action, mods)
        glfwSetMouseButtonCallback(self.windowID, mouse_button_callback)

        self.scroll_callback = []
        def scroll_callback( window, xoffset, yoffset ):
            for callback in self.scroll_callback:
                callback(xoffset, yoffset)
        glfwSetScrollCallback(self.windowID, scroll_callback)

        self.previous_second = glfwGetTime()
        self.frame_count = 0.0

        if show_at_center:
            monitors = glfwGetMonitors()
            assert monitor >= 0 and monitor < len(monitors), 'Invalid monitor selected.'
            vidMode = glfwGetVideoMode(monitors[monitor])
            glfwSetWindowPos(self.windowID, 
                            vidMode.width/2-self.framebuf_width/2, 
                            vidMode.height/2-self.framebuf_height/2)

    def update_fps_counter(self):
        current_second = glfwGetTime()
        elapsed_seconds = current_second - self.previous_second
        if elapsed_seconds > 1.0:
            self.previous_second = current_second
            fps = float(self.frame_count) / float(elapsed_seconds)
            glfwSetWindowTitle(self.windowID, '%s @ FPS: %.2f' % (self.window_title, fps))
            self.frame_count = 0.0
        self.frame_count += 1.0

    def is_open(self):
        return not glfwWindowShouldClose(self.windowID)

    def swap_buffers(self):
        glfwSwapBuffers(self.windowID)

    def poll_events(self):
        glfwPollEvents()

    def update(self):
        self.swap_buffers()
        self.poll_events()
        self.update_fps_counter()    

    def close(self):
        glfwTerminate()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()