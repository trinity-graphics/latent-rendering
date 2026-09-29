import mitsuba

if mitsuba.variant():
    from .manager import RendererManager