import mitsuba as mi

from custom_plugins.custom_bsdf import NegativeBSDF, NegativeBSDFSplit
from custom_plugins.custom_emitter import (
    NegativeEmitter,
    NegativeEmitterSplit,
)
from custom_plugins.custom_integrator import LatentIntegrator

_registered_plugins = set()
_registered_bsdf = []
_registered_integrator = []
_registered_emitter = []

def register_bsdfs(cfg):
    mi.register_bsdf("negative_bsdf", lambda props: NegativeBSDF(props))
    mi.register_bsdf('negative_bsdf_split', lambda props: NegativeBSDFSplit(props))
    
    if cfg["negative_rendering"]:
        if cfg["split_terms"]:
            _registered_plugins.add("negative_bsdf_split")
            _registered_bsdf.append("negative_bsdf_split")
        else:
            _registered_plugins.add("negative_bsdf")
            _registered_bsdf.append("negative_bsdf")


def register_integrators(cfg):
    mi.register_integrator(
        "latent_integrator",
        lambda props: LatentIntegrator(
            props,
            ambient_term=cfg["ambient_term"],
            separate_terms=cfg["seperate_term"],
        ),
    )

    if cfg["negative_rendering"]:
        _registered_plugins.add("latent_integrator")
        _registered_integrator.append("latent_integrator")
    else:
        _registered_plugins.add("prb")
        _registered_integrator.append("prb")
        # raise Exception("Not a valid set of custom rendering parameters")
        

def register_emitters(cfg):

    mi.register_emitter("negative_emitter", lambda props: NegativeEmitter(props))
    mi.register_emitter("negative_emitter_split", lambda props: NegativeEmitterSplit(props))

    if not cfg["negative_rendering"]:
        return

    if cfg["split_terms"]:
        _registered_plugins.add("negative_emitter_split")
        _registered_emitter.append("negative_emitter_split")
    else:
        _registered_plugins.add("negative_emitter")
        _registered_emitter.append("negative_emitter")


def register_plugins(cfg) -> None:
    register_bsdfs(cfg)
    register_integrators(cfg)
    register_emitters(cfg)

def has_custom_plugin(plugin: str) -> bool:
    return plugin in _registered_plugins

def registered_bsdf() -> str:
    if not _registered_bsdf:
        return None
    else:
        return _registered_bsdf[0]

def registered_integrator() -> str:
    if not _registered_integrator:
        return None
    else:
        return _registered_integrator[0]

def registered_emitter() -> str:
    if not _registered_emitter:
        return None
    else:
        return _registered_emitter[0]