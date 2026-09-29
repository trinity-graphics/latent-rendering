import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mitsuba as mi
import numpy as np

from utils import plugin_utils


def find_parent(root: ET.Element, child: ET.Element) -> ET.Element:
    for parent in list(root.iter()):
        for elem in parent:
            if elem is child:
                return parent
    return None


def read_spd_irregular(filename):
    wavelengths = []
    values = []
    with open(filename, "r") as spd:
        for line in spd:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            wl, val = line.split()
            wavelengths.append(str(int(float(wl))))
            values.append(f"{float(val):.10f}")
    return wavelengths, values


def create_irregular_spectrum(name, wavelengths, values):
    spectrum = ET.Element(
        "spectrum",
        attrib={
            "name": name,
            "type": "irregular",
        },
    )
    ET.SubElement(
        spectrum,
        "string",
        attrib={
            "name": "wavelengths",
            "value": ",".join(wavelengths),
        },
    )
    ET.SubElement(
        spectrum,
        "string",
        attrib={
            "name": "values",
            "value": ",".join(values),
        },
    )

    return spectrum


def initialize_latent(mode="uniform", scale=1, signed=False):
    n = int(mi.variant().split("latent")[1])

    if mode == "uniform":
        values = np.ones(n) * scale
    elif mode == "random":
        values = np.random.rand(n)
        if signed:
            values = values * 2 - 1
        values = values * scale
    else:
        raise ValueError(f"Invalid initialization mode `{mode}` passed.")

    inp_string = [f"{float(val):.10f}" for val in values]
    inp_string = ", ".join(inp_string)
    return inp_string


def initialize_rgb(mode="uniform", scale=1):
    n = 3

    if mode == "uniform":
        values = np.ones(n) * scale
    elif mode == "random":
        values = np.random.randn(n) * scale
    else:
        raise ValueError(f"Invalid initialization mode `{mode}` passed")

    inp_string = [f"{float(val):.10f}" for val in values]
    inp_string = ", ".join(inp_string)
    return inp_string


def get_impl_details(cfg, uniform_latent=False) -> dict:
    """
    Returns implementation details used for generating a latent XML file for the scene.

    :return: A dictionary containing variant and plugin implementation details. e.g. "emitter": "negative_emitter"
    :rtype: dict
    """
    impl = {}

    # Variant details
    mi_var = mi.variant()
    if "spectral" in mi_var:
        impl["variant"] = "spectral"
    elif "latent" in mi_var:
        impl["variant"] = "latent"
    else:
        raise NotImplementedError(
            "load_file_as_latent requires the use of a spectral or latent Mitsuba variant!"
        )

    # Plugin details
    impl["split_terms"] = cfg["split_terms"]
    impl["bsdf"] = plugin_utils.registered_bsdf()
    impl["ambient_term"] = cfg["ambient_term"]
    impl["occlusion_term"] = cfg["occlusion_term"]
    impl["emitter"] = plugin_utils.registered_emitter()
    impl["integrator"] = (
        plugin_utils.registered_integrator()
        if plugin_utils.registered_integrator() is not None
        else "prb"
    )

    # XML details
    impl["latent_init"] = "uniform" if uniform_latent else "random"
    impl["irregular_spectra"] = cfg["irregular_spec"]
    impl["uniform_spd"] = cfg["uniform_spd"]
    impl["duplicate_bsdfs"] = cfg["duplicate_bsdfs"]

    return impl


def get_latent_xml_fname(scene_file: Path, impl: dict):
    """
    Transforms a filename based on available variant/plugins.

    :param scene_file: Path to the XML scene description to be transformed.
    :type scene_file: Path
    :param impl: Dict of implementation details.
    :type impl: dict
    """

    # Create filename for the updated file
    aliases = {
        "negative_bsdf": "nBSDF",
        "negative_bsdf_split": "nBSDF_split",
        "custom_bsdf": "cBSDF",
        "negative_prototype": "pBSDF",
        "ambient_bsdf": "aBSDF",
        "negative_integrator": "nInt",
        "latent_integrator": "lInt",
        "negative_emitter": "nEm",
        "spectral": "spec",
        "latent": "lat",
    }

    new_fname = scene_file.stem

    # Add BSDF implementation to filename
    if impl["bsdf"] is not None:
        new_fname += f"_{aliases[impl['bsdf']]}"
    # Add variant to filename
    new_fname += f"_{impl['variant']}"

    return scene_file.with_stem(new_fname)


def load_file_as_latent(
    scene_file: Path | str,
    cfg,
    force_regenerate=False,
    **kwargs,
):
    r"""Read in a Mitsuba scene XML file in a way that's ready for latent rendering. It will:
    - Replace all the rgb radiance/reflectance with multichannel radiance/reflectance.  (Can be spectral or latent depending on the variant.)
    - Duplicate all BSDFs so they act as per-object materials.

    NOTE: This function only replaces the reflectance and radiance of scene objects, and does not affect film SRFs.
    Use the functions in sensor_utils for that.
    """
    scene_file = Path(scene_file)

    # Get implementation dictionary
    impl = get_impl_details(cfg, False)

    # Get implementation-specific scene file
    latent_fname = get_latent_xml_fname(scene_file, impl)

    # Short path: load existing file.
    # Parallel loading intermittently fails to load meshes in the latent variants, so load serially.
    if Path(latent_fname).exists() and not force_regenerate:
        return mi.load_file(str(latent_fname), optimize=False, parallel=False, **kwargs)

    # Long path: read original scene file and update with implementation-specific changes
    tree = ET.parse(scene_file)
    root = tree.getroot()

    assign_shape_ids(root)

    if impl["duplicate_bsdfs"]:
        duplicate_shared_bsdfs(root)

    if impl["emitter"] is not None and "negative_emitter" in impl["emitter"]:
        replace_emitters(root, impl)
    if impl["bsdf"] is not None:
        replace_bsdfs(root, impl)

    replace_textures(root, impl)

    replace_rgb(root, impl)

    replace_integrator(root, impl)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(latent_fname, encoding="utf-8")

    return mi.load_file(str(latent_fname), optimize=False, parallel=False, **kwargs)

def replace_integrator(root, impl):
    integrator_elem = root.find("integrator")
    if integrator_elem is not None:
        integrator_elem.set("type", impl["integrator"])
        integrator_elem.set("id", "integrator")
        if impl["variant"] == "latent" and impl["integrator"] == "latent_integrator":
            ET.SubElement(
                integrator_elem,
                "latent",
                {
                    "name": "miss_color",
                    "value": initialize_latent(
                        mode=impl["latent_init"], signed=not impl["split_terms"]
                    ),
                },
            )





def replace_rgb(root, impl):
    rgb_elems = root.findall(".//rgb")
    print(f"Need {len(rgb_elems)} SPDs for this scene.")

    for i, rgb in enumerate(rgb_elems):
        parent = find_parent(root, rgb)
        name = rgb.attrib["name"]

        if name == "eta" or name == "k":
            continue

        if impl["variant"] == "latent":
            # For latent variants: dynamically initialize
            latent = ET.Element(
                "latent",
                attrib={
                    "name": name,
                    "value": initialize_latent(mode=impl["latent_init"], signed=not impl["split_terms"]),
                },
            )
            parent.append(latent)
        elif impl["variant"] == "spectral":
            # For spectral variants: read [(wl, val), ...] from SPD files
            if impl["uniform_spd"]:
                spd_file = Path("spds") / "alternatives" / "uniform.spd"
            else:
                spd_file = Path("spds") / f"{i}.spd"
            if not spd_file.exists():
                raise FileNotFoundError(f"Missing .spd file at {spd_file}")

            wavelengths, values = read_spd_irregular(spd_file)

            # File-read spectra are automatically regarded as regular, even when
            # their values vary from wavelength to wavelength.  So we manually
            # generate irregular spectrum strings to circumvent this.
            if impl["irregular_spectra"]:
                spectrum = create_irregular_spectrum(name, wavelengths, values)
            else:
                spectrum = ET.Element(
                    "spectrum", attrib={"name": name, "filename": spd_file}
                )
            parent.append(spectrum)

        parent.remove(rgb)


def replace_bsdfs(root, impl):
    UNSUPPORTED_BSDFS = [
        "twosided",
        "mask",
        "blendbsdf",
        "normalmap",
        "bumpmap",
    ]
    SPECULAR_BSDFS = [
        "conductor",
        "dielectric",
        "thindielectric",
        "roughconductor",
        "roughdielectric",
        "plastic",
        "roughplastic",
    ]
    DIELECTRIC_BSDFS = [
        "dielectric",
        "thindielectric",
        "roughdielectric",
    ]

    bsdf_elems = root.findall(".//bsdf")
    print(f"Need {len(bsdf_elems)} custom BSDFs.")

    # Extra ambient terms
    extra_terms = []
    if impl["ambient_term"]:
        extra_terms.append("ambient")
        if impl["split_terms"]:
            extra_terms.append("neg_ambient")
    if impl["occlusion_term"]:
        extra_terms.append("occ_ambient")
        if impl["split_terms"]:
            extra_terms.append("neg_occ_ambient")

    # Replace BSDFs
    for i, bsdf in enumerate(bsdf_elems):
        if bsdf.get("type") in UNSUPPORTED_BSDFS:
            continue

        parent = find_parent(root, bsdf)

        # Variant-specific element attributes
        if impl["variant"] == "latent":
            elem_tag = "latent"
            elem_input_key = "value"

            def elem_input():
                return initialize_latent(mode=impl["latent_init"], scale=0.1, signed=not impl["split_terms"])
        elif impl["variant"] == "spectral":
            elem_tag = "spectrum"
            elem_input_key = "filename"

            def elem_input():
                return (
                    "spds/alternatives/uniform.spd"
                    if impl["uniform_spd"]
                    else f"spds/{i}.spd"
                )
        else:
            raise ValueError(
                f"Invalid Mitsuba variant `{impl['variant']}`.  Must be using spectral or latent variant."
            )

        # Create BSDF wrapper
        bsdf_spec = ET.Element(
            "bsdf",
            attrib={
                "type": impl["bsdf"],
            },
        )
        if bsdf.get("id") is not None:
            bsdf_spec.set("id", bsdf.get("id"))
            bsdf.set("id", "")

        # Add specular reflectance and transmittance to the bsdf file!
        bsdf_copy = deepcopy(bsdf)
        if bsdf.get("type") in SPECULAR_BSDFS:
            removal_list = [
                "material",
                "specular_reflectance",
                "specular_transmittance",
            ]
            for node in bsdf:
                if node.get("name") in removal_list:
                    bsdf.remove(node)
            if elem_tag == "latent":
                specular_refl = ET.Element(
                    elem_tag,
                    attrib={
                        "name": "specular_reflectance",
                        elem_input_key: initialize_latent("uniform"),
                    },
                )
                specular_tran = ET.Element(
                    elem_tag,
                    attrib={
                        "name": "specular_transmittance",
                        elem_input_key: initialize_latent("uniform"),
                    },
                )
                bsdf.append(specular_refl)
                if bsdf.get("type") in DIELECTRIC_BSDFS:
                    bsdf.append(specular_tran)
            else:
                specular_refl = ET.Element(
                    elem_tag,
                    attrib={
                        "name": "specular_reflectance",
                        elem_input_key: elem_input(),
                    },
                )
                specular_tran = ET.Element(
                    elem_tag,
                    attrib={
                        "name": "specular_transmittance",
                        elem_input_key: elem_input(),
                    },
                )
                bsdf.append(specular_refl)
                if bsdf.get("type") in DIELECTRIC_BSDFS:
                    bsdf.append(specular_tran)

            for node in bsdf_copy:
                if node.get("name") in removal_list:
                    bsdf_copy.remove(node)
            if elem_tag == "latent":
                specular_refl_0 = ET.Element(
                    elem_tag,
                    attrib={
                        "name": "specular_reflectance",
                        elem_input_key: initialize_latent("uniform", 0),
                    },
                )
                specular_tran_0 = ET.Element(
                    elem_tag,
                    attrib={
                        "name": "specular_transmittance",
                        elem_input_key: initialize_latent("uniform", 0),
                    },
                )
                bsdf_copy.append(specular_refl_0)
                if bsdf.get("type") in DIELECTRIC_BSDFS:
                    bsdf_copy.append(specular_tran_0)
            else:
                specular_refl = ET.Element(
                    elem_tag,
                    attrib={
                        "name": "specular_reflectance",
                        elem_input_key: elem_input(),
                    },
                )
                specular_tran = ET.Element(
                    elem_tag,
                    attrib={
                        "name": "specular_transmittance",
                        elem_input_key: elem_input(),
                    },
                )
                bsdf_copy.append(specular_refl)
                if bsdf.get("type") in DIELECTRIC_BSDFS:
                    bsdf_copy.append(specular_tran)

        # Add base BSDF
        bsdf_spec.append(bsdf)

        # (neg prototype only): add negative BSDF
        if impl["split_terms"]:
            bsdf_spec.append(bsdf_copy)

        # Add extra terms
        for prop in extra_terms:
            p = ET.Element(
                elem_tag, attrib={"name": prop, elem_input_key: elem_input()}
            )
            bsdf_spec.append(p)

        # Replace BSDF
        parent.remove(bsdf)
        parent.append(bsdf_spec)


def replace_textures(root: ET.Element, impl: dict) -> None:
    _TEXTURE_REFL_NAMES = {
        "diffuse_reflectance",
        "reflectance",
        "specular_reflectance",
        "specular_transmittance",
        "base_color"
    }

    if impl["variant"] != "latent":
        return

    for tex in list(root.findall('.//texture[@type="bitmap"]')):
        tex_name = tex.get("name", "")
        if tex_name not in _TEXTURE_REFL_NAMES:
            continue

        parent = find_parent(root, tex)
        if parent is None:
            continue

        parent.remove(tex)
        tex.set("name", "texture")

        wrapper = ET.Element(
            "texture",
            attrib={"type": "lattex", "name": tex_name},
        )
        wrapper.append(tex)

        ET.SubElement(
            wrapper,
            "latent",
            attrib={
                "name": "tex_scale",
                "value": initialize_latent(mode=impl["latent_init"], signed=not impl["split_terms"]),
            },
        )
        ET.SubElement(
            wrapper,
            "latent",
            attrib={
                "name": "tex_offset",
                "value": initialize_latent(mode=impl["latent_init"], signed=not impl["split_terms"]),
            },
        )
        parent.append(wrapper)



def replace_emitters(root, impl):
    emit_elems = root.findall(".//emitter")
    print(f"Need {len(emit_elems)} custom emitters.")

    for emitter in emit_elems:
        parent = find_parent(root, emitter)

        em_type = impl["emitter"]

        # Create negative wrapper emitter
        latent_emitter = ET.Element(
            "emitter",
            attrib={
                "type": em_type,
            },
        )
        if emitter.get("id") is not None:
            latent_emitter.set("id", emitter.get("id"))
            emitter.set("id", "")

        # Duplicate the original emitter inside the wrapper
        parent.remove(emitter)
        latent_emitter.append(emitter)  # Positive
        if impl["split_terms"]:
            latent_emitter.append(deepcopy(emitter))  # Negative
        parent.append(latent_emitter)


def assign_shape_ids(root: ET.Element) -> None:
    counters = {}
    for shape in root.findall("shape"):
        if shape.get("id") is not None:
            continue
        shape_type = shape.get("type", "shape")
        counters[shape_type] = counters.get(shape_type, 0) + 1
        shape.set("id", f"{shape_type}{counters[shape_type]:04d}")


def duplicate_shared_bsdfs(root: ET.Element):
    # Get {ID: BSDF} dictionary
    bsdfs = {}
    for bsdf in root.findall(".//bsdf"):
        bsdf_id = bsdf.get("id")
        if bsdf_id is not None:
            bsdfs[bsdf_id] = bsdf

    # Get {ID: [ref1, ref2, ...]} dictionary
    bsdf_refs = {}
    for shape in root.findall(".//shape"):
        ref = shape.find("ref")
        if ref is not None:
            ref_id = ref.get("id")
            bsdf_refs.setdefault(ref_id, []).append(ref)

    # Duplicate BSDFs
    for bsdf_id, refs in bsdf_refs.items():
        # Skip single-use BSDFs
        if len(refs) <= 1:
            continue

        # Insert duplicates after original BSDF
        original_bsdf = bsdfs[bsdf_id]
        original_idx = list(root).index(original_bsdf)

        # One duplicate for each reference
        for i, ref in enumerate(refs):
            new_bsdf_id = f"{bsdf_id}_{i + 1}"

            # Create and insert duplicate BSDF
            new_bsdf = deepcopy(original_bsdf)
            new_bsdf.set("id", new_bsdf_id)
            root.insert(original_idx + i, new_bsdf)

            # Update object to reference duplicate BSDF
            ref.set("id", new_bsdf_id)

        # Remove the original BSDF
        root.remove(original_bsdf)


def _latent_key_map(root: ET.Element) -> dict:
    """Map each ``<latent>`` element to the Mitsuba parameter key it backs.

    Reconstructs the keys ``mi.traverse`` exposes for this scene family so the
    optimized values can be written back into the matching elements:

    - ``integrator.<name>``                       (e.g. ``integrator.miss_color``)
    - ``<shape>.bsdf.<name>.value``               (negative_bsdf terms: ambient, ...)
    - ``<shape>.bsdf.m_bsdf.<name>.value``         (the wrapped inner BSDF)
    - ``<shape>.emitter.m_emitter.<name>.value``   (negative_emitter radiance)
    """
    top_bsdfs = {b.get("id"): b for b in root.findall("bsdf") if b.get("id")}

    def resolve_bsdf(shape):
        inline = shape.find("bsdf")
        if inline is not None:
            return inline
        ref = shape.find("ref")
        return top_bsdfs.get(ref.get("id")) if ref is not None else None

    keymap = {}

    integ = root.find("integrator")
    if integ is not None:
        for lat in integ.findall("latent"):
            keymap[f"{integ.get('id')}.{lat.get('name')}"] = lat

    for shape in root.findall("shape"):
        sid = shape.get("id")
        bsdf = resolve_bsdf(shape)
        if bsdf is not None:
            for lat in bsdf.findall("latent"):
                keymap[f"{sid}.bsdf.{lat.get('name')}.value"] = lat
            inner = bsdf.find("bsdf")
            if inner is not None:
                for lat in inner.findall("latent"):
                    keymap[f"{sid}.bsdf.m_bsdf.{lat.get('name')}.value"] = lat
        em = shape.find("emitter")
        if em is not None:
            inner_em = em.find("emitter")
            if inner_em is not None:
                for lat in inner_em.findall("latent"):
                    keymap[f"{sid}.emitter.m_emitter.{lat.get('name')}.value"] = lat

    return keymap


def write_optimized_scene(src_xml: Path | str, dst_xml: Path | str, opt_params: dict):
    """Write a copy of the generated latent scene with optimized values baked in.

    ``opt_params`` maps Mitsuba parameter keys to 1-D sequences of latent values
    (the *raw* optimized ``scene_opt_params``; the pipeline re-applies any
    per-term scaling at load via ``get_scaled_opt_params``). Each value overwrites
    the ``value`` attribute of its matching ``<latent>`` element. Keys without a
    matching element are returned so the caller can warn.
    """
    src_xml, dst_xml = Path(src_xml), Path(dst_xml)
    tree = ET.parse(src_xml)
    root = tree.getroot()
    keymap = _latent_key_map(root)

    unmatched = []
    for key, values in opt_params.items():
        elem = keymap.get(key)
        if elem is None:
            unmatched.append(key)
            continue
        elem.set("value", ", ".join(f"{float(v):.10f}" for v in values))

    ET.indent(tree, space="    ")
    tree.write(dst_xml, encoding="utf-8")
    return unmatched