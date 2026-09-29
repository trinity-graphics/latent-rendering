import drjit as dr
import mitsuba as mi


class NegativeBSDF(mi.BSDF):
    def __init__(self, props):
        super().__init__(props)
        # ==================================================
        #   Get wrapped BSDF
        # ==================================================
        self.m_bsdf = props[props.keys()[0]]

        # ==================================================
        #   Prepare optional terms
        # ==================================================
        if "spectral" in mi.variant():
            self.prop_init_fn = self.props_spectral
        elif "latent" in mi.variant():
            self.prop_init_fn = self.props_latent
        else:
            raise RuntimeError(
                "CustomBSDF requires the use of a spectral or latent Mitsuba variant!"
            )
        
        self.ambient_term = None
        self.occluded_term = None
        if props.has_property("ambient"):
            self.ambient_term = self.prop_init_fn(props["ambient"])
        if props.has_property("occ_ambient"):
            self.occluded_term = self.prop_init_fn(props["occ_ambient"])

        # ==================================================
        #   Set BSDF Flags
        # ==================================================
        reflection_flags = self.m_bsdf.flags()
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags

    def props_spectral(self, props_spd):
        wav = ", ".join(str(x) for x in props_spd.wavelengths)
        vals = ", ".join(str(x) for x in props_spd.values)
        irreg_spect = mi.load_dict(
            {"type": "irregular", "wavelengths": wav, "values": vals}
        )
        return irreg_spect

    def props_latent(self, props_lat):
        lat_spect = mi.load_dict({"type": "latent", "value": props_lat})
        return lat_spect

    def traverse(self, cb):
        cb.put("m_bsdf", self.m_bsdf, mi.ParamFlags.Differentiable)
        if self.ambient_term is not None:
            cb.put("ambient", self.ambient_term, mi.ParamFlags.Differentiable)
        if self.occluded_term is not None:
            cb.put("occ_ambient", self.occluded_term, mi.ParamFlags.Differentiable)

    def sample(self, ctx, si, sample1, sample2, active=True):
        return self.m_bsdf.sample(ctx, si, sample1, sample2, active)

    def eval(self, ctx, si, wo, active=True):
        return self.m_bsdf.eval(ctx, si, wo, active)

    def pdf(self, ctx, si, wo, active=True):
        return self.m_bsdf.pdf(ctx, si, wo, active)

    def eval_pdf(self, ctx, si, wo, active=True):
        return self.m_bsdf.eval_pdf(ctx, si, wo, active)

    def eval_pdf_sample(self, ctx, si, wo, sample1, sample2, active=True):
        return self.m_bsdf.eval_pdf_sample(ctx, si, wo, sample1, sample2, active)

    def eval_null_transmission(self, si, active=True):
        return self.m_bsdf.eval_null_transmission(si, active)

    def eval_diffuse_reflectance(self, si, active=True):
        return self.m_bsdf.eval_diffuse_reflectance(si, active)

    def has_attribute(self, name, active=True):
        if active and self.ambient_term is not None and name == "ambient" or active and self.occluded_term is not None and name == "occ_ambient":
            return True
        else:
            return self.m_bsdf.has_attribute(name, active)
    
    def eval_attribute(self, name, si, active=True):
        if name == "ambient":
            if self.ambient_term is None:
                return 0
            return mi.depolarizer(self.ambient_term.eval(si, active)) & active
        elif name == "occ_ambient":
            if self.occluded_term is None:
                return 0
            return mi.depolarizer(self.occluded_term.eval(si, active)) & active
        else:
            return self.m_bsdf.eval_attribute(name, si, active)

    def eval_attribute_1(self, name, si, active=True):
        return self.m_bsdf.eval_attribute_1(name, si, active)

    def eval_attribute_3(self, name, si, active=True):
        return self.m_bsdf.eval_attribute_3(name, si, active)

    def to_string(self):
        s = "BSDFWrapper\n"
        s += f"\tBSDF={self.m_bsdf},\n"
        if self.ambient_term is not None:
            s += f"\tAmbient={self.ambient_term},\n"
        if self.occluded_term is not None:
            s += f"\tOccludedAmbient={self.occluded_term},\n"
        s += "]"
        return s


 
# Negative Prototype
class NegativeBSDFSplit(mi.BSDF):
    def __init__(self, props):
        super().__init__(props)

        # ==================================================
        #   Get wrapped BSDFs
        # ==================================================
        self.m_bsdf = props[props.keys()[0]]
        self.m_bsdf_neg = props[props.keys()[1]]

        # ==================================================
        #   Prepare optional terms
        # ==================================================
        if "spectral" in mi.variant():
            self.prop_init_fn = self.props_spect
        elif "latent" in mi.variant():
            self.prop_init_fn = self.props_latent
        else:
            raise RuntimeError(
                "NegativeBSDF requires the use of a spectral or latent Mitsuba variant!"
            )
        
        self.ambient_term = None
        self.neg_ambient_term = None
        self.occluded_term = None
        self.neg_occluded_term = None
        if props.has_property("ambient") and props.has_property("neg_ambient"):
            self.ambient_term = self.prop_init_fn(props["ambient"])
            self.neg_ambient_term = self.prop_init_fn(props["neg_ambient"])
        if props.has_property("occ_ambient") and props.has_property("neg_occ_ambient"):
            self.occluded_term = self.prop_init_fn(props["occ_ambient"])
            self.neg_occluded_term = self.prop_init_fn(props["neg_occ_ambient"])

        # ==================================================
        #   Set BSDF Flags
        # ==================================================
        reflection_flags = self.m_bsdf.flags()
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags

    def props_spect(self, props_spd):
        wav = ", ".join(str(x) for x in props_spd.wavelengths)
        vals = ", ".join(str(x) for x in props_spd.values)
        irreg_spect = mi.load_dict(
            {"type": "irregular", "wavelengths": wav, "values": vals}
        )
        return irreg_spect

    def props_latent(self, props_lat):
        lat_spect = mi.load_dict({"type": "latent", "value": props_lat})
        return lat_spect

    def traverse(self, cb):
        cb.put('m_bsdf', self.m_bsdf, mi.ParamFlags.Differentiable)
        cb.put('m_bsdf_neg', self.m_bsdf_neg, mi.ParamFlags.Differentiable)
        if self.ambient_term is not None and self.neg_ambient_term is not None:
            cb.put("ambient", self.ambient_term, mi.ParamFlags.Differentiable)
            cb.put("neg_ambient", self.neg_ambient_term, mi.ParamFlags.Differentiable)
        if self.occluded_term is not None and self.neg_occluded_term is not None:
            cb.put('occ_ambient', self.occluded_term, mi.ParamFlags.Differentiable)
            cb.put('neg_occ_ambient', self.neg_occluded_term, mi.ParamFlags.Differentiable)

    def sample(self, ctx, si, sample1, sample2, active = True):
        keep_neg = dr.select(sample1 < 0.5, mi.Bool(True), mi.Bool(False))
        new_sample = dr.select(sample1 < 0.5, sample1 * 2.0, (sample1 - 0.5) * 2.0)

        bs_pos, weight_pos = self.m_bsdf.sample(ctx, si, new_sample, sample2, active)
        bs_neg, weight_neg = self.m_bsdf_neg.sample(ctx, si, new_sample, sample2, active)

        # ASSUMING That the samples going in the sample function are the same
        # That is the seed is the same to sample
        # That is wo for both bs_pos and bs_neg is the same
        # That is the weight value caluclated is calculated for the same angle
        # and the pdf calculated is also the pdf for the same angle
        bs = dr.select(keep_neg, bs_neg, bs_pos)
        bs.pdf = 0.5 * bs_pos.pdf + 0.5 * bs_neg.pdf

        weight = weight_pos * bs_pos.pdf - weight_neg * bs_neg.pdf
        weight = weight / bs.pdf

        return bs, weight

    def eval(self, ctx, si, wo, active=True):
        return self.m_bsdf.eval(ctx, si, wo, active) - self.m_bsdf_neg.eval(ctx, si, wo, active)

    def pdf(self, ctx, si, wo, active=True):
        pdf = 0.5 * self.m_bsdf.pdf(ctx, si, wo, active) + \
            0.5 * self.m_bsdf_neg.pdf(ctx, si, wo, active)
        return pdf

    def eval_pdf(self, ctx, si, wo, active=True):
        v_pos, pdf_pos = self.m_bsdf.eval_pdf(ctx, si, wo, active)
        v_neg, pdf_neg = self.m_bsdf_neg.eval_pdf(ctx, si, wo, active)
        return v_pos - v_neg, 0.5 * pdf_neg + 0.5 * pdf_pos

    def eval_pdf_sample(self, ctx, si, wo, sample1, sample2, active=True):
        # return self.m_bsdf.eval_pdf_sample(ctx, si, wo, sample1, sample2, active)
        # Cannot do this directly since we have two BSDFs
        keep_neg = dr.select(sample1 < 0.5, mi.Bool(True), mi.Bool(False))
        new_sample = dr.select(sample1 < 0.5, sample1 * 2.0, (sample1 - 0.5) * 2.0)

        v_pos, pdf_pos, sample_pos, w_pos = self.m_bsdf.eval_pdf_sample(
            ctx, si, wo, new_sample, sample2, active
        )
        v_neg, pdf_neg, sample_neg, w_neg = self.m_bsdf_neg.eval_pdf_sample(
            ctx, si, wo, new_sample, sample2, active
        )

        sample = dr.select(keep_neg, sample_neg, sample_pos)
        sample.pdf = self.pdf(ctx, si, sample.wo, active)  # Mixture PDF
        weight = self.eval(ctx, si, sample.wo, active) / sample.pdf

        return v_pos - v_neg, sample.pdf, sample, weight

    def eval_null_transmission(self, si, active=True):
        return self.m_bsdf.eval_null_transmission(si, active)

    def eval_diffuse_reflectance(self, si, active=True):
        return self.m_bsdf.eval_diffuse_reflectance(si, active) -  \
            self.m_bsdf_neg.eval_diffuse_reflectance(si, active)

    def has_attribute(self, name, active=True):
        if active \
            and self.ambient_term is not None \
            and self.neg_ambient_term is not None \
            and name == "ambient" or active \
            and self.occluded_term is not None \
            and self.neg_occluded_term is not None \
            and name == "occ_ambient":
            return True
        else:
            return self.m_bsdf.has_attribute(name, active)
        
    def eval_attribute(self, name, si, active=True):
        eval_fns = {
            'ambient':      lambda s, a: self.ambient_term.eval(s, a) - self.neg_ambient_term.eval(s, a),
            'occ_ambient':  lambda s, a: self.occluded_term.eval(s, a) - self.neg_occluded_term.eval(s, a),
        }
        if name == "ambient":
            if self.ambient_term is None or self.neg_ambient_term is None:
                return 0
            return mi.depolarizer(eval_fns[name](si, active)) & active
        elif name == "occ_ambient":
            if self.occluded_term is None or self.neg_occluded_term is None:
                return 0
            return mi.depolarizer(eval_fns[name](si, active)) & active
        else:
            return self.m_bsdf.eval_attribute(name, si, active)

    def eval_attribute_1(self, name, si, active=True):
        return self.m_bsdf.eval_attribute_1(name, si, active)

    def eval_attribute_3(self, name, si, active=True):
        return self.m_bsdf.eval_attribute_3(name, si, active)

    def to_string(self):
        s = "BSDFWrapperSplit\n"
        s += f"\tBSDF={self.m_bsdf},\n"
        s += f"\tNegative BSDF={self.m_bsdf_neg},\n"
        if self.ambient_term is not None and self.neg_ambient_term is not None:
            s += f"\tAmbient={self.ambient_term},\n"
            s += f"\tNegative Ambient={self.neg_ambient_term},\n"
        if self.occluded_term is not None and self.neg_occluded_term is not None:
            s += f"\tOccluded Ambient={self.occluded_term},\n"
            s += f"\tNegative Occluded Ambient={self.neg_occluded_term},\n"
        s += "]"
        return s