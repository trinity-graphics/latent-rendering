import mitsuba as mi

#mi.set_variant("cuda_ad_rgb")


class NegativeEmitter(mi.Emitter):
    """
        Pythonified version of the Point Light Emitter in Mitsuba, as it cannot be directly subclassed.
    """
    def __init__(self, props):
        mi.Emitter.__init__(self, props)
        self.m_emitter = props[props.keys()[0]]
        self.m_flags = self.m_emitter.flags()

    def traverse(self, cb):
        cb.put("m_emitter", self.m_emitter, mi.ParamFlags.Differentiable)
        if(not self.m_emitter.get_shape() and self.get_shape()):
            self.m_emitter.set_shape(self.get_shape())
        if(not self.m_emitter.get_medium() and self.get_medium()):
            self.m_emitter.set_medium(self.get_medium())
    
    def eval(self, si, active = True):
        return self.m_emitter.eval(si, active)
    
    def eval_direction(self, ref, ds, active = True):
        return (self.m_emitter.eval_direction(ref, ds, active))
    
    def pdf_direction(self, it, ds, active = True):
        return self.m_emitter.pdf_direction(it, ds, active)
    
    def pdf_position(self, ps, active = True):
        return self.m_emitter.pdf_position(ps, active)
    
    def sample_direction(self, it, sample, active=True):
        return self.m_emitter.sample_direction(it, sample, active)
    
    def sample_position(self, time, sample, active = True):
        return self.m_emitter.sample_position(time, sample, active)

    def sample_ray(self, time, sample1, sample2, sample3, active = True):
        return self.m_emitter.sample_ray(time, sample1, sample2, sample3, active)
    
    def sample_wavelengths(self, si, sample, active = True):
        return self.m_emitter.sample_wavelengths(si, sample, active)
    
    def bbox(self):
        return self.get_shape().bbox()

    def to_string(self):
        return "NegativeEmitter[\n" + \
            "  Emitter = " + str(self.m_emitter) + ",\n" + \
            "\n],"


class NegativeEmitterSplit(NegativeEmitter):
    """
        Pythonified version of the Point Light Emitter in Mitsuba, as it cannot be directly subclassed.
    """
    def __init__(self, props):
        super().__init__(props)
        self.m_neg_emitter = props[props.keys()[1]]

    def traverse(self, cb):
        super().traverse(cb)
        cb.put("m_neg_emitter", self.m_neg_emitter, mi.ParamFlags.Differentiable)
        if(not self.m_neg_emitter.get_shape() and self.get_shape()):
            self.m_neg_emitter.set_shape(self.get_shape())
        if(not self.m_neg_emitter.get_medium() and self.get_medium()):
            self.m_neg_emitter.set_medium(self.get_medium())
    
    def eval(self, si, active = True):
        return self.m_emitter.eval(si, active) - self.m_neg_emitter.eval(si, active)
    
    #TODO: Replace the sampling functions to sample from the bigger radiance value
    
    def eval_direction(self, ref, ds, active = True):
        return self.m_emitter.eval_direction(ref, ds, active) - self.m_neg_emitter.eval_direction(ref, ds, active)
    
    def sample_direction(self, it, sample, active=True):
        dirSamp, spec = self.m_emitter.sample_direction(it, sample, active)
        _, neg_spec = self.m_neg_emitter.sample_direction(it, sample, active)
        return dirSamp, spec - neg_spec

    def to_string(self):      # TODO: check whether to use to_string, __str__, or __repr__
        return "NegativeEmitter[\n" + \
            "  Emitter = " + str(self.m_emitter) + ",\n" + \
            "  Negative Emitter = " + str(self.m_neg_emitter) + ",\n" + \
            "\n],"